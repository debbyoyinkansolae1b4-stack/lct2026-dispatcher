"""Популяционный поиск: цикл эпох — одна машина для сети b и её потомка U2n (PORT-PLAN.md).

Эпоха: граф плана-родителя → проход политики → population кандидатов (действие,
мутация, проверка) → награда-ступенька и шаг обучения (у сети) → рекорд по cost_tuple →
отбор родителя по кортежу. Лучший допустимый план хранится отдельно от текущего и
возвращается всегда, даже если отбор ушёл в ухудшение.

Режимы выбора хода: learned — сеть b (одноразовая, учится под задачу), uniform — потомок U2n
(правило: равномерно, «куда бить» частью по потенциалу). Отличия режимов — только настройки
SearchConfig; пресеты — portfolio.PRESETS.
"""
# refactor: Claude, 21.09.2026 — из main() experiment_population_gnn.py (шаг 4)
# refactor: Claude, 21.09.2026 — шаг 5б: нативное восстановление и после события
# refactor: Claude, 22.09.2026 — Н39: лимит эпох и горизонт расписания раздельно
# refactor: Claude, 22.09.2026 — Н41: история эпох по флагу keep_history
# перенос: Claude, 26.09.2026 — машина сети b и U2n; энергия, подсказки и старые режимы удалены
import itertools
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field

import torch

from src.engine.compiled import CompiledInstance
from src.engine.repair import NativeRepair, RouteCache
from src.search.golden_trace import golden_trace, plan_digest, tensor_digest
from src.search.acceptance import anneal_tuple
from src.search.actions import CHAINS
from src.search.operators import NOISE, mutate
from src.search.policy.gnn.graph import enrich_graph, graph
from src.search.policy.gnn.weighting import balanced_batch
from src.plan.validate import validate

# Порог и шаг отбора кэша — как в прогонах принятой базы.
CACHE_LIMIT = 15000
MODES = ('learned', 'uniform')
INSERTIONS = ('six', 'product')
# Без повторов: предел перевыборов одинакового хода (дёшево) и перемутаций копии родителя (дорого) —
# страховка от зацикливания.
DEDUP_SAMPLES, DEDUP_REMUTATIONS = 64, 2


def signature(plan):
    """Какие заявки у кого и в каком порядке: отличает копию родителя от изменения."""
    return tuple(sorted((eid, tuple(s.request_id for s in r.stops)) for eid, r in plan.routes.items()))


@dataclass(frozen=True)
class SearchConfig:
    """Настройки поиска. Остановка: по epochs (лимит; None — без лимита), по seconds (время),
    по patience (эпох без рекорда) — что наступит раньше. Температуру отбора задаёт горизонт:
    schedule_epochs, а если он не задан — epochs (Н39: лимит и горизонт раздельно).
    Значения по умолчанию — сеть b; потомок U2n — пресет portfolio.PRESETS['uniform']."""
    population: int = 64
    epochs: int | None = 500
    schedule_epochs: int | None = None
    seconds: float = math.inf
    patience: int | None = None
    mode: str = 'learned'
    seed: int = 29
    chain_max: int = 3
    chain_learned: bool = True
    normalize_features: bool = True     # нормировка входа сети по задаче (gnn/normalization.py)
    rehomable_feature: bool = True
    hidden: int = 256
    experts: int = 3
    lr: float = .001
    dedup: bool = True                  # без повторов: одинаковый ход и копия родителя перевыбираются (сеть b)
    insertion: str = 'six'              # 'six' — 6 вставок сети b; 'product' — продуктовая, +цена открытия бригады (U2n)
    potential: bool = True              # потенциал в графе: признаки сети b, прицел U2n
    target_potential: float = 0.        # uniform: доля «куда бить» пропорционально потенциалу (U2n — 0.5)
    open_cost: float = 300.             # продуктовая вставка: сколько км стоит открыть новую бригаду
    defer_explanations: bool = True
    repair_noise: float = NOISE     # шум вставки при восстановлении; после события адаптер задаёт свой
    keep_history: bool = True       # строки эпох в SearchResult.history; приложению не нужны (Н41)

    def __post_init__(self):
        integer = lambda v: isinstance(v, int) and not isinstance(v, bool)
        number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        problems = []
        for name, low in (('population', 2), ('chain_max', 1), ('hidden', 1), ('experts', 1)):
            if not integer(getattr(self, name)) or getattr(self, name) < low:
                problems.append(f'{name} — целое ≥ {low}')
        for name in ('epochs', 'schedule_epochs'):
            value = getattr(self, name)
            if value is not None and (not integer(value) or value < 1):
                problems.append(f'{name} — целое ≥ 1 или None')
        if self.epochs is None and self.schedule_epochs is None:
            problems.append('без лимита эпох нужен горизонт расписания schedule_epochs')
        if self.epochs is None and number(self.seconds) and math.isinf(self.seconds):
            problems.append('без лимита эпох нужен конечный seconds')
        if not integer(self.seed):
            problems.append('seed — целое')
        if self.patience is not None and (not integer(self.patience) or self.patience < 1):
            problems.append('patience — целое ≥ 1 или None')
        if not number(self.seconds) or math.isnan(self.seconds) or self.seconds <= 0:
            problems.append('seconds > 0 (допустима бесконечность)')
        if not number(self.repair_noise) or not math.isfinite(self.repair_noise) or not 0 <= self.repair_noise < 1:
            problems.append('repair_noise — конечное в [0, 1)')
        if not number(self.lr) or not math.isfinite(self.lr) or self.lr <= 0:
            problems.append('lr — конечное > 0')
        if self.mode not in MODES:
            problems.append(f'mode — один из {MODES}')
        if self.insertion not in INSERTIONS:
            problems.append(f'insertion — один из {INSERTIONS}')
        for name in ('dedup', 'potential'):
            if not isinstance(getattr(self, name), bool):
                problems.append(f'{name} — да или нет')
        if not number(self.open_cost) or not math.isfinite(self.open_cost) or self.open_cost < 0:
            problems.append('open_cost — конечное ≥ 0 (км)')
        if not number(self.target_potential) or not 0 <= self.target_potential <= 1:
            problems.append('target_potential — доля в [0, 1]')
        if self.target_potential and not self.potential:
            problems.append('прицел по потенциалу требует potential')
        if self.chain_learned and integer(self.chain_max) and self.chain_max > max(CHAINS):
            problems.append(f'chain_max ≤ {max(CHAINS)} при chain_learned: шире головы цепочки нет')
        if problems:
            raise ValueError('SearchConfig: ' + '; '.join(problems))

    @property
    def horizon(self):
        """Горизонт расписания: температура отбора считается как доля epoch / horizon."""
        return self.schedule_epochs if self.schedule_epochs is not None else self.epochs


class Clock:
    """Часы поиска: только для решения «пора ли остановиться». Время в отчёте меряется отдельно."""

    def now(self):
        return time.perf_counter()


class TickClock(Clock):
    """Управляемые часы для тестов: каждое обращение — ровно одна «секунда»."""

    def __init__(self):
        self.ticks = 0

    def now(self):
        self.ticks += 1
        return float(self.ticks - 1)


@dataclass
class SearchResult:
    best: object
    current: object
    best_epoch: int
    stop_reason: str
    epochs_completed: int
    candidates: int
    seconds: float
    history: list = field(default_factory=list)
    transferred: dict | None = None
    policy: object = None


def make_policy(inst, start, config, trace=False, cache=None):
    """uniform — политика без сети (U2n); learned — сеть b.

    cache — сервис маршрута поиска: разогрев сети идёт через него же."""
    if config.mode == 'uniform':
        from src.search.policy.netless import NetlessPolicy
        return NetlessPolicy(config)
    from src.search.policy.gnn.policy import GnnPolicy
    return GnnPolicy(inst, start, config, trace=trace, cache=cache)


# Награда-ступенька (Михаил 24.09): награда кандидата — ступенька первой строки кортежа, где он отличается от
# родителя. Главное (строки 1–5) — фиксированная ступенька: аварии ±7/7, подключения ±6/7, назначено ±5/7, аварии сверх
# ориентира ±4/7, исполнители ±3/7. Стоимость дня — полоса ±(1/7 … 2/7), ожидание — полоса ±(0 … 1/7); внутри полосы —
# tanh(|Δ| / масштаб эпохи), не больше 0.99 (полосы не перекрываются — порядок цели держится без весов). Масштаб — 90-й
# перцентиль |Δ| строки по уникальным кандидатам, равным родителю во всех строках выше, делённый на atanh(0.9); пол —
# STEP_FLOORS (км, минуты). Проверенные альтернативы — ранги и единая полоса — хуже на участках (RESULTS 26.09).
STRICT_ROWS = 5
STEP_FLOORS = {5: .5, 6: 1.}


def _band_scale(costs, plans, parent_cost, k):
    unique = {}
    for c, p in zip(costs, plans):
        if tuple(c[:k]) == tuple(parent_cost[:k]) and c[k] != parent_cost[k]:
            unique.setdefault(signature(p), abs(c[k] - parent_cost[k]))
    if not unique:
        return STEP_FLOORS[k]
    q = float(torch.quantile(torch.tensor(list(unique.values()), dtype=torch.float64), .9))
    return max(STEP_FLOORS[k], q / math.atanh(.9))


def step_credit(costs, plans, parent_cost):
    """Награды-ступеньки кандидатов (тензор) и масштабы полос {строка: масштаб}."""
    scales = {k: _band_scale(costs, plans, parent_cost, k) for k in (5, 6)}
    out = []
    for c in costs:
        k = next((i for i, (a, b) in enumerate(zip(parent_cost, c)) if a != b), None)
        if k is None:
            out.append(0.)
            continue
        sign = 1. if c[k] < parent_cost[k] else -1.
        if k < STRICT_ROWS:
            out.append(sign * (7 - k) / 7)
        else:
            inside = min(math.tanh(abs(c[k] - parent_cost[k]) / scales[k]), .99)
            out.append(sign * ((1. if k == 5 else 0.) + inside) / 7)
    return torch.tensor(out, dtype=torch.float32), scales


def run(inst, start, config, policy=None, clock=None, sink=None, trace=False):
    """Поиск от допустимого стартового плана. Возвращает SearchResult с лучшим допустимым планом.

    Воспроизводимость побайтовая при одном потоке torch в процессе (torch.set_num_threads(1)
    ставит входная точка процесса): с несколькими потоками градиенты LayerNorm расходятся
    в младших битах.

    policy — готовая политика (по умолчанию GnnPolicy по config.mode); clock — часы
    остановки; sink(row) — получатель строки журнала каждой эпохи; trace — дописывать
    в строку трассу эталона шага 0.
    """
    check = validate(start, inst)
    if not check['ok']:
        broken = {c['code']: c['violations'][:3] for c in check['checks'] if c['violations']}
        raise ValueError(f'Недопустимый стартовый план: {broken}')
    clock = clock or Clock()
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    numpy_start = None
    if trace:
        import numpy as np
        numpy_start = repr(np.random.get_state())
    current = best = start
    begin, deadline = time.perf_counter(), clock.now() + config.seconds
    # Сервис маршрута на весь запуск: снимок истории (RouteState), колонки вставок, кэш
    # маршрутов. Задача после события своя, поэтому снимок и кэш новые (Н12, Н29).
    cache = NativeRepair(inst, compiled=CompiledInstance(inst), limit=CACHE_LIMIT,
                         defer_reasons=config.defer_explanations, product_insertion=config.insertion == 'product',
                         open_cost=config.open_cost)
    state = cache.state
    if policy is None:
        policy = make_policy(inst, start, config, trace, cache=cache)

    stall = epochs_without_record = best_epoch = 0
    history, candidates_total, stop_reason, epochs_completed = [], 0, 'epoch_limit', 0
    dead = False                                   # мёртвая зона: прошлая эпоха ничем не отличилась от родителя
    for epoch in (range(config.epochs) if config.epochs is not None else itertools.count()):
        if clock.now() >= deadline:
            stop_reason = 'time_limit'
            break
        # Место подключения: история задачи не менялась с начала запуска (раз в эпоху).
        state.check(inst)
        g = graph(inst, current, epoch, config.horizon, stall, penalty_hints=True, state=state)
        g = enrich_graph(inst, current, g, cache, rehomable=config.rehomable_feature, potential=config.potential)
        # Мёртвая зона: прошлая эпоха не дала ни одного отличия от родителя — сигнала нет, эта эпоха ходит
        # равномерно (исследование), пока не появится разница. У uniform выбор и так равномерный.
        g['explore_all'] = dead
        out = policy.forward(g)

        parent_cost, parent_plan, parent_signature = current.cost_tuple(), current, signature(current)
        traces, plans, mutation_traces, costs, noops, seeds = [], [], [], [], [], []
        seen_actions = set()                                  # без повторов: сочетания хода этой эпохи
        for _ in range(config.population):
            if clock.now() >= deadline:
                break
            samples = remutations = 0
            while True:
                step = policy.sample(out, g, rng)
                key = json.dumps(step.choice, sort_keys=True, default=lambda x: int(x) if not isinstance(x, bool) else x)
                if config.dedup and key in seen_actions and samples < DEDUP_SAMPLES:
                    samples += 1
                    continue
                seen_actions.add(key)
                seed = rng.randrange(2 ** 31)
                candidate, mutation = mutate(inst, current, step.action, seed, constructor=cache, state=state,
                                             noise=config.repair_noise)
                if config.dedup and signature(candidate) == parent_signature and remutations < DEDUP_REMUTATIONS:
                    remutations += 1
                    continue
                break
            seeds.append(seed)
            verdict = validate(candidate, inst)
            if not verdict['ok']:
                raise RuntimeError(str(verdict))
            traces.append(step)
            plans.append(candidate)
            mutation_traces.append(mutation)
            costs.append(candidate.cost_tuple())
            noops.append(signature(candidate) == parent_signature)
        # Пустая популяция — только когда дедлайн наступил до первого кандидата.
        if not plans:
            stop_reason = 'time_limit'
            break
        candidates_total += len(plans)
        actions = [t.action for t in traces]

        rewards, scale = step_credit(costs, plans, parent_cost)
        sample_weights, adv, batch_stats = balanced_batch([signature(p) for p in plans], actions, rewards, noops)
        golden_targets = [] if trace else None
        dead = all(tuple(c) == tuple(parent_cost) for c in costs)
        update = policy.update(out, traces, rewards, adv, sample_weights, golden_targets)

        record = min(plans, key=lambda p: p.cost_tuple())
        # Лечение Н76: кандидат в родители — лучший из ИЗМЕНЁННЫХ планов (копии родителя не участвуют), принять ли
        # его — решает отбор по кортежу. Иначе копия родителя всегда «лучшая» и план стоит.
        changed = [i for i in range(len(plans)) if not noops[i]]
        candidate = min((plans[i] for i in changed), key=lambda p: p.cost_tuple()) if changed else record
        record_improved = record.cost_tuple() < best.cost_tuple()
        if record_improved:
            best = record
            stall = epochs_without_record = 0
            best_epoch = epoch + 1
        else:
            stall += 1
            epochs_without_record += 1
        accepted, acceptance = anneal_tuple(current.cost_tuple(), candidate.cost_tuple(), epoch, config.horizon,
                                            costs, rng)
        if accepted:
            current = candidate

        row = dict(
            best_epoch=best_epoch, epochs_without_record=epochs_without_record, training_batch=batch_stats,
            graph_stats=g.get('graph_stats', {}), reward_scale=scale,
            reward_saturation=float((rewards.abs() > .99).float().mean()), acceptance=acceptance, epoch=epoch,
            mode=config.mode, elapsed=time.perf_counter() - begin, population=len(plans),
            full_population=len(plans) == config.population, unique_plans=len({signature(p) for p in plans}),
            interroute_moves=0, unique_genotypes=len({json.dumps(x, sort_keys=True) for x in actions}),
            noop_fraction=sum(noops) / len(noops), distinct_objectives=len(set(costs)),
            reward_std=float(rewards.std(unbiased=False)),
            positive_noop_advantages=sum(bool(n and v > 0) for n, v in zip(noops, adv)),
            parent=list(parent_cost), improving_candidates=sum(c < parent_cost for c in costs),
            best=list(best.cost_tuple()), current=list(current.cost_tuple()), record_improved=record_improved,
            best_km=round(best.total_km, 2), best_engineers=best.used_engineers, best_assigned=best.assigned_count,
            escape=False, loss=update.diagnostics,
            gates=(dict(actor=out['gate_action'].detach().tolist(), critic=out['gate_value'].detach().tolist())
                   if 'gate_action' in out else {}),
            cache_stats=dict(cache.stats) if cache else {},
            coverage={key: dict(Counter(str(x[key]) for x in actions if key in x))
                      for key in ('operator', 'engineer', 'route', 'request', 'scale', 'repair', 'chain',
                                  'link0_op', 'link1_op')},
            candidates=[dict(action=x, trace=t, cost=list(c), noop=n, reward=float(r))
                        for x, t, c, n, r in zip(actions, mutation_traces, costs, noops, rewards)])
        if trace:
            row['golden'] = golden_trace(rng, policy.net, policy.opt, seeds, numpy_start, dict(
                candidate_plans=[plan_digest(p) for p in plans],
                epoch_plans=dict(parent=plan_digest(parent_plan), current=plan_digest(current),
                                 best=plan_digest(best)),
                graph=tensor_digest(g), outputs=tensor_digest(out),
                sampling=tensor_digest(dict(logps=[t.logp for t in traces], entropies=[t.entropy for t in traces])),
                credit=tensor_digest(dict(rewards=rewards, adv=adv, sample_weights=sample_weights)),
                targets=tensor_digest(golden_targets), gradients=update.gradients))
        if sink:
            sink(row)
        epochs_completed += 1
        if config.keep_history:
            history.append({k: v for k, v in row.items() if k != 'candidates'})
        # Партия, оборванная дедлайном, — остановка по времени, даже если это последняя
        # эпоха или в ней же исчерпано терпение: неполной партию сделало именно время.
        if len(plans) < config.population:
            stop_reason = 'time_limit'
            break
        if config.patience and epochs_without_record >= config.patience:
            stop_reason = 'patience'
            break

    if config.defer_explanations:
        # Объяснения неназначенных — свежим кэшем, как в прогонах принятой базы.
        RouteCache(inst).materialize(best)
        RouteCache(inst).materialize(current)
    assert validate(best, inst)['ok'], 'лучший план недопустим'
    return SearchResult(best=best, current=current, best_epoch=best_epoch, stop_reason=stop_reason,
                        epochs_completed=epochs_completed, candidates=candidates_total,
                        seconds=time.perf_counter() - begin, history=history,
                        transferred=getattr(policy, 'transferred', None), policy=policy)
