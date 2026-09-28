"""Портфель: несколько независимых поисков одновременно, ответ — лучший план по cost_tuple.

Каждый участник — популяционный поиск (population.run) со своим сидом и режимом
(learned — сеть b, одноразовая под задачу; uniform — её потомок U2n без сети) в отдельном
рабочем процессе. Режимы — пресеты одной машины поиска (PRESETS). Замер 20.09 (DECISIONS.md): три сида по 150 эпох обыграли один прогон
на 450 эпох 10/0/2 — портфель пробует разные траектории, длинный прогон углубляет одну.

Устройство (шаг 6 REFACTOR-PLAN.md):
- рабочие процессы запускаются заранее и живут между расчётами (spawn, один поток
  torch): импорт torch и загрузка ядер numba не входят в бюджет расчёта;
- критерий — один снимок весов (Weights), прочитанный координатором и переданный
  всем участникам (Н9): правка в админке действует со следующего расчёта и не может
  попасть посреди него; внутри участника кэш создаётся под этот снимок;
- таймаут и аварийное завершение участника не роняют расчёт: ответ — лучший из
  успевших, в отчёте — причина сбоя; процессы после сбоя пересоздаются;
- ответ проверяется валидатором и не хуже стартового плана.

Контракт: портфель — последовательный компонент, один вызов за раз (run, prepare, close).
Своих замков у него нет; приложение сериализует все расчёты и замену пула одним замком
(app.COMPUTE), CLI вызывает по очереди сам.
"""
# refactor: Claude, 22.09.2026 — шаг 6а
# refactor: Claude, 22.09.2026 — поправки Codex к 6а: Н36–Н40, остановка процессов, очередь расчётов
# refactor: Claude, 22.09.2026 — шаг 6б: статистика операторов участника для отчёта
# refactor: Claude, 22.09.2026 — Н47: прогрев каждого поколения процессов; процессорное время участника
# refactor: Claude, 22.09.2026 — упрощение: портфель последовательный, замок и очередь — в приложении; один прогрев
# refactor: Claude, 22.09.2026 — Н51: память по физическому следу, отказ по доле всей памяти, иначе предупреждение
# перенос: Claude, 26.09.2026 — пресеты сети b и U2n; веса энергии и подсказки удалены (PORT-PLAN.md)
import concurrent.futures as futures
import math
import multiprocessing
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace

from src.search.operators import NOISE
from src.search.population import SearchConfig

# Отступ на передачу задачи, построение кэша и сборку ответа сверх бюджета поиска.
GRACE_SECONDS = 5.
# Горизонт расписания поиска (температура отжига, вес подсказок) — как в замерах базы
# и в перепланировании. Это не лимит: граница расчёта — время (Н39).
SCHEDULE_EPOCHS = 500
# Предел ожидания прогрева процесса: холодная компиляция ядер и загрузка сети (замер 1.3 с).
WARMUP_TIMEOUT = 120.
# Пиковая память рабочего процесса без буферов восстановления, по режимам: пик физического
# следа (phys_footprint, как в мониторе системы), а не RSS — на macOS RSS держит освобождённые
# malloc страницы, которые система заберёт по требованию (у learned пик RSS 1.5 ГБ при следе
# 0.6 ГБ). Замер 22.09 (artifacts/n50/memory_footprint.py), 20 с поиска после прогрева:
# uniform 339–473 МБ (верх — процесс, прогретый learned), learned 565 МБ на Юго-востоке (83×12)
# и 564 МБ без буферов на roomy (300×40); в приложении (портфель по умолчанию, Юго-восток, 20 с)
# uniform 476–495, learned 616–630 МБ. Здесь с запасом ~20% от наибольших.
# Буферы NativeRepair растут как n·m·(n+1) и считаются по задаче (Н38, Н14).
BASE_PEAK_MB = {'uniform': 600, 'learned': 760}
# Пресеты режимов одной машины поиска (замеры 25–26.09, RESULTS.md, research/night26/GLOSSARY.md):
# learned — сеть b: без повторов, 6 вставок сети, потенциал — признак на входе;
# uniform — потомок U2n: без сети, продуктовая вставка, «куда бить» наполовину по потенциалу, без повторов.
PRESETS = {
    'learned': dict(dedup=True, insertion='six', potential=True, target_potential=0., open_cost=300.),
    'uniform': dict(dedup=False, insertion='product', potential=True, target_potential=.5, open_cost=300.),
}
# Отказ, если портфель после расчёта займёт больше этой доли всей памяти машины (Н51, решение
# Дебби 22.09). Больше «свободной» по psutil — только предупреждение: на macOS она занижена
# (сжатие, вытесняемый кэш).
MEMORY_SHARE = .7


@dataclass(frozen=True)
class Member:
    """Участник портфеля (поток): режим выбора действий, сид, параметры поиска и сети (для learned).
    Параметры поиска у каждого свои (решение Михаила 23.09); по умолчанию — принятая база."""
    mode: str = 'uniform'
    seed: int = 29
    hidden: int = 256
    experts: int = 3
    lr: float = .001
    population: int = 64                    # кандидатов (мутаций) за эпоху
    chain_max: int = 3                      # разрушений в цепочке до одного восстановления
    schedule_epochs: int = SCHEDULE_EPOCHS  # горизонт расписания: отжиг и доля подсказок
    repair_noise: float | None = None       # шум вставки; None — как у расчёта (план 0.15, событие 0)
    chain_learned: bool = True              # длину цепочки выбирает голова (иначе равномерно)
    rehomable_feature: bool = True          # признак «заявку можно переселить» в графе
    normalize_features: bool = True         # нормировка входа сети по задаче
    # Настройки машины поиска; None — как в пресете режима (PRESETS).
    dedup: bool | None = None               # без повторов
    insertion: str | None = None            # 'six' — вставки сети b; 'product' — продуктовая (+цена открытия бригады)
    target_potential: float | None = None   # uniform: доля «куда бить» пропорционально потенциалу
    open_cost: float | None = None          # продуктовая вставка: цена открытия новой бригады, км

    def preset(self):
        """Настройки машины поиска участника: пресет режима с его уточнениями."""
        values = dict(PRESETS.get(self.mode, PRESETS['learned']))
        for name in ('dedup', 'insertion', 'target_potential', 'open_cost'):
            if getattr(self, name) is not None:
                values[name] = getattr(self, name)
        return values


@dataclass(frozen=True)
class Weights:
    """Снимок весов критерия на один расчёт (config/weights.json)."""
    unassigned_request: float
    engineer_used: float
    urgent_delay_minute: float
    open_engineer_penalty: float
    urgent_delay_mode: str = 'linear'     # хвост цели: 'linear' или 'square' (solver.delay_cost)
    urgent_delay_square: float = 0.
    urgent_target_first: bool = False       # компонента «аварии сверх ориентира» (solver.URGENT_TARGET_FIRST)
    urgent_target_minutes: float = 120.
    event_move_km: float = 0.               # цена переназначения после события (solver.W_EVENT_MOVE)

    @classmethod
    def read(cls):
        from src import settings
        values = settings.read_weights()['values']
        return cls(float(values['unassigned_request']), float(values['engineer_used']),
                   float(values['urgent_delay_minute']),
                   float(values.get('open_engineer_penalty', values['engineer_used'])),
                   values.get('urgent_delay_mode', 'linear'), float(values.get('urgent_delay_square', 0.)),
                   values.get('urgent_target_first', False), float(values.get('urgent_target_minutes', 120.)),
                   float(values.get('event_move_km', 0.)))

    def problems(self):
        """Что не так со снимком: веса — конечные неотрицательные числа (Н37), режим задержки известен."""
        out = []
        for name, value in asdict(self).items():
            if name == 'urgent_delay_mode':
                if value not in ('linear', 'square'):
                    out.append(f'{name} = {value!r}')
            elif name == 'urgent_target_first':
                if not isinstance(value, bool):
                    out.append(f'{name} = {value!r}')
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                out.append(f'{name} = {value!r}')
        return out

    def apply(self):
        """Выставить веса в решателе этого процесса (рабочего — на время одного расчёта)."""
        from src.plan import solver
        solver.W_UNASSIGNED = self.unassigned_request
        solver.W_ENGINEER = self.engineer_used
        solver.W_URGENT_DELAY = self.urgent_delay_minute
        solver.OPEN_ENGINEER_PENALTY = self.open_engineer_penalty
        solver.URGENT_DELAY_MODE = self.urgent_delay_mode
        solver.W_URGENT_DELAY_SQUARE = self.urgent_delay_square
        solver.URGENT_TARGET_FIRST = self.urgent_target_first
        solver.URGENT_TARGET_MIN = self.urgent_target_minutes
        solver.W_EVENT_MOVE = self.event_move_km


@dataclass
class MemberReport:
    """Что сделал участник: итог или причина сбоя."""
    member: Member
    ok: bool
    error: str | None = None
    cost: tuple | None = None
    stop_reason: str | None = None
    epochs: int = 0
    best_epoch: int = 0
    candidates: int = 0
    seconds: float = 0.
    peak_memory_mb: float | None = None  # пик процесса с его запуска, не только этого расчёта
    cpu_seconds: float = 0.             # процессорное время участника (process_time в рабочем процессе)
    pid: int | None = None
    wins: dict = field(default_factory=dict)   # операторы: кандидаты лучше родителя и рекорды
    progress: list = field(default_factory=list)   # по циклу: цикл, секунды, заявок, бригад, км лучшего плана

    def as_dict(self):
        return {**asdict(self), 'member': asdict(self.member)}


@dataclass
class PortfolioResult:
    best: object                    # лучший допустимый план
    winner: int | None              # номер участника с лучшим планом; None — стартовый план
    reports: list = field(default_factory=list)
    seconds: float = 0.
    prepare_seconds: float = 0.     # прогрев новых процессов перед расчётом (вне бюджета поиска)
    memory_warning: str | None = None   # памяти нужно больше «свободной» (Н51); расчёт шёл
    weights: Weights | None = None
    urgent_front_moves: int | None = None   # постобработка «аварии вперёд»: принятых ходов; None — выключена


def search_config(member, seconds, patience=None, repair_noise=NOISE):
    """Настройки поиска участника: пресет режима (PRESETS) с уточнениями участника, сид — от участника.

    Предела эпох нет: расчёт идёт до времени (или до patience эпох без рекорда).
    """
    # История эпох приложению не нужна: отчёт — из счётчиков OperatorWins (Н41).
    return SearchConfig(epochs=None, schedule_epochs=member.schedule_epochs, seconds=seconds, keep_history=False,
                        patience=patience, mode=member.mode, seed=member.seed, hidden=member.hidden,
                        experts=member.experts, lr=member.lr, population=member.population,
                        chain_max=member.chain_max, chain_learned=member.chain_learned,
                        rehomable_feature=member.rehomable_feature, normalize_features=member.normalize_features,
                        repair_noise=repair_noise if member.repair_noise is None else member.repair_noise,
                        **member.preset())


def run_member(inst, start, config, weights):
    """Задача рабочего процесса: один поиск под снимком весов. Возвращает (план, отчёт)."""
    weights.apply()
    from src.search.population import run
    from src.search.report import OperatorWins
    wins = OperatorWins()
    cpu = time.process_time()
    result = run(inst, start, config, sink=wins)
    return result.best, dict(cost=result.best.cost_tuple(), stop_reason=result.stop_reason,
                             epochs=result.epochs_completed, best_epoch=result.best_epoch,
                             candidates=result.candidates, seconds=result.seconds, peak_memory_mb=peak_memory_mb(),
                             wins=wins.summary(), progress=wins.progress, cpu_seconds=time.process_time() - cpu,
                             pid=os.getpid())


def warm_member(inst, start, config, weights, hold=.5):
    """Прогрев: одна эпоха поиска, затем процесс держится занятым — так каждое задание
    прогрева уходит в свой процесс, а не в тот, что освободился первым."""
    plan, summary = run_member(inst, start, config, weights)
    time.sleep(hold)
    return plan, summary


def _footprint(pid):
    """Физический след процесса на macOS (текущий и пик с запуска), байты; None — не прочитан."""
    import ctypes

    class Info(ctypes.Structure):      # rusage_info_v4: uuid, затем поля uint64
        _fields_ = [('uuid', ctypes.c_uint8 * 16), ('fields', ctypes.c_uint64 * 64)]
    info = Info()
    if ctypes.CDLL('/usr/lib/libSystem.B.dylib').proc_pid_rusage(pid, 4, ctypes.byref(info)) != 0:
        return None
    return info.fields[7], info.fields[28]    # ri_phys_footprint, ri_lifetime_max_phys_footprint


def process_memory_mb(pid):
    """Сколько памяти держит процесс: на macOS — физический след, иначе RSS."""
    if sys.platform == 'darwin':
        footprint = _footprint(pid)
        if footprint is not None:
            return footprint[0] / 2 ** 20
    import psutil
    return psutil.Process(pid).memory_info().rss / 2 ** 20


def peak_memory_mb():
    """Пик памяти процесса с его запуска (на macOS — физический след, иначе ru_maxrss): рабочий
    процесс живёт между расчётами, поэтому это пик за все его расчёты, а не текущий расход."""
    if sys.platform == 'darwin':
        footprint = _footprint(os.getpid())
        if footprint is not None:
            return footprint[1] / 2 ** 20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == 'darwin' else peak / 2 ** 10


def member_memory_mb(inst, member):
    """Оценка пиковой памяти участника на этой задаче: база режима + буферы восстановления."""
    n, m = len(inst.requests), len(inst.engineers)
    buffers = (2 * n * m * (n + 1) + n * m) * 8 / 2 ** 20
    return BASE_PEAK_MB[member.mode] + buffers


def _worker_init():
    """Прогрев рабочего процесса: один поток torch (побайтовая воспроизводимость), импорты."""
    import torch
    torch.set_num_threads(1)
    import src.search.population  # noqa: F401 — torch, numba-ядра и сеть загружаются здесь, а не в расчёте


def default_workers():
    """Сколько участников одновременно: ядра без одного на приложение, не больше четырёх."""
    return max(1, min(4, (os.cpu_count() or 2) - 1))


class Portfolio:
    """Пул рабочих процессов и расчёт портфелем. Один экземпляр на приложение."""

    def __init__(self, workers=None):
        if workers is None:
            workers = default_workers()
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError('Число рабочих процессов — целое ≥ 1')
        self.workers = workers
        limit = os.cpu_count() or 1
        if self.workers > limit:
            raise ValueError(f'Поисков одновременно {self.workers}, а ядер {limit}: '
                             f'поиски будут делить ядра и каждый успеет меньше')
        self._pool = None
        # Поколение процессов растёт при каждом создании пула (в том числе после сбоя);
        # прогрев привязан к поколению (Н47).
        self._generation = 0
        self._warmed = (-1, set())       # (поколение, прогретые режимы)
        self._warmup = None

    def _executor(self):
        if self._pool is None:
            self._pool = futures.ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'),
                                                     initializer=_worker_init)
            self._generation += 1
        return self._pool

    def set_warmup(self, inst, start):
        """Задача для прогрева: на ней каждый новый процесс выполнит короткий поиск learned."""
        self._warmup = (inst, start)

    def _ready(self, mode):
        """Прогрето ли текущее поколение для режима: learned покрывает и uniform (те же ядра)."""
        generation, modes = self._warmed
        return self._pool is not None and generation == self._generation and (mode in modes or 'learned' in modes)

    def warm_members(self, modes):
        """Участники прогрева для режимов расчёта: сеть греется, только если она нужна."""
        mode = 'learned' if 'learned' in modes else 'uniform'
        return [Member(mode, 0)] * self.workers

    def prepare(self, modes=('learned',), rounds=3):
        """Прогреть текущее поколение процессов для режимов modes: в каждом PID — одна полная
        эпоха поиска (ядра numba; для learned — сеть, прямой и обратный проход). Граница прогрева —
        эпоха, а не время: холодная инициализация не может съесть его целиком (Н47).
        Возвращает секунды прогрева; 0 — уже прогрето или задачи для прогрева нет."""
        mode = self.warm_members(modes)[0].mode
        if self._warmup is None or self._ready(mode):
            return 0.
        begin = time.perf_counter()
        inst, start = self._warmup
        weights = Weights.read()
        config = replace(search_config(Member(mode, 0), WARMUP_TIMEOUT), epochs=1, population=4)
        seen = set()
        for _ in range(rounds):
            pool = self._executor()
            jobs = [pool.submit(warm_member, inst, start, config, weights) for _ in range(self.workers)]
            done, pending = futures.wait(jobs, timeout=WARMUP_TIMEOUT)
            if pending:
                self._restart()
                raise RuntimeError('Прогрев рабочих процессов не уложился во время')
            for job in jobs:
                summary = job.result()[1]                  # ошибка прогрева поднимется здесь
                if summary.get('epochs', 0) < 1 or summary.get('candidates', 0) < 1:
                    raise RuntimeError(f'Прогрев процесса {summary.get("pid")}: ни одной эпохи поиска')
                seen.add(summary['pid'])
            if len(seen) >= self.workers:
                generation, modes_done = self._warmed
                self._warmed = (self._generation, (modes_done if generation == self._generation else set()) | {mode})
                return time.perf_counter() - begin
        raise RuntimeError(f'Прогрелось {len(seen)} рабочих процессов из {self.workers}')

    def processes(self):
        """Живые процессы пула. Единственное место, где читается приватное поле
        ProcessPoolExecutor._processes (CPython 3.10–3.13): у пула нет публичного списка."""
        pool = self._pool
        return list(getattr(pool, '_processes', None).values()) if pool is not None else []

    def _restart(self, wait=2.):
        """После таймаута или падения: остановить процессы и дождаться их смерти.

        Зависший участник сам не завершится: terminate, ограниченное ожидание, затем kill
        для не завершившихся. Следующий расчёт поднимет новые процессы.
        """
        processes, pool, self._pool = self.processes(), self._pool, None
        if pool is None:
            return
        for process in processes:
            process.terminate()
        for process in processes:
            process.join(wait)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(wait)
        pool.shutdown(wait=False, cancel_futures=True)
        alive = [p.pid for p in processes if p.is_alive()]
        if alive:
            raise RuntimeError(f'Рабочие процессы не остановились: {alive}')

    def check_memory(self, inst, members):
        """Хватит ли памяти на участников этой задачи: оценка по размеру задачи и режиму.

        Уже занятое процессом засчитывается только тому участнику, который в нём будет
        считать, и не больше его потребности; простаивающие процессы свою память держат (Н38).
        Какой процесс возьмёт какое задание, заранее неизвестно, поэтому оценка осторожная:
        самые требовательные участники — в паре с самыми «пустыми» процессами.

        Отказ (ValueError) — портфель займёт больше MEMORY_SHARE всей памяти машины. Больше
        «свободной» — возвращает предупреждение для отчёта, расчёт идёт (Н51); иначе None.
        """
        try:
            import psutil
        except ImportError:
            return None
        needs = sorted((member_memory_mb(inst, m) for m in members), reverse=True)
        held = []
        for process in self.processes():
            try:
                held.append(process_memory_mb(process.pid))
            except psutil.Error:
                held.append(0.)
        # Нули ещё не созданных процессов — до сортировки: они идут в пару первыми (Н38).
        held = sorted(held + [0.] * max(0, len(needs) - len(held)))
        extra = sum(max(0., need - have) for need, have in zip(needs, held))
        memory = psutil.virtual_memory()
        total, available = memory.total / 2 ** 20, memory.available / 2 ** 20
        if sum(held) + extra > MEMORY_SHARE * total:
            raise ValueError(f'{len(members)} параллельных поисков на этой задаче займут до '
                             f'{sum(held) + extra:.0f} МБ — больше {MEMORY_SHARE:.0%} памяти машины '
                             f'({total:.0f} МБ): уменьшите состав портфеля')
        if extra > available:
            return (f'Расчёту нужно ещё до {extra:.0f} МБ, а свободно {available:.0f} МБ: возможны '
                    f'вытеснение памяти в своп и замедление расчёта')
        return None

    def close(self):
        """Закрыть пул. По контракту — не во время расчёта (вызывающий сериализует)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def run(self, inst, start, members, seconds, patience=None, repair_noise=NOISE, weights=None,
            task=run_member, urgent_front=False):
        """Лучший план портфеля за seconds секунд поиска. Не хуже start; сбои участников — в отчёте.
        urgent_front — после поиска постобработка «аварии вперёд» (src/search/urgent_front.py), до 1 с.

        Всё проверяется до первого задания (Н36, Н37): стартовый план, состав, время,
        настройки каждого участника, снимок весов, память.
        """
        from src.plan.validate import validate
        if not members:
            raise ValueError('Нет ни одного поиска')
        if len(members) > self.workers:
            raise ValueError(f'Поисков {len(members)}, а рабочих процессов {self.workers}: '
                             f'уменьшите состав портфеля или увеличьте число процессов в настройках')
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) \
                or seconds <= 0:
            raise ValueError('Время расчёта портфеля — конечное положительное число секунд')
        check = validate(start, inst)
        if not check['ok']:
            broken = {c['code']: c['violations'][:3] for c in check['checks'] if c['violations']}
            raise ValueError(f'Недопустимый стартовый план: {broken}')
        configs = [search_config(m, seconds, patience, repair_noise) for m in members]
        weights = weights or Weights.read()
        if not isinstance(weights, Weights):
            raise ValueError('Снимок весов — Weights')
        if weights.problems():
            raise ValueError('Веса критерия — конечные неотрицательные числа: ' + ', '.join(weights.problems()))
        modes = {m.mode for m in members}
        warnings = []                          # (стадия, текст): прогрев и расчёт — в одном отчёте
        if self._warmup is not None and not self._ready(self.warm_members(modes)[0].mode):
            # Память на прогрев — до его запуска: иначе learned занял бы все процессы (Н47).
            warnings.append(('Прогрев', self.check_memory(self._warmup[0], self.warm_members(modes))))
        prepared = self.prepare(modes)         # новые процессы прогреваются вне бюджета поиска
        begin = time.perf_counter()
        warnings.append(('Расчёт', self.check_memory(inst, members)))
        result = self._run(inst, start, members, configs, seconds, weights, task, begin)
        if urgent_front:
            from src.search.urgent_front import urgent_front as finish
            weights.apply()          # цель координатора — та же, что у участников: сравнение по одному снимку
            result.best, result.urgent_front_moves = finish(inst, result.best)
        result.prepare_seconds = prepared
        result.memory_warning = ' '.join(f'{stage}: {text}' for stage, text in warnings if text) or None
        return result

    def _run(self, inst, start, members, configs, seconds, weights, task, begin):
        from src.plan.validate import validate
        pool = self._executor()
        jobs = []
        try:
            for config in configs:
                jobs.append(pool.submit(task, inst, start, config, weights))
        except Exception:
            # Частичная отправка: уже отправленные задания не должны остаться в пуле.
            self._restart()
            raise
        futures.wait(jobs, timeout=seconds + GRACE_SECONDS)
        reports, plans, broken = [], [], False
        for member, job in zip(members, jobs):
            if not job.done():
                reports.append(MemberReport(member, ok=False, error='timeout'))
                broken = True
                continue
            try:
                plan, summary = job.result()
            except futures.process.BrokenProcessPool as exc:
                reports.append(MemberReport(member, ok=False, error=f'процесс завершился аварийно: {exc}'))
                broken = True
                continue
            except Exception as exc:        # ошибка внутри поиска участника — не роняет портфель
                reports.append(MemberReport(member, ok=False, error=f'{type(exc).__name__}: {exc}'))
                continue
            reports.append(MemberReport(member, ok=True, **summary))
            plans.append((len(reports) - 1, plan))
        if broken:
            self._restart()
        best, winner = start, None
        for index, plan in plans:
            if not validate(plan, inst)['ok']:
                reports[index].ok, reports[index].error = False, 'недопустимый план'
                continue
            if plan.cost_tuple() < best.cost_tuple():
                best, winner = plan, index
        return PortfolioResult(best=best, winner=winner, reports=reports,
                               seconds=time.perf_counter() - begin, weights=weights)
