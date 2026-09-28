"""Построение и улучшение плана.

Три уровня, все три нужны для сдачи:
  baseline_greedy — базовый вариант, заданный в ТЗ дословно (раздел 2.3), для сравнения;
  construct       — жадная вставка с regret-2, стартовое допустимое решение;
  alns            — ruin & recreate поверх конструктора, с отжигом и сменной политикой
                    выбора операторов (адаптивные веса или обученная — см. policy.py).

Критерий сравнения планов — лексикографический, как задал постановщик [33:18]:
сначала больше назначенных, потом меньше исполнителей, потом меньше пробег.
Для отжига та же иерархия свёрнута в скаляр с большими коэффициентами.
"""

import json
import math
import random
from functools import partial
from datetime import timedelta
from pathlib import Path

from src.instance import Instance
from src.plan.resources import equipment_reason
from src.model import Plan, Route, Unassigned
from src.plan.fastroute import build_view, insertion_delta
from src.plan.scheduling import SKILL_TITLES, TRANSPORT_TITLES, day_start, evaluate, static_reason, urgent_delays

# Приоритеты планирования живут в config/weights.json: их правит администратор,
# и они выражены в условных километрах — так видно курс обмена между критериями.
_WEIGHTS_FILE = Path(__file__).resolve().parents[2] / "config" / "weights.json"


def load_weights() -> dict:
    """Читает приоритеты из конфига. Вызывается при старте и после правки в админке."""
    global W_UNASSIGNED, W_ENGINEER, W_URGENT_DELAY, OPEN_ENGINEER_PENALTY, URGENT_DELAY_MODE, W_URGENT_DELAY_SQUARE, \
        URGENT_TARGET_FIRST, URGENT_TARGET_MIN, W_EVENT_MOVE
    cfg = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8"))
    W_UNASSIGNED = float(cfg["unassigned_request"])
    W_ENGINEER = float(cfg["engineer_used"])
    W_URGENT_DELAY = float(cfg["urgent_delay_minute"])
    OPEN_ENGINEER_PENALTY = float(cfg.get("open_engineer_penalty", W_ENGINEER))
    URGENT_DELAY_MODE = cfg.get("urgent_delay_mode", "linear")
    W_URGENT_DELAY_SQUARE = float(cfg.get("urgent_delay_square", 0.0))
    URGENT_TARGET_FIRST = bool(cfg.get("urgent_target_first", False))
    URGENT_TARGET_MIN = float(cfg.get("urgent_target_minutes", 120.0))
    W_EVENT_MOVE = float(cfg.get("event_move_km", 0.0))
    return cfg


W_UNASSIGNED = 10_000.0     # одна неназначенная дороже любого пробега
W_ENGINEER = 300.0          # лишний исполнитель дороже десятков километров
W_URGENT_DELAY = 0.02       # аварию просили делать как можно раньше [15:54]
OPEN_ENGINEER_PENALTY = 300.0
# Цена задержки аварий в хвосте цели (км + цена задержек): линейно — W_URGENT_DELAY за минуту,
# квадратично — W_URGENT_DELAY_SQUARE за минуту² каждой аварии отдельно (одна долгая задержка
# дороже многих коротких). Линейный режим — для сравнения с решателями с линейной стоимостью.
URGENT_DELAY_MODE = "linear"
W_URGENT_DELAY_SQUARE = 0.0
# Аварии сверх ориентира реакции (эксперты 22.09, message644: «1–2 часа») — компонента цели выше
# числа исполнителей: включённая, она разрешает вывести бригаду, если та вводит аварию в ориентир.
# Выключена — компонента 0, цель прежняя (решение Михаила 23.09).
URGENT_TARGET_FIRST = False
URGENT_TARGET_MIN = 120.0
# Цена переназначения после события, км за заявку, сменившую исполнителя (в хвосте цели): перестройка после
# аварии не переставляет полдня ради километров. 0 — выключено.
W_EVENT_MOVE = 0.0
load_weights()
# Патч learned_full (Михаил 25.09, «победить +300»): открыть новую бригаду при вставке — только если заявку некуда
# поставить в работающие (порядок строк цели: бригады выше км). OPEN_LAST — не цена, а лексикографический сдвиг:
# больше любого возможного прироста км. «Свободная» вставка (NativeRepair, mode=1) открывает без сдвига.
OPEN_LAST = 1e6


def delay_cost(delays) -> float:
    """Цена задержек срочных заявок в условных км по действующему режиму."""
    if URGENT_DELAY_MODE == "square":
        return W_URGENT_DELAY_SQUARE * sum(d * d for d in delays)
    return W_URGENT_DELAY * sum(delays)


def plan_scalar(plan: Plan, inst: Instance) -> float:
    delays = [d for r in plan.routes.values() for d in urgent_delays(r, inst.by_id, day_start(inst))]
    plan.urgent_delay_min = sum(delays)
    plan.delay_cost = delay_cost(delays)
    plan.urgent_late = sum(d > URGENT_TARGET_MIN for d in delays) if URGENT_TARGET_FIRST else 0
    owners = getattr(inst, "previous_owner", None)       # у задач из старых наборов поля нет
    if W_EVENT_MOVE and owners:
        moved = sum(1 for eid, r in plan.routes.items() for s in r.stops if owners.get(s.request_id, eid) != eid)
        plan.delay_cost += W_EVENT_MOVE * moved         # хвост цели: км + цена задержек + цена переназначений
    assigned_requests = [inst.by_id[s.request_id] for route in plan.routes.values() for s in route.stops]
    plan.priority_counts = (sum(r.priority == 2 for r in assigned_requests),
                            sum(r.priority == 1 for r in assigned_requests))
    radix = len(inst.by_id) + 1
    unassigned = sum(radix ** inst.by_id[u.request_id].priority for u in plan.unassigned)
    return (W_UNASSIGNED * unassigned + W_ENGINEER * plan.used_engineers
            + plan.total_km + plan.delay_cost)


def empty_plan(inst: Instance) -> Plan:
    return Plan(routes={e.id: Route(engineer_id=e.id) for e in inst.engineers})


def _orders(plan: Plan, inst: Instance) -> dict[str, list]:
    return {eid: [inst.by_id[s.request_id] for s in r.stops] for eid, r in plan.routes.items()}


def _rebuild(inst: Instance, orders: dict[str, list], unassigned: list[Unassigned],
             *, evaluate_route=evaluate) -> Plan:
    """evaluate_route — расчёт маршрута; кэш популяции подставляет свой, запоминающий."""
    routes = {}
    for eng in inst.engineers:
        route, bad = evaluate_route(eng, orders.get(eng.id, []), inst.matrix, inst.index, earliest=inst.earliest_start, exempt=inst.earliest_exempt, committed=inst.committed.get(eng.id, []))
        if route is None:      # не должно случаться: порядок собираем только из допустимых
            raise RuntimeError(f"Недопустимый маршрут у {eng.id}: {bad}")
        routes[eng.id] = route
    plan = Plan(routes=routes, unassigned=list(unassigned))
    plan_scalar(plan, inst)
    return plan


# --------------------------------------------------------------------------- вставка

def _insertion_candidates(req, eng, order, inst, allow_new_engineer=True, min_pos=0,
                          noise=None, opening=None):
    """Лучшая позиция вставки заявки в маршрут исполнителя. Возвращает (дельта, позиция).

    min_pos — сколько заявок в начале трогать нельзя: при перепланировании это работы,
    которые инженер уже начал или закончил, вставлять перед ними бессмысленно.
    """
    if static_reason(req, eng):
        return None
    base, _ = evaluate(eng, order, inst.matrix, inst.index, earliest=inst.earliest_start, exempt=inst.earliest_exempt, committed=inst.committed.get(eng.id, []))
    base_km = base.distance_km if base else 0.0
    best = None
    for pos in range(min_pos, len(order) + 1):
        trial = order[:pos] + [req] + order[pos:]
        route, _ = evaluate(eng, trial, inst.matrix, inst.index, earliest=inst.earliest_start, exempt=inst.earliest_exempt, committed=inst.committed.get(eng.id, []))
        if route is None:
            continue
        delta = route.distance_km - base_km
        if not order and not allow_new_engineer:
            continue
        if not order:
            delta += OPEN_ENGINEER_PENALTY if opening is None else opening
        if noise is not None:
            # Шум во вставке — классический приём ALNS. Без него пересборка каждый раз
            # укладывает заявки одинаково, и разрушать план бессмысленно: поиск
            # возвращается туда же. Разбор отставания показал, что у эталонного решателя
            # 58 заявок из 66 стоят у других исполнителей — значит нужна другая укладка,
            # а не другие куски для разрушения.
            delta *= noise()
        if best is None or delta < best[0]:
            best = (delta, pos)
    return best


def _reason_for(req, inst, orders, *, evaluate_route=evaluate) -> Unassigned:
    """Почему заявку никуда не поставили.

    Причина не «самая частая», а **связывающая**. Раньше бралась мода по исполнителям,
    и получалось «ни у кого нет навыка» при том, что исполнитель с навыком есть — просто
    занят. Диспетчер по такому объяснению принял бы неверное решение: стал бы искать
    человека с навыком вместо того, чтобы разгрузить окно.

    Порядок разбора: если подходящих по квалификации и транспорту нет вовсе — причина
    в них. Если есть, но никому не влезает по времени — причина во времени, и мы честно
    говорим, сколько исполнителей подходили и почему не сложилось.
    """
    total = [e for e in inst.engineers if e.available]
    if not total:
        return Unassigned(req.id, "на этот день нет доступных исполнителей", "no_engineer")

    skill_ok = [e for e in total if req.skill not in {"__none__"} and req.skill in e.skills]
    if not skill_ok:
        title = SKILL_TITLES.get(req.skill, req.skill)
        return Unassigned(req.id, f"ни у кого из {len(total)} исполнителей нет навыка «{title}»",
                          "skill")

    fit_transport = [e for e in skill_ok if not req.required_transport or e.transport == req.required_transport]
    if not fit_transport:
        need = TRANSPORT_TITLES.get(req.required_transport, req.required_transport)
        return Unassigned(req.id,
                          f"навык есть у {len(skill_ok)}, но никто из них не на транспорте «{need}»",
                          "transport")

    fit_equipment = [e for e in fit_transport if equipment_reason(orders.get(e.id, []) + [req], e) is None]
    if not fit_equipment:
        return Unassigned(req.id, 'у подходящих по навыку и транспорту бригад недостаточно выданного на день оборудования', 'equipment')
    fit_transport = fit_equipment

    # Квалификация и транспорт подходят: значит дело во времени. Разбираем, в чём именно
    window_blocked, shift_blocked = 0, 0
    for eng in fit_transport:
        _, err = evaluate_route(eng, orders.get(eng.id, []) + [req], inst.matrix, inst.index,
                          earliest=inst.earliest_start, exempt=inst.earliest_exempt, committed=inst.committed.get(eng.id, []))
        if err and err[0] == "shift":
            shift_blocked += 1
        else:
            window_blocked += 1

    if shift_blocked > window_blocked:
        return Unassigned(req.id,
                          f"работа на {req.duration_min} мин не помещается в смену: "
                          f"проверено {len(fit_transport)} подходящих исполнителей",
                          "shift")
    return Unassigned(req.id,
                      f"окно {req.window_start:%H:%M}–{req.window_end:%H:%M} закрыто: "
                      f"все {len(fit_transport)} подходящих исполнителей в это время заняты",
                      "window")


def construct(inst: Instance, pool=None, orders=None, regret: int = 2,
              locked: dict[str, int] | None = None, noise: float = 0.0,
              rng: random.Random | None = None, forbid: set[str] | None = None,
              *, evaluate_route=evaluate, insertion=None, reason=None, rebuild=None, opening=None) -> Plan:
    """Regret-k вставка: первой ставим заявку, которой дороже всего достанется вторая позиция.

    forbid — исполнители, которых нельзя открывать заново на этой сборке. Нужен, чтобы
    распустить маршрут и не получить его обратно: жадная вставка первым делом кладёт
    заявку туда же, откуда её сняли, потому что оттуда ближе всего.

    evaluate_route, insertion, reason, rebuild — расчёт маршрута, быстрая вставка,
    объяснение отказа и сборка плана. По умолчанию это функции этого модуля; кэш
    популяции (src/engine/repair.py) передаёт свои, запоминающие, с тем же результатом.
    Медленная вставка при перепланировании их не использует: она только для событий.
    opening — цена открытия нового исполнителя во вставке; None — OPEN_ENGINEER_PENALTY (продукт).
    Кэш популяции передаёт свою: у сети b «плотная» вставка открывает бригаду, только если некуда.
    """
    if insertion is None:
        insertion = _insertion_candidates_fast if opening is None else partial(_insertion_candidates_fast, opening=opening)
    reason = reason or _reason_for
    rebuild = rebuild or _rebuild
    orders = {eid: list(v) for eid, v in (orders or {}).items()}
    for eng in inst.engineers:
        orders.setdefault(eng.id, [inst.by_id[s.request_id] for s in inst.committed.get(eng.id, [])])
    # Removing a visit is not necessarily safe: transport switching and traffic
    # can violate the triangle inequality. Requeue the infeasible suffix before
    # insertion; committed/locked history must never be removed.
    displaced = []
    for eng in inst.engineers:
        order = orders[eng.id]
        lock = max((locked or {}).get(eng.id, 0), len(inst.committed.get(eng.id, [])))
        while order:
            route, bad = evaluate_route(eng, order, inst.matrix, inst.index,
                                  earliest=inst.earliest_start, exempt=inst.earliest_exempt,
                                  committed=inst.committed.get(eng.id, []))
            if route is not None:
                break
            if len(order) <= lock:
                raise RuntimeError(f"Недопустимая фиксированная история у {eng.id}: {bad}")
            displaced.append(order.pop())
    assigned = {r.id for seq in orders.values() for r in seq}
    pending = list(pool if pool is not None else [r for r in inst.requests if r.id not in assigned])
    pending = list({r.id: r for r in pending + displaced if r.id not in assigned}.values())
    # Срочные вперёд: у аварий окно на сутки, но их просили делать как можно раньше
    pending.sort(key=lambda r: (-r.priority, r.window_start, -r.duration_min))

    unassigned: list[Unassigned] = []
    views: dict = {}
    equip: dict = {}
    jitter = None
    if noise > 0:
        rng = rng or random.Random(0)
        jitter = lambda: 1.0 + rng.uniform(-noise, noise)
    while pending:
        scored = []
        for req in pending:
            options = []
            for eng in inst.engineers:
                if not eng.available or (forbid and eng.id in forbid):
                    continue
                lock = max((locked or {}).get(eng.id, 0), len(inst.committed.get(eng.id, [])))
                if lock or inst.earliest_start is not None:        # при перепланировании начало маршрута трогать нельзя
                    cand = _insertion_candidates(req, eng, orders[eng.id], inst, min_pos=lock, opening=opening)
                else:
                    cand = insertion(req, eng, orders[eng.id], inst, views,
                                                      noise=jitter, equip_cache=equip)
                if cand:
                    options.append((cand[0], cand[1], eng.id))
            if not options:
                continue
            options.sort()
            best = options[0]
            second = options[min(regret - 1, len(options) - 1)][0] if len(options) > 1 else best[0] + 1e6
            urgency_bonus = 1e6 if req.urgent else 0.0
            scored.append((-(second - best[0]) - urgency_bonus, best[0], req, best))

        if not scored:
            for req in pending:
                unassigned.append(reason(req, inst, orders))
            break

        scored.sort(key=lambda x: (-x[2].priority, x[0], x[1]))
        _, _, req, (delta, pos, eid) = scored[0]
        orders[eid] = orders[eid][:pos] + [req] + orders[eid][pos:]
        views.pop(eid, None)        # маршрут изменился — развёртка и расход устарели
        equip.pop(eid, None)
        pending.remove(req)

    return rebuild(inst, orders, unassigned)


def baseline_greedy(inst: Instance) -> Plan:
    """Базовый вариант ТЗ: заявки по порядку поступления — первому подходящему исполнителю.

    Дословно из раздела 2.3: порядок посещения совпадает с порядком назначения,
    глобальной оптимизации нет.
    """
    orders = {e.id: [inst.by_id[s.request_id] for s in inst.committed.get(e.id, [])] for e in inst.engineers}
    fixed_ids = {r.id for seq in orders.values() for r in seq}
    unassigned = []
    for req in inst.requests:                       # порядок как во входных данных
        if req.id in fixed_ids:
            continue
        placed = False
        for eng in inst.engineers:                  # первый по порядку доступный
            if not eng.available or static_reason(req, eng):
                continue
            trial = orders[eng.id] + [req]          # только в конец: порядок = порядок назначения
            route, _ = evaluate(eng, trial, inst.matrix, inst.index, earliest=inst.earliest_start, exempt=inst.earliest_exempt, committed=inst.committed.get(eng.id, []))
            if route is not None:
                orders[eng.id] = trial
                placed = True
                break
        if not placed:
            unassigned.append(_reason_for(req, inst, orders))
    return _rebuild(inst, orders, unassigned)


def _route_equipment(eng, order, cache: dict):
    """Сколько оборудования уже занято маршрутом. Считается один раз на исполнителя.

    Профиль показал, что проверка оборудования съедала сорок процентов времени поиска:
    на каждую пробную позицию пересчитывался весь маршрут и создавался Counter.
    Расход маршрута не зависит от того, куда именно вставляют заявку, поэтому считаем
    его один раз, а кандидата проверяем добавлением. Полная проверка остаётся
    в scheduling.evaluate, через который проходит каждый собранный план.
    """
    used = cache.get(eng.id)
    if used is None:
        used = {}
        for r in order:
            for item, quantity in r.equipment.items():
                used[item] = used.get(item, 0) + quantity
        cache[eng.id] = used
    return used


def _equipment_fits(req, eng, used: dict) -> bool:
    for item, quantity in req.equipment.items():
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 0:
            return False
        if used.get(item, 0) + quantity > eng.equipment.get(item, 0):
            return False
    return True


def _insertion_candidates_fast(req, eng, order, inst, view_cache: dict, noise=None,
                               equip_cache: dict | None = None, opening=None):
    """То же, что _insertion_candidates, но через прямой/обратный проходы.

    Развёртка маршрута считается один раз на исполнителя и переиспользуется для всех
    заявок: это и есть основная экономия, а не скорость самой арифметики.
    """
    if static_reason(req, eng):
        return None
    if equip_cache is not None:
        if not _equipment_fits(req, eng, _route_equipment(eng, order, equip_cache)):
            return None
    elif equipment_reason(order + [req], eng):
        return None
    view = view_cache.get(eng.id, False)
    if view is False:
        view = build_view(eng, order, inst, eng.transport, earliest=inst.earliest_start, exempt=inst.earliest_exempt)
        view_cache[eng.id] = view
    if view is None:
        return None

    best = None
    for pos in range(len(order) + 1):
        delta = insertion_delta(view, pos, req, inst, eng, eng.transport)
        if delta is None:
            continue
        if not order:
            delta += OPEN_ENGINEER_PENALTY if opening is None else opening
        if noise is not None:
            # Шум во вставке — классический приём ALNS. Без него пересборка каждый раз
            # укладывает заявки одинаково, и разрушать план бессмысленно: поиск
            # возвращается туда же. Разбор отставания показал, что у эталонного решателя
            # 58 заявок из 66 стоят у других исполнителей — значит нужна другая укладка,
            # а не другие куски для разрушения.
            delta *= noise()
        if best is None or delta < best[0]:
            best = (delta, pos)
    return best
