"""Ручное переназначение заявки одной транзакцией: порядок маршрутов сохраняется, меняется только выбранная заявка."""
import copy

from src.plan.events import frozen_ids
from src.plan.scheduling import evaluate
from src.plan.solver import _rebuild
from src.plan.validate import validate


def assign(inst, plan, request_id, engineer_id, at):
    if plan is None:
        raise ValueError("Сначала постройте план")
    if inst.earliest_start is not None and at < inst.earliest_start:
        raise ValueError("Время события не может идти назад")
    if request_id not in {r.id for r in inst.requests}:
        raise ValueError("Заявка отсутствует в активном наборе")
    eng = next((e for e in inst.engineers if e.id == engineer_id and e.available), None)
    if eng is None:
        raise ValueError("Инженер отсутствует или недоступен")
    keep = frozen_ids(plan, at)
    if request_id in keep:
        raise ValueError("Выезд уже начат: переназначение запрещено")

    trial = copy.deepcopy(inst)
    trial.earliest_start = at
    trial.earliest_exempt = keep
    trial.committed = {eid: [copy.deepcopy(s) for s in route.stops if s.request_id in keep]
                       for eid, route in plan.routes.items()}
    orders = {eid: [trial.by_id[s.request_id] for s in route.stops if s.request_id != request_id]
              for eid, route in plan.routes.items()}
    order = orders[engineer_id]
    req = trial.by_id[request_id]
    best = None
    reasons = []
    for pos in range(len(trial.committed.get(engineer_id, [])), len(order) + 1):
        candidate = order[:pos] + [req] + order[pos:]
        route, bad = evaluate(eng, candidate, trial.matrix, trial.index, earliest=at,
                              committed=trial.committed.get(engineer_id, []))
        if route is not None and (best is None or route.distance_km < best[0]):
            best = (route.distance_km, candidate)
        if bad:
            reasons.append(bad[1])
    if best is None:
        raise ValueError("Переназначение невозможно: " + (reasons[0] if reasons else "нет допустимой позиции"))
    orders[engineer_id] = best[1]
    try:
        result = _rebuild(trial, orders, [copy.deepcopy(u) for u in plan.unassigned
                                         if u.request_id != request_id])
    except RuntimeError as exc:
        raise ValueError("Остальные маршруты не выполнимы на выбранный момент; сначала перепланируйте день") from exc
    report = validate(result, trial)
    if not report['ok']:
        raise ValueError("Переназначение нарушает ограничения плана")
    return result, trial
