"""Постобработка «аварии вперёд»: авария встаёт в начало маршрута, если цель от этого лучше.

После поиска, над лучшим планом портфеля. Вставка выбирает место заявки по километрам и ожидание аварий не
видит; этот проход пробует ход, которого поиск сам почти не делает: авария — первой в подвижной части маршрута
своей или другой уже работающей бригады с аварийным допуском (сразу за закреплённым после события началом и за
авариями, уже поставленными вперёд). Берётся лучший ход по cost_tuple, если он строго лучше; повторяется, пока
ходы улучшают. Новые маршруты не открываются, заявки не снимаются — план по цели не хуже исходного.

Проход ограничен временем; результат проверяется валидатором — недопустимый не выдаётся.
Замер 23.09 (RESULTS.md, «иголка v3»): при квадратичном хвосте улучшал план в 17 из 30 прогонов.
"""
# refactor: Claude, 23.09.2026 — постобработка по решению Михаила (флаг в настройках поиска, по умолчанию включён)
import time

from src.plan.solver import _rebuild
from src.plan.validate import validate


def urgent_front(inst, plan, seconds: float = 1.0):
    """(план не хуже исходного по цели, число принятых ходов)."""
    deadline = time.perf_counter() + seconds
    orders = {eid: [inst.by_id[s.request_id] for s in r.stops] for eid, r in plan.routes.items()}
    engineers = {e.id: e for e in inst.engineers}
    locked = {eid: len(inst.committed.get(eid, [])) for eid in orders}      # начатое до события не двигается
    best, moves = plan, 0
    while time.perf_counter() < deadline:
        move = None
        for source, seq in orders.items():
            for req in seq[locked[source]:]:
                if not req.urgent:
                    continue
                for eid, target in orders.items():
                    eng = engineers[eid]
                    if not target or not getattr(eng, 'available', True) or 'emergency' not in eng.skills:
                        continue
                    rest = [r for r in target if r is not req]
                    head = locked[eid]
                    while head < len(rest) and rest[head].urgent:
                        head += 1
                    trial = {**orders, eid: rest[:head] + [req] + rest[head:]}
                    if trial[eid] == target:
                        continue                              # уже на месте
                    if eid != source:
                        trial[source] = [r for r in seq if r is not req]
                    try:
                        candidate = _rebuild(inst, trial, list(best.unassigned))
                    except RuntimeError:
                        continue                              # недопустимо: окна, смена, навыки
                    if candidate.cost_tuple() < (move[0] if move else best).cost_tuple():
                        move = (candidate, trial)
                if time.perf_counter() >= deadline:
                    break
        if move is None:
            break
        best, orders = move
        moves += 1
    if moves and not validate(best, inst)['ok']:
        return plan, 0
    return best, moves
