"""Что видит сеть: проекция плана-родителя на подвижную часть (шаг 5г.3, EVENTS-5G3-DESIGN.md).

Закреплённые заявки не становятся узлами выбора, соседства и целей. История видна
только через последствия: исполнитель готов к новой работе с момента готовности, у него
остаток оборудования, он уже в составе. Маршрут в признаках — подвижный хвост, чтобы
неизменный пробег истории не звал разрушить маршрут.

Мутация, критерий и валидатор работают с полным планом. Здесь только вход сети.
Утром (без истории) проекция тождественна: те же объекты, граф побайтово прежний.
"""
# refactor: Claude, 21.09.2026 — шаг 5г.3
import copy
from dataclasses import dataclass, replace
from datetime import timedelta

from src.plan.fastroute import day_zero
from src.model import Route


@dataclass(frozen=True)
class View:
    inst: object              # задача для признаков: исполнители с готовностью и остатком, заявки без закреплённых
    plan: object              # план для признаков: маршруты — подвижные хвосты
    request_ids: tuple        # полный индекс в исходных inst.requests для каждой заявки проекции
    already_used: frozenset   # исполнители, уже задействованные историей
    origin: dict              # точка матрицы, откуда стартует хвост, по исполнителю


def project(inst, plan, state):
    """Проекция (inst, plan) на подвижную часть по снимку истории state (RouteState)."""
    if not state.has_history:
        depot = inst.index['depot']
        return View(inst, plan, tuple(range(len(inst.requests))), frozenset(), {e.id: depot for e in inst.engineers})
    tails = state.tails
    engineers = []
    for eng in inst.engineers:
        tail = tails[eng.id]
        used = dict(tail.used)
        # Готовность — момент, с которого исполнитель свободен; остаток — запас минус история.
        engineers.append(replace(eng, shift_start=day_zero(eng) + timedelta(minutes=tail.ready),
                                 equipment={k: v - used.get(k, 0) for k, v in eng.equipment.items()}))
    view_inst = copy.copy(inst)
    view_inst.engineers = engineers
    view_inst.requests = [r for r in inst.requests if r.id not in state.pinned]
    routes = {eid: Route(engineer_id=eid, stops=route.stops[tails[eid].lock:]) for eid, route in plan.routes.items()}
    request_ids = tuple(i for i, r in enumerate(inst.requests) if r.id not in state.pinned)
    return View(view_inst, replace(plan, routes=routes), request_ids,
                frozenset(eid for eid, tail in tails.items() if tail.already_used),
                {eid: tail.origin for eid, tail in tails.items()})
