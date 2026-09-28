"""Объяснения для диспетчера.

Постановщик попросил не разворачивать весь алгоритм по каждой заявке — «получится портянка,
её никто не читает» [33:56]. Поэтому здесь два уровня: одна строка на назначение и короткий
разбор по запросу. Причина неназначения обязательна и живёт в scheduling/solver.
"""

from datetime import timedelta

from src.instance import Instance
from src.model import Plan
from src.plan.scheduling import SKILL_TITLES, TRANSPORT_TITLES, evaluate, static_reason


def assignment_line(plan: Plan, inst: Instance, request_id: str) -> str:
    """Одна фраза: кто, когда, почему он."""
    for eid, route in plan.routes.items():
        for i, stop in enumerate(route.stops):
            if stop.request_id != request_id:
                continue
            eng = next(e for e in inst.engineers if e.id == eid)
            req = inst.by_id[request_id]
            prev = "из офиса" if i == 0 else f"после заявки {route.stops[i - 1].request_id}"
            slack = (req.window_end - stop.start).total_seconds() / 60
            return (f"{eng.name} приезжает в {stop.arrive:%H:%M} {prev}, "
                    f"начинает в {stop.start:%H:%M} — это внутри окна "
                    f"{req.window_start:%H:%M}–{req.window_end:%H:%M} (запас {slack:.0f} мин), "
                    f"заканчивает в {stop.end:%H:%M}.")
    return "Заявка не назначена."


def explain_assignment(plan: Plan, inst: Instance, request_id: str) -> dict:
    """Разбор одного назначения: какие ограничения сработали и кто ещё мог бы поехать."""
    req = inst.by_id[request_id]
    assigned_eid = next((eid for eid, r in plan.routes.items()
                         if any(s.request_id == request_id for s in r.stops)), None)

    # Альтернатива — изменение пробега ВСЕГО плана, если отдать заявку этой бригаде: сколько добавится к её маршруту
    # минус сколько сэкономит нынешний маршрут. Полные длины маршрутов разной загрузки сравнивать нельзя (замечание
    # Codex 26.09): пустой маршрут выглядел «дешёвым», хотя это ещё одна бригада. refactor: Claude, 26.09.2026
    def km(eng, order):
        route, _ = evaluate(eng, order, inst.matrix, inst.index, earliest=inst.earliest_start, exempt=inst.earliest_exempt)
        return None if route is None else route.distance_km

    engineers = {e.id: e for e in inst.engineers}
    saving = 0.
    if assigned_eid:
        own = [inst.by_id[s.request_id] for s in plan.routes[assigned_eid].stops]
        rest = [r for r in own if r.id != request_id]
        without = km(engineers[assigned_eid], rest) if rest else 0.
        saving = plan.routes[assigned_eid].distance_km - (without if without is not None else plan.routes[assigned_eid].distance_km)
    alternatives, blocked = [], []
    for eng in inst.engineers:
        bad = static_reason(req, eng)
        if bad:
            blocked.append({"engineer": eng.name, "code": bad[0], "why": bad[1]})
            continue
        if eng.id == assigned_eid:
            alternatives.append({"engineer": eng.name, "delta_km": 0., "extra_crew": False, "current": True})
            continue
        order = [inst.by_id[s.request_id] for s in plan.routes[eng.id].stops]
        best = min((d for pos in range(len(order) + 1)
                    if (d := km(eng, order[:pos] + [req] + order[pos:])) is not None), default=None)
        if best is None:
            blocked.append({"engineer": eng.name, "code": "window",
                            "why": "не помещается в его текущий маршрут по времени"})
            continue
        alternatives.append({"engineer": eng.name, "delta_km": round(best - plan.routes[eng.id].distance_km - saving, 1),
                             "extra_crew": not order, "current": False})

    alternatives.sort(key=lambda a: (not a["current"], a["extra_crew"], a["delta_km"]))
    reasons = [
        f"Требуемый навык: {SKILL_TITLES[req.skill]}.",
        f"Окно клиента: {req.window_start:%H:%M}–{req.window_end:%H:%M}, работа {req.duration_min} мин.",
    ]
    if req.required_transport:
        reasons.append(f"Нужен транспорт: {TRANSPORT_TITLES[req.required_transport]}.")
    if req.urgent:
        reasons.append("Заявка срочная — ставим как можно раньше.")
    skill_blocked = sum(1 for b in blocked if b["code"] == "skill")
    if skill_blocked:
        reasons.append(f"По навыку отпало исполнителей: {skill_blocked}.")

    return {
        "request_id": request_id,
        "headline": assignment_line(plan, inst, request_id),
        "constraints": reasons,
        "alternatives": alternatives[:5],
        "blocked_count": len(blocked),
    }


def route_summary(plan: Plan, inst: Instance, engineer_id: str) -> dict:
    route = plan.routes[engineer_id]
    eng = next(e for e in inst.engineers if e.id == engineer_id)
    items = []
    for s in route.stops:
        r = inst.by_id[s.request_id]
        items.append({
            "request_id": r.id, "status": r.status, "priority": r.priority, "equipment": r.equipment, "district": r.district, "address": r.address,
            "hd_type": r.hd_type, "skill": SKILL_TITLES[r.skill],
            "window": f"{r.window_start:%H:%M}–{r.window_end:%H:%M}",
            "arrive": f"{s.arrive:%H:%M}", "start": f"{s.start:%H:%M}", "end": f"{s.end:%H:%M}",
            "travel_min": round(s.travel_min), "travel_km": round(s.travel_km, 1),
            "urgent": r.urgent, "lat": r.lat, "lon": r.lon,
        })
    return {
        "engineer_id": engineer_id, "engineer": eng.name,
        "transport": TRANSPORT_TITLES[eng.transport],
        "skills": sorted(SKILL_TITLES[s] for s in eng.skills),
        "shift": f"{eng.shift_start:%H:%M}–{eng.shift_end:%H:%M}",
        "km": round(route.distance_km, 1), "stops": items,
        "why": (f"Маршрут построен от офиса: заявок — {len(items)}, пробег {round(route.distance_km, 1)} км, "
                f"в пути {round(route.travel_min)} мин. Порядок объезда выбран так, чтобы каждое "
                f"прибытие попадало в своё окно при минимальном пробеге.") if items else
               "Исполнитель не задействован — его заявки уместились у других без потери допустимости.",
    }


def plan_diff(before: Plan, after: Plan, inst: Instance) -> dict:
    """Что изменилось после перепланирования — это ТЗ требует показывать явно."""
    def where(plan):
        return {s.request_id: eid for eid, r in plan.routes.items() for s in r.stops}

    b, a = where(before), where(after)
    moved, added, removed = [], [], []
    names = {e.id: e.name for e in inst.engineers}
    starts = {s.request_id: s.start for r in after.routes.values() for s in r.stops}
    for rid, eid in a.items():
        if rid not in b:
            # Диспетчеру — адрес, исполнитель и начало работ, а не внутренний номер (SOS-… у аварии).
            req = inst.by_id.get(rid)
            added.append({"request_id": rid, "to": eid, "engineer": names.get(eid, eid),
                          "address": req.address if req else rid, "urgent": bool(req and req.urgent),
                          "start": f"{starts[rid]:%H:%M}"})
        elif b[rid] != eid:
            moved.append({"request_id": rid, "from": b[rid], "to": eid})
    for rid, eid in b.items():
        if rid not in a:
            removed.append({"request_id": rid, "from": eid})

    order_changed = []
    for eid in after.routes:
        seq_b = [s.request_id for s in before.routes.get(eid).stops] if eid in before.routes else []
        seq_a = [s.request_id for s in after.routes[eid].stops]
        if seq_b != seq_a and set(seq_b) == set(seq_a) and seq_a:
            order_changed.append(eid)

    return {
        "moved": moved, "added": added, "removed": removed,
        "order_changed": order_changed,
        "engineers_before": before.used_engineers, "engineers_after": after.used_engineers,
        "km_before": round(before.total_km, 1), "km_after": round(after.total_km, 1),
        "assigned_before": before.assigned_count, "assigned_after": after.assigned_count,
    }
