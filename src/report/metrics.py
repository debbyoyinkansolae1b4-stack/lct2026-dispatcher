"""Метрики плана и сравнение вариантов.

Две обязательные метрики ТЗ — число задействованных исполнителей и пробег по каждому
исполнителю (и суммарный). Остальное добавлено, чтобы было видно, чем платим.
"""

from src.instance import Instance
from src.model import Plan
from src.plan.scheduling import day_start, urgent_delay_min


def urgent_wait(plan: Plan, inst: Instance) -> dict:
    """Ожидание аварий: сколько назначено, сумма и максимум ожидания; по каждой — база отсчёта
    (max(поступление, начало смен), как в цели), плановое начало и +минуты; неназначенная —
    статус без времени. Прогноз опозданий ТЗ (2.5) для аварий; формулировка — Codex, 22.09."""
    from src.plan import solver
    first = day_start(inst)
    target = solver.URGENT_TARGET_MIN        # ориентир реакции экспертов (config/weights.json)
    over = 0
    placed = {s.request_id: (eid, s) for eid, r in plan.routes.items() for s in r.stops}
    names = {e.id: e.name for e in inst.engineers}
    items = []
    # Завершённые остаются в плане и в дневной цели, но не в inst.requests (Н62).
    done = [inst.by_id[i] for i in sorted(inst.completed) if i in inst.by_id and i in placed]
    for req in list(inst.requests) + done:
        if not req.urgent:
            continue
        base = max(req.window_start, first) if first is not None else req.window_start
        item = {"id": req.id, "address": req.address, "base": f"{base:%H:%M}"}
        if req.id in placed:
            eid, stop = placed[req.id]
            wait = max(0., (stop.start - base).total_seconds() / 60)
            over += wait > target
            item.update(status="выполнена" if req.id in inst.completed else "назначена", start=f"{stop.start:%H:%M}",
                        wait_min=round(wait), engineer=names.get(eid, eid))
        else:
            item.update(status="не назначена", start=None, wait_min=None, engineer=None)
        items.append(item)
    waits = [i["wait_min"] for i in items if i["wait_min"] is not None]
    items.sort(key=lambda i: (i["wait_min"] is not None, -(i["wait_min"] or 0)))   # неназначенные, затем дольше ждущие
    return {"total": len(items), "assigned": len(waits), "sum_min": sum(waits), "max_min": max(waits, default=0),
            "target_min": round(target), "over_target": over, "items": items}


# refactor: Claude, 26.09.2026 — пробки в плане: сколько времени в пути добавила загруженность (любой источник)
def traffic_effect(plan: Plan, inst: Instance) -> dict:
    """Время в пути плана против свободной дороги: сколько добавили пробки, всего и по транспорту.

    Каждое плечо пересчитывается на момент выезда (прибытие минус время в пути — как считало расписание) и без
    загруженности (без коэффициента и без слоёв: свободная дорога OSRM плюс надбавка режима). Одинаково для профиля
    по часам, файла со слоями и внешнего сервиса — слои задают время на момент выезда, база та же.
    """
    matrix, engineers = inst.matrix, {e.id: e for e in inst.engineers}
    total = free = 0.
    modes: dict[str, list[float]] = {}
    for eid, route in plan.routes.items():
        mode, here = engineers[eid].transport, inst.index["depot"]
        for stop in route.stops:
            there = inst.index[stop.request_id]
            base = matrix._compute_minutes(here, there, mode, None)
            total, free = total + stop.travel_min, free + base
            m = modes.setdefault(mode, [0., 0.])
            m[0], m[1] = m[0] + stop.travel_min, m[1] + base
            here = there
    pct = lambda a, b: round((a / b - 1) * 100) if b else 0
    return {"source": matrix.time_source, "travel_min": round(total), "free_min": round(free),
            "extra_min": round(total - free), "extra_pct": pct(total, free),
            "by_mode": {k: {"extra_min": round(a - b), "extra_pct": pct(a, b)} for k, (a, b) in modes.items()}}


def summarize(plan: Plan, inst: Instance, title: str = "") -> dict:
    routes = []
    for eid, route in plan.routes.items():
        if not route.used:
            continue
        eng = next(e for e in inst.engineers if e.id == eid)
        routes.append({
            "engineer_id": eid,
            "engineer": eng.name,
            "transport": eng.transport,
            "stops": len(route.stops),
            "km": round(route.distance_km, 2),
            "travel_min": round(route.travel_min),
            "work_min": sum(inst.by_id[s.request_id].duration_min for s in route.stops),
            "first_start": min((s.start for s in route.stops), default=None),
            "last_end": max((s.end for s in route.stops), default=None),
        })
    routes.sort(key=lambda r: -r["stops"])

    total_requests = len(inst.requests) + len(inst.nogeo) + len(inst.completed)
    return {
        "title": title,
        "region": inst.region,
        "requests_total": total_requests,
        "assigned": plan.assigned_count,
        "completed": len(inst.completed),
        "priority_assigned": {"emergency": plan.priority_counts[0], "connection": plan.priority_counts[1]},
        "unassigned": total_requests - plan.assigned_count,
        "used_engineers": plan.used_engineers,
        "available_engineers": len(inst.engineers),
        "total_km": round(plan.total_km, 2),
        "km_per_engineer": round(plan.total_km / max(plan.used_engineers, 1), 2),
        "max_route_km": round(max((r["km"] for r in routes), default=0.0), 2),
        "urgent_delay_min": round(sum(urgent_delay_min(r, inst.by_id, day_start(inst)) for r in plan.routes.values())),
        "routes": routes,
        "unassigned_list": [{"id": u.request_id, "code": u.reason_code, "reason": u.reason}
                            for u in plan.unassigned],
        "cost_tuple": plan.cost_tuple(),
        "urgent_wait": urgent_wait(plan, inst),
        "traffic": traffic_effect(plan, inst),
    }


def compare(base: dict, other: dict) -> dict:
    """Насколько other лучше base по трём метрикам. Положительное — в нашу пользу."""
    def pct(a, b):
        return round((a - b) / a * 100, 1) if a else 0.0

    return {
        "assigned_delta": other["assigned"] - base["assigned"],
        "engineers_delta": other["used_engineers"] - base["used_engineers"],
        "engineers_saved_pct": pct(base["used_engineers"], other["used_engineers"]),
        "km_delta": round(other["total_km"] - base["total_km"], 2),
        "km_saved_pct": pct(base["total_km"], other["total_km"]),
    }


def control_reference(inst: Instance) -> dict | None:
    """Контрольная выборка: сколько бригад выполнили эти заявки — доступный состав участка на день.

    Это ориентир, а не эталон [24:06]: пробег по контролю мы посчитать не можем — в нём
    нет порядка объезда, только факт назначения.
    """
    from src.data.loading import load_control
    try:
        ctrl = load_control(inst.region)
    except FileNotFoundError:
        return None                                 # загруженный набор: контрольного дня нет
    brigades = {str(b).strip() for b in ctrl["Бригада"].dropna() if str(b).strip()}
    return {
        "requests": len(ctrl),
        "used_engineers": len(brigades),
        "per_engineer": round(len(ctrl) / max(len(brigades), 1), 1),
    }


def as_text(summary: dict) -> str:
    s = summary
    lines = [
        f"=== {s['title'] or 'план'} · {s['region']} ===",
        f"Заявок {s['requests_total']}: назначено {s['assigned']}, не назначено {s['unassigned']}",
        f"Исполнителей задействовано {s['used_engineers']} из {s['available_engineers']}",
        f"Пробег {s['total_km']} км (в среднем {s['km_per_engineer']} км на исполнителя, "
        f"максимум {s['max_route_km']} км)",
    ]
    if s["urgent_delay_min"]:
        lines.append(f"Суммарная задержка срочных от начала окна: {s['urgent_delay_min']} мин")
    lines.append("")
    lines.append(f"{'Исполнитель':<24}{'тр-т':<8}{'заявок':>7}{'км':>9}{'в пути':>9}{'работа':>9}")
    for r in s["routes"]:
        lines.append(f"{r['engineer']:<24}{r['transport']:<8}{r['stops']:>7}{r['km']:>9.1f}"
                     f"{r['travel_min']:>8}м{r['work_min']:>8}м")
    if s["unassigned_list"]:
        lines.append("")
        lines.append("Не назначены:")
        for u in s["unassigned_list"][:15]:
            lines.append(f"  {u['id']}: {u['reason']}")
        if len(s["unassigned_list"]) > 15:
            lines.append(f"  … ещё {len(s['unassigned_list']) - 15}")
    return "\n".join(lines)


def capacity_advice(plan: Plan, inst: Instance) -> list[dict]:
    """Сколько ещё исполнителей нужно, чтобы закрыть неназначенное.

    Постановщик назвал это правильной реакцией системы: «нужно дополнительных 5 исполнителей,
    чтобы всё это выполнить» [47:36] — вместо того чтобы молча потерять заявки.
    """
    import math
    from collections import defaultdict

    by_skill = defaultdict(list)
    for u in plan.unassigned:
        req = inst.by_id.get(u.request_id)
        if req:
            by_skill[(req.skill, u.reason_code)].append(req)

    shift_min = max(
        (e.shift_end - e.shift_start).total_seconds() / 60 for e in inst.engineers
    ) if inst.engineers else 720
    usable = shift_min * 0.7        # 30% смены съедает дорога между адресами

    advice = []
    for (skill, code), reqs in sorted(by_skill.items(), key=lambda x: -len(x[1])):
        work = sum(r.duration_min for r in reqs)
        need = max(1, math.ceil(work / usable))
        have = sum(1 for e in inst.engineers if skill in e.skills)
        title = {"local": "Локальные работы", "connect": "Работы на подключение и дозаказы",
                 "emergency": "Аварийные работы"}[skill]
        why = {"skill": f"исполнителей с этим навыком в участке всего: {have}",
               "window": "все подходящие исполнители заняты в эти окна",
               "shift": "работа не помещается в смену",
               "transport": "нет исполнителя с нужным транспортом",
               "no_engineer": "нет доступных исполнителей"}[code]
        advice.append({
            "skill": skill, "skill_title": title, "reason_code": code,
            "requests": len(reqs), "work_minutes": work,
            "need_extra_engineers": need,
            "text": (f"Чтобы закрыть оставшиеся заявки «{title}» (их {len(reqs)}, работы на {work} мин), "
                     f"нужно ещё исполнителей с этим навыком: {need}. Причина: {why}."),
        })
    return advice
