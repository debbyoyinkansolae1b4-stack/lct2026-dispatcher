"""Признаки плана для графа сети: маршруты, неназначенные заявки, глобальное состояние поиска.

Перенесено дословно из src/policy_net.py (сеть ALNS уходит на шаге 8, а этот кодировщик
нужен графу популяционного поиска — context.encode_context). Порядок и смысл колонок —
часть эталона шага 0 и схемы весов: менять только вместе с новой версией эталона.
"""
# refactor: Claude, 22.09.2026 — шаг 8а: из src/policy_net.py без изменений
import torch

EQUIPMENT = ('repair_kit', 'router', 'stb', 'cable_kit')
ROUTE_FEATURES = 23
REQUEST_FEATURES = 13
GLOBAL_FEATURES = 8


def encode(inst, plan, iteration: int, iterations: int, no_improve: int):
    """План → три тензора: маршруты, неназначенные, глобальное состояние."""
    skills = ("local", "connect", "emergency")
    transports = ("car", "public", "foot", "bike")

    rows = []
    for eng in inst.engineers:
        route = plan.routes[eng.id]
        span = (eng.shift_end - eng.shift_start).total_seconds() / 60 or 1.0
        work = sum(inst.by_id[s.request_id].duration_min for s in route.stops)
        assigned = [inst.by_id[s.request_id] for s in route.stops]
        stock = [eng.equipment.get(item, 0) for item in EQUIPMENT]
        remaining = [eng.equipment.get(item, 0) - sum(r.equipment.get(item, 0) for r in assigned)
                     for item in EQUIPMENT]
        # Величины сырые. Понижающие коэффициенты (/10, /50) подбирались на глаз и после
        # нормировки тождественно не влияют на вход: (x/10 - m)/s совпадает с (x - 10m)/(10s).
        # Доли вроде travel_min/span оставлены: это не подгонка масштаба, а смысл —
        # какая часть смены ушла на дорогу.
        rows.append([
            len(route.stops),
            route.distance_km,
            route.travel_min / span,
            work / span,
            1.0 if route.used else 0.0,
            *[1.0 if s in eng.skills else 0.0 for s in skills],
            *[1.0 if eng.transport == t else 0.0 for t in transports],
            float(eng.available),
            float(sum(r.priority == 2 for r in assigned)),
            float(sum(r.priority == 1 for r in assigned)),
            *[float(v) for v in stock],
            *[float(v) for v in remaining],
        ])

    day_start = min((e.shift_start for e in inst.engineers), default=None)
    reqs = []
    for u in plan.unassigned:
        r = inst.by_id.get(u.request_id)
        if r is None:
            continue
        hour = ((r.window_start - day_start).total_seconds() / 3600.0) if day_start else 0.0
        reqs.append([
            *[1.0 if r.skill == s else 0.0 for s in skills],
            1.0 if r.urgent else 0.0,
            float(r.duration_min),
            hour,
            (r.window_end - r.window_start).total_seconds() / 3600.0,
            1.0 if r.required_transport else 0.0,
            float(r.priority),
            *[float(r.equipment.get(item, 0)) for item in EQUIPMENT],
        ])

    n = max(len(inst.requests), 1)
    glob = [
        len(plan.unassigned) / n,
        plan.used_engineers / max(len(inst.engineers), 1),
        plan.total_km,
        iteration / max(iterations, 1),
        float(no_improve),          # обрезание на 200 теряло информацию; нормировка справится
        sum(1 for r in plan.routes.values() if r.used and len(r.stops) <= 2) / max(len(inst.engineers), 1),
        sum(inst.by_id[u.request_id].priority == 2 for u in plan.unassigned if u.request_id in inst.by_id) / n,
        sum(inst.by_id[u.request_id].priority == 1 for u in plan.unassigned if u.request_id in inst.by_id) / n,
    ]
    return (torch.tensor(rows, dtype=torch.float32).reshape(-1, ROUTE_FEATURES),
            torch.tensor(reqs, dtype=torch.float32) if reqs else torch.zeros((0, REQUEST_FEATURES)),
            torch.tensor(glob, dtype=torch.float32))
