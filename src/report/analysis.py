"""Сводка по набору до всякого поиска: сколько работы на каждый навык и что достижимо для аварий.

Для вкладки «Анализ» админки и разговора «что даст ещё одна бригада». Ожидание аварий — по реальной матрице
времени (профиль транспорта бригады), от начала смен, как в цели:
  * без занятости — к каждой аварии сразу едет ближайшая бригада с допуском по кратчайшему пути через любые точки
    и наименьшему за день времени плеча: нижняя граница ожидания этой аварии в любом плане (Н68);
  * только аварии — бригады с допуском с утра объезжают одни аварии (освободившаяся берёт ту, где раньше всех
    начнёт, успевая до конца смены): оценка, если отдать аварии утро; запасы, транспорт и плановые заявки этих
    бригад не учтены — это не граница и не проверенный план.
Считается за доли секунды, без поиска. Логика — research/bound_urgent_wait.py (замер 23.09, RESULTS.md).
"""
# refactor: Claude, 23.09.2026 — вкладка «Анализ»: сводка по набору
from datetime import timedelta

from src.plan.scheduling import day_start

SKILLS = (("emergency", "Аварийные работы"), ("connect", "Подключения и дозаказы"), ("local", "Локальные работы"))


def _minutes(inst, eng, a, b, at):
    depart = (at - at.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 60
    return inst.matrix.minutes(inst.index[a], inst.index[b], eng.transport, depart)


def _edge_floor(inst, mode):
    """Нижняя оценка минут на каждое плечо за весь день: наименьшее время по всем моментам выезда.

    Без загруженности — готовая таблица. С часовым профилем коэффициент один на все плечи, поэтому минимум — в час
    с наименьшим коэффициентом. Со слоями внешнего источника — минимум по всем слоям и часам для каждого плеча.
    """
    m = inst.matrix
    n = len(m.points)
    plain = m._plain_matrix(mode)
    if plain is not None:
        return plain
    if not m.traffic.layers:
        hour = min(range(24), key=lambda h: m.traffic.factor(h * 60, mode))
        return [[m.minutes(i, j, mode, hour * 60) for j in range(n)] for i in range(n)]
    departs = sorted({h * 60 for h in range(24)} | set(m.traffic.layers))
    return [[min(m.minutes(i, j, mode, d) for d in departs) for j in range(n)] for i in range(n)]


def _shortest_from(table, source):
    """Кратчайшие пути из source через любые точки (Дейкстра на плотной матрице, O(n²))."""
    n = len(table)
    dist, done = [float("inf")] * n, [False] * n
    dist[source] = 0.
    for _ in range(n):
        u = min((i for i in range(n) if not done[i]), key=dist.__getitem__)
        done[u] = True
        row = table[u]
        for v in range(n):
            if not done[v] and dist[u] + row[v] < dist[v]:
                dist[v] = dist[u] + row[v]
    return dist


def urgent_bounds(inst, target_min: float = 120.) -> dict:
    """Ожидание аварий: нижняя граница по каждой и оценка «только аварии».

    Граница (Н68): к аварии сразу едет ближайшая бригада с допуском, путь — кратчайший через любые точки по наименьшему
    за день времени плеча, без времени работ по пути. Она верна при любой матрице (без неравенства треугольника и FIFO)
    и любом профиле пробок: быстрее не приехать ни в одном плане. «Только аварии» — жадное расписание одних аварий
    бригадами с допуском с начала смен с учётом конца смены; запасы, требования к транспорту и плановые заявки этих
    бригад не учтены — это оценка, а не гарантированно допустимый план.
    """
    first = day_start(inst)
    crews = [e for e in inst.engineers if "emergency" in e.skills]
    urgent = [r for r in inst.requests if r.urgent]
    if not urgent or not crews:
        return {"urgent": len(urgent), "crews": len(crews), "alone_max": None, "alone_within": None,
                "only_max": None, "only_within": None, "only_unassigned": None, "target_min": target_min}
    base = {r.id: max(r.window_start, first) for r in urgent}
    depot = inst.index["depot"]
    reach = {mode: _shortest_from(_edge_floor(inst, mode), depot) for mode in {e.transport for e in crews}}
    alone = [min(max(0., reach[e.transport][inst.index[r.id]] + (e.shift_start - base[r.id]).total_seconds() / 60)
                 for e in crews) for r in urgent]
    free = {e.id: (e.shift_start, "depot") for e in crews}
    waits, left = [], {r.id: r for r in urgent}
    while left:
        best = None
        for e in crews:
            at, here = free[e.id]
            for r in left.values():
                start = max(at + timedelta(minutes=_minutes(inst, e, here, r.id, at)), base[r.id])
                if start + timedelta(minutes=r.duration_min) > e.shift_end:
                    continue                                   # не успевает до конца смены
                if best is None or start < best[0]:
                    best = (start, e, r)
        if best is None:
            break                                              # оставшиеся аварии не помещаются в смены
        start, e, r = best
        waits.append((start - base[r.id]).total_seconds() / 60)
        free[e.id] = (start + timedelta(minutes=r.duration_min), r.id)
        del left[r.id]
    return {"urgent": len(urgent), "crews": len(crews), "target_min": target_min,
            "alone_max": round(max(alone)), "alone_within": sum(w <= target_min for w in alone),
            "only_max": round(max(waits)) if waits else None, "only_within": sum(w <= target_min for w in waits),
            "only_unassigned": len(left)}


def summary(inst, target_min: float = 120.) -> dict:
    """Работа по навыкам (заявок, часов, бригад с навыком) и граница ожидания аварий."""
    skills = []
    for key, title in SKILLS:
        reqs = [r for r in inst.requests if r.skill == key]
        engs = [e for e in inst.engineers if key in e.skills]
        skills.append({"skill": title, "requests": len(reqs), "work_hours": round(sum(r.duration_min for r in reqs) / 60, 1),
                       "engineers": len(engs),
                       "shift_hours": round(sum((e.shift_end - e.shift_start).total_seconds() for e in engs) / 3600, 1)})
    return {"requests": len(inst.requests), "engineers": len(inst.engineers), "skills": skills,
            "urgent": urgent_bounds(inst, target_min)}


# refactor: Claude, 26.09.2026 — пробки по часам для вкладки «Данные»: профиль, файл со слоями или внешний сервис
TRAFFIC_SAMPLE = 4000        # пар точек для разброса по слоям: 300 точек — 90 000 пар, хватает выборки


def traffic_hours(inst) -> dict:
    """Во сколько раз дорога на машине дольше свободной по часу выезда, 0–23.

    Профиль — один коэффициент на весь участок (по машине и по общественному транспорту). Слои (файл, внешний сервис) —
    у каждой пары точек свой: среднее и разброс (10-й и 90-й процентили) отношения слоя к свободной дороге OSRM по
    выборке пар; час без слоя считается по профилю, как в самом расчёте. Часы смен отмечаются для графика.
    """
    import random
    m, t = inst.matrix, inst.matrix.traffic
    n = len(m.points)
    if t.layers:                  # точки, добавленные событием, в слои не входят — сводка по точкам слоя
        n = min(n, len(next(iter(t.layers.values()))))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and m.car_min[i][j] >= 1.]
    if len(pairs) > TRAFFIC_SAMPLE:
        pairs = random.Random(0).sample(pairs, TRAFFIC_SAMPLE)
    hours = []
    for h in range(24):
        layer = t.layers.get((h * 60 // t.layer_step) * t.layer_step) if t.layers else None
        row = {"hour": h, "car": round(t.factor(h * 60, "car"), 2), "public": round(t.factor(h * 60, "public"), 2),
               "from_layer": layer is not None}
        if layer is not None:
            ratios = sorted(layer[i][j] / m.car_min[i][j] for i, j in pairs)
            row.update(car=round(sum(ratios) / len(ratios), 2), p10=round(ratios[len(ratios) // 10], 2),
                       p90=round(ratios[len(ratios) * 9 // 10], 2))
        hours.append(row)
    starts = [e.shift_start.hour + e.shift_start.minute / 60 for e in inst.engineers]
    ends = [e.shift_end.hour + e.shift_end.minute / 60 for e in inst.engineers]
    return {"source": m.time_source, "enabled": t.enabled or bool(t.layers), "weekday": t.weekday,
            "sensitivity": t.sensitivity, "shift": [min(starts, default=0), max(ends, default=24)], "hours": hours}
