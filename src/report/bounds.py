"""Теоретические границы: насколько план далёк от недостижимого идеала.

«Лучше жадного варианта» — слабое утверждение: жадный вариант плох. Сильное утверждение
звучит иначе: вот граница, которую не перейти никакому алгоритму, и мы в стольких-то
процентах от неё.

Границы получены ослаблением задачи — выбрасываем часть ограничений и решаем то, что
осталось. Решение исходной задачи не может быть лучше решения ослабленной, поэтому
это честные границы, а не оценки на глаз.

  сколько заявок максимум — ослабляем до вместимости смен по навыкам и по окнам;
  сколько исполнителей минимум — ослабляем до объёма работ и до пиковой нагрузки окна;
  сколько километров минимум — ослабляем маршруты до «в каждую точку надо как-то въехать»;
  сколько аварий начать в пределах ориентира — ослабляем до «к каждой сразу едет ближайшая бригада» (src/report/analysis.py).
"""
# refactor: Claude, 26.09.2026 — шаг 6 PORT-PLAN: граница по авариям в ориентир — из вкладки «Анализ» на главную

import math
from collections import defaultdict

from src.instance import Instance
from src.model import Plan


def max_assignable(inst: Instance) -> dict:
    """Верхняя граница числа выполнимых заявок."""
    shift = {e.id: (e.shift_end - e.shift_start).total_seconds() / 60 for e in inst.engineers}

    # Граница по объёму работ: сколько коротких заявок в принципе поместится в смены
    by_skill_requests = defaultdict(list)
    for r in inst.requests:
        by_skill_requests[r.skill].append(r.duration_min)
    total_by_skill = {}
    for skill, durations in by_skill_requests.items():
        capacity = sum(shift[e.id] for e in inst.engineers if skill in e.skills)
        fit, used = 0, 0.0
        for d in sorted(durations):            # короткие вперёд — их влезает больше
            if used + d > capacity:
                break
            used += d
            fit += 1
        total_by_skill[skill] = fit
    volume_bound = sum(total_by_skill.values())

    # Граница по окнам. Осторожно с арифметикой: начать работу нужно внутри окна,
    # а закончить можно и после него. Значит в окне шириной W помещается не W/d работ,
    # а на одну больше: все, кроме последней, должны завершиться внутри окна, последняя
    # может выйти за край. Прежняя формула занижала предел и делала «верхнюю границу»
    # ниже достижимого — то есть неверной.
    window_bound = 0
    by_window = defaultdict(list)
    for r in inst.requests:
        by_window[(r.window_start, r.window_end, r.skill)].append(r)
    for (start, end, skill), group in by_window.items():
        width = (end - start).total_seconds() / 60
        able = sum(1 for e in inst.engineers if skill in e.skills)
        shortest = min(r.duration_min for r in group)
        per_engineer = int(width // max(shortest, 1)) + 1
        window_bound += min(len(group), able * per_engineer)

    bound = min(len(inst.requests), volume_bound, window_bound)
    return {"bound": bound, "by_volume": volume_bound, "by_windows": window_bound,
            "total": len(inst.requests), "by_skill": total_by_skill}


def min_engineers(inst: Instance, assigned_ids: set[str] | None = None) -> dict:
    """Нижняя граница числа исполнителей для заданного набора заявок."""
    reqs = [r for r in inst.requests if assigned_ids is None or r.id in assigned_ids]
    if not reqs:
        return {"bound": 0, "by_volume": 0, "by_peak": 0}

    longest_shift = max((e.shift_end - e.shift_start).total_seconds() / 60
                        for e in inst.engineers)
    work = sum(r.duration_min for r in reqs)
    by_volume = math.ceil(work / longest_shift)

    # Пик окна. Делить работу окна на его ширину нельзя: начать нужно внутри окна,
    # а закончить можно и после. Поэтому на одного исполнителя приходится не больше
    # ширины окна плюс одна самая длинная работа — та, что выходит за границу.
    peak = 0
    groups = defaultdict(float)
    longest = defaultdict(float)
    widths = {}
    for r in reqs:
        key = (r.window_start, r.window_end)
        groups[key] += r.duration_min
        longest[key] = max(longest[key], r.duration_min)
        widths[key] = (r.window_end - r.window_start).total_seconds() / 60
    for key, minutes in groups.items():
        capacity = widths[key] + longest[key]
        if capacity > 0:
            peak = max(peak, math.ceil(minutes / capacity))

    return {"bound": max(by_volume, peak), "by_volume": by_volume, "by_peak": peak}


def min_distance(inst: Instance, assigned_ids: set[str] | None = None) -> dict:
    """Нижняя граница суммарного пробега.

    Две оценки, берём сильнейшую.

    Въезды: в каждую точку надо откуда-то приехать, дешевле самого дешёвого въезда
    не выйдет. Оценка честная, но слабая: в плотном городе ближайший сосед всегда рядом.

    Остовное дерево: все маршруты выходят из одного офиса и никуда не возвращаются,
    значит вместе они образуют дерево на множестве «офис плюс назначенные точки».
    Любое дерево не дешевле минимального остовного, поэтому его вес — нижняя граница
    пробега. Она заметно сильнее первой и учитывает дальние точки вроде Каширы.
    """
    ids = [r.id for r in inst.requests if assigned_ids is None or r.id in assigned_ids]
    if not ids:
        return {"bound": 0.0, "by_arcs": 0.0, "by_tree": 0.0}

    idx = inst.index
    nodes = [idx["depot"]] + [idx[i] for i in ids]

    by_arcs = sum(min(inst.matrix.distance(i, j) for i in nodes if i != j) for j in nodes[1:])

    # Дерево строим по симметричному весу min(d(i,j), d(j,i)). Дороги односторонние,
    # и расстояние туда не равно расстоянию обратно; обычный Прим по одному направлению
    # может дать «нижнюю границу» выше настоящего маршрута — то есть неверную.
    def w(i, j):
        return min(inst.matrix.distance(i, j), inst.matrix.distance(j, i))

    unseen = set(nodes[1:])
    nearest = {j: w(nodes[0], j) for j in unseen}
    by_tree = 0.0
    while unseen:
        j = min(unseen, key=lambda x: nearest[x])
        by_tree += nearest[j]
        unseen.discard(j)
        for k in unseen:
            d = w(j, k)
            if d < nearest[k]:
                nearest[k] = d

    return {"bound": round(max(by_arcs, by_tree), 1), "by_arcs": round(by_arcs, 1),
            "by_tree": round(by_tree, 1), "nodes": len(nodes)}


def gap_report(plan: Plan, inst: Instance) -> dict:
    """План против границ: в скольких процентах от недостижимого мы находимся."""
    assigned = {s.request_id for r in plan.routes.values() for s in r.stops}
    cap = max_assignable(inst)
    eng = min_engineers(inst, assigned)
    dist = min_distance(inst, assigned)

    def pct(value, bound):
        return round((value - bound) / bound * 100, 1) if bound else 0.0

    return {
        "urgent": urgent_within(plan, inst),
        "assigned": {"plan": len(assigned), "bound": cap["bound"], "total": cap["total"],
                     "gap_pct": pct(cap["bound"], len(assigned)) if len(assigned) else 100.0,
                     "detail": f"по объёму работ {cap['by_volume']}, по окнам {cap['by_windows']}"},
        "engineers": {"plan": plan.used_engineers, "bound": eng["bound"],
                      "gap_pct": pct(plan.used_engineers, eng["bound"]),
                      "detail": f"по объёму {eng['by_volume']}, по пиковому окну {eng['by_peak']}"},
        "distance": {"plan": round(plan.total_km, 1), "bound": dist["bound"],
                     "gap_pct": pct(plan.total_km, dist["bound"]),
                     "detail": f"остовное дерево {dist['by_tree']} км, "
                               f"сумма въездов {dist['by_arcs']} км"},
    }


def urgent_within(plan: Plan, inst: Instance) -> dict | None:
    """Аварий, начатых в пределах ориентира: план против границы «к каждой сразу едет ближайшая бригада»
    (Н68, src/report/analysis.py). Аварий или бригад с допуском нет — None. Считается за миллисекунды (300×40 — 8 мс)."""
    from src.plan import solver
    from src.report import analysis, metrics
    bound = analysis.urgent_bounds(inst, solver.URGENT_TARGET_MIN)
    if not bound["urgent"] or not bound["crews"]:
        return None
    wait = metrics.urgent_wait(plan, inst)
    return {"plan": wait["assigned"] - wait["over_target"], "bound": bound["alone_within"], "total": bound["urgent"],
            "target_min": round(solver.URGENT_TARGET_MIN)}
