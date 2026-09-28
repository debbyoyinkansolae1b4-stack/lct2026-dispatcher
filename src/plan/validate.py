"""Независимая проверка готового плана.

Смысл в независимости. Планировщик сам себя проверять не может: он строил план по своим
же правилам, и если в них ошибка, он её не увидит. Здесь план перечитывается с нуля —
берутся исходные заявки, исполнители и матрица, и каждое назначение проверяется заново.

Эксперты на защите смотрят именно на это: «все ли обязательные ограничения реально
проверяются алгоритмом» — прямая формулировка из раздела 8.1 ТЗ.
"""

from datetime import timedelta
from collections import Counter

from src.instance import Instance
from src.model import Plan
from src.plan.scheduling import SKILL_TITLES, TRANSPORT_TITLES


def validate(plan: Plan, inst: Instance, not_before=None) -> dict:
    """Свод по каждой группе ограничений и список нарушений.

    not_before — момент события при перепланировании: после него работу нельзя начать
    задним числом. Проверка добавлена после того, как план с визитом в 18:00 при событии
    в 19:00 прошёл валидацию: правило существовало в голове, но не в коде.
    """
    if not_before is None:
        not_before = inst.earliest_start
    checks = {
        "equipment": {"title": "Оборудование на день", "checked": 0, "violations": []},
        "skill": {"title": "Квалификация", "checked": 0, "violations": []},
        "transport": {"title": "Тип транспорта", "checked": 0, "violations": []},
        "window": {"title": "Окно клиента", "checked": 0, "violations": []},
        "shift": {"title": "Смена исполнителя", "checked": 0, "violations": []},
        "order": {"title": "Хронология маршрута", "checked": 0, "violations": []},
        "travel": {"title": "Время в пути учтено", "checked": 0, "violations": []},
        "once": {"title": "Каждая заявка один раз", "checked": 0, "violations": []},
        "not_before": {"title": "Ничего не назначено в прошлое", "checked": 0, "violations": []},
    }
    seen: dict[str, str] = {}

    for eid, route in plan.routes.items():
        eng = next(e for e in inst.engineers if e.id == eid)
        fixed = inst.committed.get(eid, [])
        if route.stops[:len(fixed)] != fixed:
            checks["order"]["violations"].append(f"{eid}: изменён зафиксированный префикс")
        prev_end = eng.shift_start
        prev_point = inst.index["depot"]

        used_equipment = Counter()
        for stop_index, stop in enumerate(route.stops):
            if not_before is not None and stop_index >= len(fixed):
                prev_end = max(prev_end, not_before)
            req = inst.by_id[stop.request_id]
            if req.id in inst.completed and stop_index >= len(fixed):
                checks['once']['violations'].append(f'{req.id}: закрытая заявка назначена повторно')
            for item, quantity in req.equipment.items():
                checks['equipment']['checked'] += 1
                if not isinstance(quantity,int) or isinstance(quantity,bool) or quantity < 0:
                    checks['equipment']['violations'].append(f'{req.id}: некорректная потребность {item}')
                    continue
                used_equipment[item] += quantity
                if used_equipment[item] > eng.equipment.get(item,0):
                    checks['equipment']['violations'].append(f'{eng.id}: превышен дневной запас {item}')


            checks["skill"]["checked"] += 1
            if req.skill not in eng.skills:
                checks["skill"]["violations"].append(
                    f"{req.id}: нужен навык «{SKILL_TITLES[req.skill]}», у {eng.name} его нет")

            if req.required_transport:
                checks["transport"]["checked"] += 1
                if eng.transport != req.required_transport:
                    checks["transport"]["violations"].append(
                        f"{req.id}: нужен «{TRANSPORT_TITLES[req.required_transport]}», "
                        f"у {eng.name} «{TRANSPORT_TITLES[eng.transport]}»")

            checks["window"]["checked"] += 1
            if not (req.window_start <= stop.start <= req.window_end):
                checks["window"]["violations"].append(
                    f"{req.id}: начало в {stop.start:%H:%M} вне окна "
                    f"{req.window_start:%H:%M}–{req.window_end:%H:%M}")

            checks["shift"]["checked"] += 1
            if stop.end > eng.shift_end:
                checks["shift"]["violations"].append(
                    f"{req.id}: работа до {stop.end:%H:%M}, смена {eng.name} "
                    f"до {eng.shift_end:%H:%M}")

            checks["order"]["checked"] += 1
            if stop.start < stop.arrive or stop.end != stop.start + timedelta(minutes=req.duration_min):
                checks["order"]["violations"].append(f"{req.id}: нарушены прибытие или длительность работы")
            if stop.start < prev_end:
                checks["order"]["violations"].append(
                    f"{req.id}: начало в {stop.start:%H:%M} раньше конца предыдущей работы "
                    f"в {prev_end:%H:%M}")

            # Дорога: пересчитываем время в пути по матрице заново, а не верим плану
            checks["travel"]["checked"] += 1
            depart = (prev_end - prev_end.replace(hour=0, minute=0, second=0,
                                                  microsecond=0)).total_seconds() / 60
            need = inst.matrix.minutes(prev_point, inst.index[req.id], eng.transport, depart)
            if stop.arrive + timedelta(seconds=30) < prev_end + timedelta(minutes=need):
                checks["travel"]["violations"].append(
                    f"{req.id}: прибытие в {stop.arrive:%H:%M} быстрее, чем позволяет дорога "
                    f"({need:.0f} мин от предыдущей точки)")

            if not_before is not None:
                checks["not_before"]["checked"] += 1
                started_before_event = stop.start < not_before
                if started_before_event and req.id not in getattr(inst, "earliest_exempt", set()):
                    checks["not_before"]["violations"].append(
                        f"{req.id}: начало в {stop.start:%H:%M}, а событие было "
                        f"в {not_before:%H:%M}")

            checks["once"]["checked"] += 1
            if req.id in seen:
                checks["once"]["violations"].append(
                    f"{req.id}: назначена и {seen[req.id]}, и {eng.name}")
            seen[req.id] = eng.name

            prev_end, prev_point = stop.end, inst.index[req.id]

    # Ничего не потеряно: каждая заявка либо в маршруте, либо в списке с причиной
    explained = {u.request_id for u in plan.unassigned}
    overlap = set(seen) & explained
    cancelled = set(seen) & inst.cancelled
    if overlap or cancelled:
        checks["once"]["violations"].append(
            f"Назначенные заявки одновременно отменены или не назначены: {sorted(overlap | cancelled)}")
    lost = [r.id for r in inst.requests if r.id not in seen and r.id not in explained]
    if lost:
        checks["once"]["violations"].append(
            f"потеряно без объяснения: {', '.join(lost[:5])}")

    total_violations = sum(len(c["violations"]) for c in checks.values())
    return {
        "ok": total_violations == 0,
        "violations": total_violations,
        "assignments": len(seen),
        "unassigned_explained": len(explained),
        "checks": [{"code": k, **v, "ok": not v["violations"]} for k, v in checks.items()],
        "summary": (f"Проверено {len(seen)} назначений по {len(checks)} правилам, нарушений нет"
                    if total_violations == 0 else
                    f"Найдено нарушений: {total_violations}"),
    }
