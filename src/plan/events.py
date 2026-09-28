"""Перепланирование в течение дня.

ТЗ требует одно событие на выбор; делаем все три, они всё равно сводятся к одной механике:
меняется состав заявок или исполнителей, часть работы уже сделана и трогать её нельзя,
остальное пересобираем и показываем разницу.

Что уже нельзя трогать: заявки, которые на момент события начаты или завершены. Инженер,
которого сняли, теряет только будущие визиты — то, что он уже сделал, остаётся в плане.
"""

# refactor: Claude, 22.09.2026 — Н30: стартовый план события нативным восстановлением
# refactor: Claude, 22.09.2026 — шаг 8а: поиск после события обязателен, ALNS удалён
import copy
import math
from uuid import uuid4
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.data import normatives
from src.instance import Instance
from src.model import Plan, Request, Unassigned
from src.engine.repair import NativeRepair


@dataclass
class Event:
    kind: str                     # urgent | cancel | engineer_off
    at: datetime
    request_id: str | None = None
    engineer_id: str | None = None
    payload: dict | None = None   # для срочной — поля новой заявки
    reserve: bool = False         # эскалация: можно вызвать бригаду, не вышедшую утром (без заявок в плане)


def off_duty(inst: Instance, plan: Plan) -> list[str]:
    """Не вышедшие сегодня: доступные бригады без единой заявки в текущем плане.

    Ответ экспертов (message687, 24.09.2026): новая заявка или авария сама по себе не вызывает из дома того, кто по
    первоначальному плану не работает; привлечь его — отдельная эскалация (Event.reserve).
    """
    return [e.id for e in inst.engineers
            if e.available and not (plan.routes.get(e.id) and plan.routes[e.id].stops)]


def frozen_ids(plan: Plan, at: datetime) -> set[str]:
    """Визиты с уже начатым выездом, включая ожидание окна и работу."""
    return {s.request_id for r in plan.routes.values() for s in r.stops
            if s.arrive - timedelta(minutes=s.travel_min) < at}


def apply(inst: Instance, plan: Plan, event: Event, search) -> tuple[Plan, Instance]:
    """Apply to private copies; failure never changes the input instance or plan.

    search(inst, warm, frozen) -> Plan — поиск после подготовки события: портфель приложения
    или src.search.replan.population_search. Подготовка события, проверки результата
    и откат при ошибке — здесь, общие для любого поиска.
    """
    if search is None:
        raise ValueError("Нужна функция поиска после события")
    if plan is None:
        raise ValueError("Сначала постройте план")
    if not isinstance(event.at, datetime):
        raise ValueError("Нужно время события")
    if inst.earliest_start is not None and event.at < inst.earliest_start:
        raise ValueError("Время события не может идти назад")
    if event.kind in {"cancel", "complete"} and event.request_id not in {r.id for r in inst.requests}:
        raise ValueError("Заявка отсутствует в активном плане")
    if event.kind == "engineer_off" and not any(
            e.id == event.engineer_id and e.available for e in inst.engineers):
        raise ValueError("Инженер отсутствует или уже недоступен")
    if event.kind == 'complete':
        stop = next((s for route in plan.routes.values() for s in route.stops if s.request_id == event.request_id), None)
        if stop is None or stop.end > event.at:
            raise ValueError('Подтверждение завершения поддерживается после планового окончания назначенной работы')
    before = plan
    inst, plan = copy.deepcopy(inst), copy.deepcopy(plan)
    inst.previous_owner = {s.request_id: eid for eid, r in plan.routes.items() for s in r.stops}
    keep = frozen_ids(plan, event.at)
    if event.kind == "cancel" and event.request_id in keep:
        stop = next(s for route in plan.routes.values() for s in route.stops if s.request_id == event.request_id)
        if stop.end <= event.at:
            raise ValueError("Заявка по плану уже выполнена — отменять нечего")
        raise ValueError("Выезд уже начат: отмена с прерыванием пути пока не поддерживается")
    inst.committed = {eid: [copy.deepcopy(s) for s in r.stops if s.request_id in keep]
                      for eid, r in plan.routes.items()}
    orders = {eid: [inst.by_id[s.request_id] for s in stops]
              for eid, stops in inst.committed.items()}

    if event.kind == "cancel":
        # Убрать из маршрутов мало: заявка остаётся в списке заявок участка, попадает
        # в пул восстановления и назначается снова. Выкидываем её из планирования.
        orders = {eid: [r for r in seq if r.id != event.request_id] for eid, seq in orders.items()}
        keep.discard(event.request_id)
        inst.requests = [r for r in inst.requests if r.id != event.request_id]
        inst.cancelled.add(event.request_id)
        inst.by_id[event.request_id].status = "Отменена"

    elif event.kind == 'complete':
        inst.by_id[event.request_id].status = 'Завершено'
        inst.completed.add(event.request_id)
        inst.requests = [r for r in inst.requests if r.id != event.request_id]

    elif event.kind == "engineer_off":
        eng = next(e for e in inst.engineers if e.id == event.engineer_id)
        eng.available = False
        # Будущие визиты снятого инженера возвращаются в общий котёл
        orders[eng.id] = [r for r in orders[eng.id] if r.id in keep]

    elif event.kind == "urgent":
        req = _new_request(inst, event)
        inst.requests.append(req)
        inst.by_id[req.id] = req
        inst.index[req.id] = len(inst.matrix.points)
        _extend_matrix(inst, req)

    else:
        raise ValueError(f"неизвестное событие: {event.kind}")

    # Всё, что не в маршрутах, идёт на пересборку. Раньше момента события начинать нельзя.
    inst.earliest_start = event.at
    inst.earliest_exempt = set(keep)          # начатое до события не сдвигаем
    pool = [r for r in inst.requests
            if r.id not in {x.id for seq in orders.values() for x in seq}]
    # Стартовый план — нативным восстановлением: без шума тот же план, что solver.construct
    # (tests/test_native_events.py), но за сотые доли секунды вместо 0.6–1.3 с (Н30).
    # Закреплённое начало берётся из inst.committed — отдельный locked не нужен.
    # Не вышедшие утром в перестройке не участвуют, пока диспетчер не разрешил вызов с выходного (off_duty);
    # после неё снова доступны — для следующего события с эскалацией. refactor: Claude, 26.09.2026 — message687
    resting = [] if event.reserve else [e for e in inst.engineers if e.id in set(off_duty(inst, plan))]
    for e in resting:
        e.available = False
    warm = NativeRepair(inst)(inst, pool=pool, orders=orders)
    result = search(inst, warm, keep)
    from src.plan.validate import validate
    report = validate(result, inst)
    for e in resting:
        e.available = True
    if resting:                                   # подсказка диспетчеру: есть кого вызвать с выходного
        by_id = {r.id: r for r in inst.requests}
        for u in result.unassigned:
            r = by_id.get(u.request_id)
            n = sum(1 for e in resting if r is not None and r.skill in e.skills)
            if n:
                u.reason += (f". Не вышедшие сегодня бригады с этим навыком ({n}) без эскалации не вызываются — "
                             f"отметьте «вызвать бригаду с выходного»")
    if not report["ok"]:
        raise ValueError("Перепланирование нарушает ограничения: " + str(report["checks"]))
    old = {s.request_id: (eid, s) for eid, route in before.routes.items() for s in route.stops}
    new = {s.request_id: (eid, s) for eid, route in result.routes.items() for s in route.stops}
    if any(old[rid] != new.get(rid) for rid in keep):
        raise ValueError("Перепланирование изменило уже начатую работу")
    return result, inst


def _new_request(inst: Instance, event: Event) -> Request:
    p = dict(event.payload or {})
    rid = p.get("id") or f"urgent-{uuid4().hex}"
    if not isinstance(rid, str) or not rid.strip() or rid in inst.by_id:
        raise ValueError("ID новой заявки должен быть уникальным")
    for key, limit in (("lat", 90), ("lon", 180)):
        value = p.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
            raise ValueError(f"Некорректная координата {key}")
    # Срочная заявка — авария: на адресе норматив без резерва дороги, дорогу считает маршрут.
    norms = normatives.load()
    expected = normatives.on_site_minutes(norms, "emergency")
    duration = p.get("duration_min", expected)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Длительность должна быть положительным числом")
    if duration != expected:
        raise ValueError(f"Норматив аварии — {normatives.base_minutes(norms, 'emergency')} минут, "
                         f"из них {expected} на адресе; дорога считается по маршруту")
    start = p.get("window_start", event.at)
    end = p.get("window_end", event.at.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
    if not isinstance(start, datetime) or not isinstance(end, datetime) or start < event.at or end < start:
        raise ValueError("Некорректное окно срочной заявки")
    if p.get("skill", "emergency") not in {"local", "connect", "emergency"}:
        raise ValueError("Неизвестный навык")
    if p.get("required_transport") not in {None, "car", "public", "foot", "bike"}:
        raise ValueError("Неизвестный транспорт")
    return Request(
        id=rid,
        bk_type=p.get("bk_type", "Глобальная проблема"),
        hd_type=p.get("hd_type", "Авария"),
        window_start=start,
        window_end=end,
        district=p.get("district", "—"),
        address=p.get("address", ""),
        address_query=p.get("address", ""),
        lat=p["lat"], lon=p["lon"],
        skill=p.get("skill", "emergency"),
        duration_min=int(duration),
        urgent=True,
        equipment={"repair_kit": 1},
        required_transport=p.get("required_transport"),
    )


def _extend_matrix(inst: Instance, req: Request) -> None:
    """Дописываем строку и столбец в матрицу под новую точку.

    Считаем по прямой с коэффициентом извилистости: дёргать OSRM в момент, когда диспетчер
    ждёт новый план, дороже, чем принять погрешность на одной точке. В README это допущение.
    """
    from src.travel import DETOUR, haversine_km

    m = inst.matrix
    p_new = (req.lat, req.lon)
    for i, p in enumerate(m.points):
        km = haversine_km(p, p_new) * DETOUR
        m.km[i].append(km)
        m.car_min[i].append(km / 24.0 * 60.0)
    m.points.append(p_new)
    row_km = [m.km[i][-1] for i in range(len(m.km))] + [0.0]
    m.km.append(row_km)
    m.car_min.append([km / 24.0 * 60.0 for km in row_km])


def describe(event: Event, inst: Instance) -> str:
    if event.kind == "complete":
        return f"В {event.at:%H:%M} диспетчер подтвердил завершение заявки {event.request_id}."
    if event.kind == "cancel":
        r = inst.by_id.get(event.request_id)
        what = f"{r.district}, {r.hd_type}" if r else event.request_id
        return f"В {event.at:%H:%M} клиент отменил заявку {event.request_id} ({what})."
    if event.kind == "engineer_off":
        eng = next((e for e in inst.engineers if e.id == event.engineer_id), None)
        return (f"В {event.at:%H:%M} из работы выбывает: {eng.name if eng else event.engineer_id}. "
                f"Невыполненные заявки передаются другим исполнителям.") + _escalation(event)
    p = event.payload or {}
    return (f"В {event.at:%H:%M} поступила срочная заявка: {p.get('hd_type', 'Авария')}, "
            f"{p.get('district', '')} — её нужно встроить в текущий день.") + _escalation(event)


def _escalation(event: Event) -> str:
    return " Разрешён вызов бригады с выходного." if event.reserve else ""
