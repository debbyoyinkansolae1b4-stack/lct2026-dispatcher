"""Проверка ограничений и расчёт расписания маршрута.

Три группы ограничений из ТЗ реализованы как фильтры допустимости, а не как штрафы:
заявка либо помещается в маршрут, либо уходит в список неназначенных с причиной.
Так требует ТЗ — «показать неназначенные заявки и понятным языком указать причину».
"""

import copy
from datetime import timedelta

from src.model import Engineer, Request, Route, Stop
from src.plan.resources import equipment_reason

SKILL_TITLES = {
    "local": "Локальные работы",
    "connect": "Работы на подключение и дозаказы",
    "emergency": "Аварийные работы",
}
TRANSPORT_TITLES = {
    "car": "Автомобиль", "public": "Общественный транспорт",
    "foot": "Пешеход", "bike": "Велосипед",
}


def static_reason(req: Request, eng: Engineer) -> tuple[str, str] | None:
    """Ограничения, которые не зависят от порядка объезда: навык и транспорт."""
    if req.skill not in eng.skills:
        return "skill", (f"у исполнителя нет навыка «{SKILL_TITLES[req.skill]}», "
                         f"который требует эта заявка")
    if req.required_transport and eng.transport != req.required_transport:
        return "transport", (f"заявке нужен транспорт «{TRANSPORT_TITLES[req.required_transport]}», "
                             f"а у исполнителя «{TRANSPORT_TITLES[eng.transport]}»")
    return equipment_reason([req], eng)


def evaluate(eng: Engineer, order: list[Request], matrix, index: dict,
             earliest=None,
             exempt: set[str] | None = None, committed=None) -> tuple[Route | None, tuple[str, str] | None]:
    """Считает расписание маршрута в заданном порядке.

    Возвращает (маршрут, None) либо (None, причина первой невозможности).
    Инженер стартует из офиса в начале смены; возврат в офис не требуется — так в ТЗ.
    """
    resource_error = equipment_reason(order, eng)
    if resource_error:
        return None, resource_error
    fixed = committed or []
    if [r.id for r in order[:len(fixed)]] != [s.request_id for s in fixed]:
        return None, ("history", "изменён уже начатый маршрут")
    stops: list[Stop] = copy.deepcopy(fixed)
    here = index["depot"]
    exempt = exempt or set()
    now = fixed[-1].end if fixed else eng.shift_start
    here = index[fixed[-1].request_id] if fixed else here
    if earliest is not None:
        now = max(now, earliest)

    for req in order[len(fixed):]:
        bad = static_reason(req, eng)
        if bad:
            return None, bad

        there = index[req.id]
        depart = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 60
        travel_min = matrix.minutes(here, there, eng.transport, depart)
        travel_km = matrix.distance(here, there)

        arrive = now + timedelta(minutes=travel_min)
        start = max(arrive, req.window_start)
        if earliest and start < earliest and req.id not in exempt:
            start = earliest        # после события работу нельзя начать задним числом

        if start > req.window_end:          # окно жёсткое: начать работу позже его конца нельзя (Н58)
            return None, ("window", (f"прибытие в {arrive:%H:%M} не попадает в окно "
                                     f"{req.window_start:%H:%M}–{req.window_end:%H:%M}"))

        end = start + timedelta(minutes=req.duration_min)
        if end > eng.shift_end:
            return None, ("shift", (f"работа закончилась бы в {end:%H:%M}, "
                                    f"смена исполнителя до {eng.shift_end:%H:%M}"))

        stops.append(Stop(request_id=req.id, arrive=arrive, start=start, end=end,
                          travel_min=travel_min, travel_km=travel_km))
        now, here = end, there

    return Route(engineer_id=eng.id, stops=stops), None


def route_cost(route: Route) -> float:
    return route.distance_km


def day_start(inst):
    """Самое раннее начало работы в этот день — начало смен: база отсчёта задержки аварий."""
    return min(e.shift_start for e in inst.engineers) if inst.engineers else None


def urgent_delays(route: Route, by_id: dict, since=None) -> list[float]:
    """Задержка начала каждой срочной заявки маршрута, минуты (аварии просили делать как можно
    раньше [15:54]). Отсчёт — от самого раннего возможного начала: max(поступление, начало смен
    since = day_start(inst)). Поступление в статике — начало окна (00:01): без since ~570 минут каждой аварии
    до начала смены неустранимы и заглушают то, чем план управляет (честная база, Михаил 20.09)."""
    out = []
    for s in route.stops:
        req = by_id[s.request_id]
        if req.urgent:
            base = max(req.window_start, since) if since is not None else req.window_start
            out.append(max(0., (s.start - base).total_seconds() / 60.0))
    return out


def urgent_delay_min(route: Route, by_id: dict, since=None) -> float:
    """Суммарная задержка срочных заявок маршрута, минуты."""
    return sum(urgent_delays(route, by_id, since))
