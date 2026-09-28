"""Быстрая проверка вставки заявки в маршрут — за O(1) вместо пересчёта всего маршрута.

Профиль показал, что 93% времени поиска уходит в evaluate(), и зовут её 1400 раз
на итерацию: для каждой позиции вставки маршрут пересчитывался с нуля. Это лишнее.

Классический приём для VRPTW: один проход вперёд даёт время окончания каждого визита,
один проход назад — самое позднее прибытие, при котором хвост маршрута ещё выполним.
После этого любая позиция проверяется четырьмя сравнениями.

Время держим в минутах от полуночи: datetime в горячем цикле стоит дороже самой проверки.
"""

from dataclasses import dataclass

INF = 10 ** 9


# Границы окон и смен пересчитываются миллионы раз за поиск, а меняются они только
# при смене инстанса. Профиль показал миллион с лишним вычитаний datetime на сто
# пятьдесят итераций — это дороже самой проверки вставки, ради которой всё затевалось.
_MINUTES_CACHE: dict[tuple, int] = {}


def to_min(dt, day=None) -> int:
    """Минуты от полуночи дня планирования.

    Считать «час × 60 + минуты» нельзя: смена до полуночи или заявка, переходящая
    на следующие сутки, дают ноль или маленькое число, и маршрут становится
    «заканчивающимся раньше, чем начался». Отсчитываем от даты начала смены.
    """
    if day is None:
        return dt.hour * 60 + dt.minute
    key = (dt, day)
    value = _MINUTES_CACHE.get(key)
    if value is None:
        value = int((dt - day).total_seconds() // 60)
        _MINUTES_CACHE[key] = value
    return value


@dataclass
class RouteView:
    """Развёртка маршрута в массивы + границы выполнимости."""
    idx: list[int]          # номера точек в матрице расстояний
    ends: list[int]         # ends[i] — когда закончен i-й визит; ends[-1] = старт смены
    latest: list[int]       # latest[i] — позднее какого времени нельзя приехать на i-й визит
    shift_end: int
    km: float
    day: object = None
    floor_min: int | None = None
    # Таблица минут для режима этого исполнителя, если загруженность не учитывается.
    # Берётся один раз на маршрут: раньше её искали при каждой проверке позиции
    table: object = None


def window_min(req, day) -> tuple[int, int]:
    """Границы окна заявки в минутах, посчитанные один раз на заявку и день."""
    cached = getattr(req, "_window_min", None)
    if cached is not None and len(cached) == 5 and cached[0] is day and cached[1:3] == (req.window_start, req.window_end):
        return cached[3], cached[4]
    ws, we = to_min(req.window_start, day), to_min(req.window_end, day)
    req._window_min = (day, req.window_start, req.window_end, ws, we)
    return ws, we


_DAY_CACHE: dict = {}


def day_zero(eng):
    """Полночь дня, в который начинается смена — общая точка отсчёта для всех времён."""
    start = eng.shift_start
    day = _DAY_CACHE.get(start)
    if day is None:
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        _DAY_CACHE[start] = day
    return day


def build_view(eng, order, inst, mode: str, earliest=None, exempt=None) -> RouteView | None:
    """Считает прямой и обратный проходы. None — если сам маршрут уже недопустим."""
    m, index = inst.matrix, inst.index
    depot = index["depot"]
    n = len(order)
    day = day_zero(eng)

    idx = [index[r.id] for r in order]
    table = m._plain_matrix(mode)          # None, если время зависит от момента выезда
    floor_min = to_min(earliest, day) if earliest else None
    exempt = exempt or set()
    ends = [to_min(eng.shift_start, day)]
    here = depot
    km = 0.0
    for i, req in enumerate(order):
        travel = (table[here][idx[i]] if table is not None
                  else m.minutes(here, idx[i], mode, ends[-1]))
        arrive = ends[-1] + travel
        ws, we = window_min(req, day)
        start = max(arrive, ws)
        if floor_min is not None and req.id not in exempt:
            start = max(start, floor_min)
        if start > we:
            return None
        end = start + req.duration_min
        if end > to_min(eng.shift_end, day):
            return None
        km += m.distance(here, idx[i])
        ends.append(end)
        here = idx[i]

    # Назад: самое позднее прибытие на i-й визит, при котором хвост ещё влезает
    shift_end = to_min(eng.shift_end, day)
    latest = [0] * (n + 1)
    latest[n] = shift_end
    for i in range(n - 1, -1, -1):
        req = order[i]
        # Момент выезда здесь ещё неизвестен, поэтому берём худшее время по всему
        # интервалу, когда выезд возможен. Иначе пик внутри интервала проскакивает мимо
        # оценки, и быстрая проверка начинает расходиться с точным расчётом.
        if i + 1 >= n:
            nxt_travel = 0
        else:
            nxt_travel = m.worst_minutes(idx[i], idx[i + 1], mode,
                                         to_min(req.window_start, day),
                                         to_min(eng.shift_end, day))
        cap = latest[i + 1] - nxt_travel - req.duration_min if i + 1 < n else shift_end - req.duration_min
        latest[i] = min(window_min(req, day)[1], cap)
    # Пол по времени в быстрой проверке применяем только если он не задан: смешивать
    # исключения с обратным проходом рискованно, а вставка после события и так
    # проверяется точным расчётом в _rebuild
    return RouteView(idx=idx, ends=ends, latest=latest, shift_end=shift_end, km=km, day=day,
                     floor_min=floor_min if not exempt else None, table=table)


def insertion_delta(view: RouteView, pos: int, req, inst, eng, mode: str) -> float | None:
    """Прирост пробега от вставки в позицию pos, либо None если так нельзя."""
    m, index = inst.matrix, inst.index
    new = index[req.id]
    prev = index["depot"] if pos == 0 else view.idx[pos - 1]

    table = view.table
    arrive = view.ends[pos] + (table[prev][new] if table is not None
                               else m.minutes(prev, new, mode, view.ends[pos]))
    ws, we = window_min(req, view.day)
    start = max(arrive, ws)
    if view.floor_min is not None:
        start = max(start, view.floor_min)
    if start > we:
        return None
    end = start + req.duration_min
    if end > view.shift_end:
        return None

    if pos < len(view.idx):
        nxt = view.idx[pos]
        onward = table[new][nxt] if table is not None else m.minutes(new, nxt, mode, end)
        if end + onward > view.latest[pos]:
            return None
        return (m.distance(prev, new) + m.distance(new, nxt) - m.distance(prev, nxt))
    return m.distance(prev, new)
