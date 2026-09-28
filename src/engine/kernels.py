"""Скомпилированные ядра восстановления плана: только числа, только массивы.

Здесь нет объектов задачи, настроек, логов, torch и событий. Каждое ядро — чистая
функция над массивами CompiledInstance (src/engine/compiled.py). Питоновская проверка
маршрута (src/plan/scheduling.py) остаётся независимым эталоном: её результат ядра обязаны
повторять, и тесты сверяют их между собой.

Порядок операций и типы чисел перенесены из src/alt/native_*.py без изменений:
от них зависит побайтовое совпадение траектории поиска с эталоном шага 0.

Время в минутах от полуночи дня исполнителя. times[b, i, j] — минуты в пути из i в j
при выезде в корзину b шириной step минут; km[i, j] — километры.

Маршрут для ядер — подвижный хвост: он начинается в точке origin в момент ready
(RouteState: офис и начало смены утром; последняя закреплённая точка и момент готовности
после события). Закреплённое начало ядра не видят.
"""
# refactor: Claude, 21.09.2026 — перенос из src/alt/native_insertion.py и native_repair.py
# refactor: Claude, 21.09.2026 — шаг 5г.2: origin/ready вместо depot/shift_start, общая сумма маршрута
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def insertion_options(indices, ends, latest, shift_end, floor, new, ws, we, duration, origin, km, times, step):
    """Допустимые позиции вставки точки new в хвост и прирост километров для каждой.

    indices — узлы хвоста, ends[p] — конец работы перед позицией p (ends[0] — момент
    готовности в точке origin), latest[p] — самое позднее начало работы на позиции p,
    при котором хвост ещё укладывается. floor — нижняя граница начала работы (-inf, если нет).
    """
    n = len(indices)
    deltas = np.empty(n + 1)
    valid = np.zeros(n + 1, dtype=np.bool_)
    for pos in range(n + 1):
        prev = origin if pos == 0 else indices[pos - 1]
        arrive = ends[pos] + times[int(ends[pos] // step) % len(times), prev, new]
        start = max(arrive, ws)
        if floor > start:
            start = floor
        if start > we:
            continue
        end = start + duration
        if end > shift_end:
            continue
        if pos < n:
            nxt = indices[pos]
            onward = times[int(end // step) % len(times), new, nxt]
            if end + onward > latest[pos]:
                continue
            deltas[pos] = km[prev, new] + km[new, nxt] - km[prev, nxt]
        else:
            deltas[pos] = km[prev, new]
        valid[pos] = True
    return deltas, valid


@njit(cache=True, fastmath=False)
def route_view(order, nodes, windows, durations, ready, shift_end, origin, times, step):
    """Прямой и обратный проход по хвосту из origin в момент ready: концы работ и самые поздние начала.

    Возвращает (допустим ли маршрут, узлы, ends, latest). Худшее время в пути между
    соседями берётся так же, как TravelMatrix.worst_minutes: концы интервала и часовые
    отметки внутри него. Эту трактовку матриц, зависящих от времени, менять нельзя.
    """
    n = len(order)
    indices = np.empty(n, np.int64)
    ends = np.empty(n + 1)
    latest = np.empty(n + 1)
    ends[0] = ready
    here = origin
    for i in range(n):
        q = order[i]
        node = nodes[q]
        indices[i] = node
        arrival = ends[i] + times[int(ends[i] // step) % len(times), here, node]
        start = max(arrival, windows[q, 0])
        end = start + durations[q]
        if start > windows[q, 1] or end > shift_end:
            return False, indices, ends, latest
        ends[i + 1] = end
        here = node
    latest[n] = shift_end
    for i in range(n - 1, -1, -1):
        q = order[i]
        travel = 0.
        if i + 1 < n:
            lo = int(min(windows[q, 0], shift_end))
            hi = int(max(windows[q, 0], shift_end))
            a = indices[i]
            b = indices[i + 1]
            travel = max(times[(lo // step) % len(times), a, b], times[(hi // step) % len(times), a, b])
            for h in range(lo // 60, hi // 60 + 2):
                moment = h * 60
                if lo <= moment <= hi:
                    travel = max(travel, times[(moment // step) % len(times), a, b])
            cap = latest[i + 1] - travel - durations[q]
        else:
            cap = shift_end - durations[q]
        latest[i] = min(windows[q, 1], cap)
    return True, indices, ends, latest


@njit(cache=True, fastmath=False)
def route_column(indices, ends, latest, shift_end, floor, nodes, windows, durations,
                 eligible, need, stock, used, origin, km, times, step, opening):
    """Для одного маршрута — допустимые вставки всех заявок: значения, позиции, их число.

    opening — штраф за открытие пустого маршрута, прибавляется к каждому значению.
    """
    q = len(nodes)
    width = len(indices) + 1
    values = np.zeros((q, width))
    positions = np.zeros((q, width), np.int64)
    counts = np.zeros(q, np.int64)
    for i in range(q):
        if not eligible[i]:
            continue
        fits = True
        for k in range(len(stock)):
            if used[k] + need[i, k] > stock[k]:
                fits = False
                break
        if not fits:
            continue
        deltas, valid = insertion_options(indices, ends, latest, shift_end, floor, nodes[i],
                                          windows[i, 0], windows[i, 1], durations[i], origin, km, times, step)
        for j in range(width):
            if valid[j]:
                at = counts[i]
                values[i, at] = deltas[j] + opening
                positions[i, at] = j
                counts[i] += 1
    return values, positions, counts


@njit(cache=True, fastmath=False)
def _neumaier(total, carry, x):
    """Один шаг суммы sum() CPython 3.12+ для float: алгоритм Ноймайера."""
    t = total + x
    if abs(total) >= abs(x):
        carry += (total - t) + x
    else:
        carry += (x - t) + total
    return t, carry


@njit(cache=True, fastmath=False)
def _finish(total, carry):
    if carry != 0. and np.isfinite(carry):
        return total + carry
    return total


@njit(cache=True, fastmath=False)
def history_deltas(values, positions, counts, history_km, indices, origin, nodes, km, opening, lock):
    """Прирост вставки после события — как в solver._insertion_candidates, до последнего бита.

    Питоновский путь считает прирост разностью пробега маршрута с вставкой и без:
    sum(s.travel_km) по всем остановкам, включая закреплённые. Формула
    km[p,new]+km[new,n]-km[p,n] отличается от неё в последнем знаке, и этого хватает,
    чтобы регрет выбрал другую заявку. Здесь повторена сама сумма (Ноймайер, как
    sum() в CPython 3.12+) в том же порядке слагаемых. Значения колонки переписываются
    на месте; позиции — полные, хвост начинается с lock.
    """
    base = route_total(history_km, indices, origin, -1, -1, km)
    for i in range(len(counts)):
        for a in range(counts[i]):
            values[i, a] = (route_total(history_km, indices, origin, nodes[i], positions[i, a] - lock, km)
                            - base) + opening


@njit(cache=True, fastmath=False)
def route_total(history_km, indices, origin, new, at, km):
    """Пробег полного маршрута, как sum(s.travel_km) в Python: история, затем хвост из origin.

    new, at — вставить точку new на позицию at хвоста (at < 0 — без вставки). Пустой
    маршрут — 0, как сумма пустого списка.
    """
    if len(history_km) + len(indices) == 0 and at < 0:
        return 0.
    total = 0.
    carry = 0.
    for x in history_km:
        total, carry = _neumaier(total, carry, x)
    prev = origin
    for t in range(len(indices) + 1):
        if t == at:
            total, carry = _neumaier(total, carry, km[prev, new])
            prev = new
        if t < len(indices):
            total, carry = _neumaier(total, carry, km[prev, indices[t]])
            prev = indices[t]
    return _finish(total, carry)


@njit(cache=True, fastmath=False)
def select_regret(values, positions, counts, active, priorities, urgent, eid_rank, regret, jitter):
    """Какую заявку вставить следующей и куда: regret-k с приоритетом и срочностью.

    Порядок сравнения вариантов повторяет сортировку options в solver.construct:
    прирост, позиция, строковый ID исполнителя (eid_rank). jitter — множители шума
    в порядке обхода (пустой массив — без шума). Возвращает (заявка, исполнитель,
    позиция, сколько множителей израсходовано).
    """
    n, m, _ = values.shape
    offsets = np.empty((n, m), np.int64)
    total = 0
    for q in range(n):
        for e in range(m):
            offsets[q, e] = total
            if active[q]:
                total += counts[q, e]
    winq = -1
    wine = -1
    winpos = -1
    winpriority = -1
    winregret = 0.
    windelta = 0.
    for q in range(n):
        if not active[q]:
            continue
        ds = np.empty(m)
        ps = np.empty(m, np.int64)
        es = np.empty(m, np.int64)
        size = 0
        for e in range(m):
            if counts[q, e] == 0:
                continue
            best = 0.
            pos = -1
            for j in range(counts[q, e]):
                delta = values[q, e, j]
                if len(jitter):
                    delta *= jitter[offsets[q, e] + j]
                if pos < 0 or delta < best:
                    best = delta
                    pos = positions[q, e, j]
            # Вставка в отсортированный список: прирост, позиция, ID исполнителя.
            at = size
            while at > 0 and (best < ds[at - 1] or (best == ds[at - 1] and (
                    pos < ps[at - 1] or (pos == ps[at - 1] and eid_rank[e] < eid_rank[es[at - 1]])))):
                ds[at] = ds[at - 1]
                ps[at] = ps[at - 1]
                es[at] = es[at - 1]
                at -= 1
            ds[at] = best
            ps[at] = pos
            es[at] = e
            size += 1
        if size == 0:
            continue
        second = ds[min(regret - 1, size - 1)] if size > 1 else ds[0] + 1e6
        score = -(second - ds[0]) - (1e6 if urgent[q] else 0.)
        priority = priorities[q]
        if winq < 0 or priority > winpriority or (priority == winpriority and (
                score < winregret or (score == winregret and ds[0] < windelta))):
            winq = q
            wine = es[0]
            winpos = ps[0]
            winpriority = priority
            winregret = score
            windelta = ds[0]
    return winq, wine, winpos, total


@njit(cache=True, fastmath=False)
def random_jitter(state, index, count, noise):
    """count множителей шума 1 ± noise — ровно те же, что дал бы random.Random.uniform.

    Повторяет вихрь Мерсенна CPython (MT19937, 27+26 бит на 53-битное число), чтобы
    нативное восстановление давало те же планы, что питоновское. Замена на генератор
    numpy меняет траекторию и отложена до шага 9 (REFACTOR-PLAN.md).
    """
    out = np.empty(count)
    for i in range(count):
        parts = np.empty(2, np.uint64)
        for part in range(2):
            if index >= 624:
                for k in range(624):
                    y = (state[k] & np.uint64(0x80000000)) | (state[(k + 1) % 624] & np.uint64(0x7fffffff))
                    state[k] = state[(k + 397) % 624] ^ (y >> np.uint64(1)) ^ (
                        np.uint64(0x9908b0df) if y & np.uint64(1) else np.uint64(0))
                index = 0
            y = state[index]
            index += 1
            y ^= y >> np.uint64(11)
            y ^= (y << np.uint64(7)) & np.uint64(0x9d2c5680)
            y ^= (y << np.uint64(15)) & np.uint64(0xefc60000)
            y ^= y >> np.uint64(18)
            parts[part] = y
        u = float(parts[0] >> np.uint64(5)) * 67108864. + float(parts[1] >> np.uint64(6))
        u /= 9007199254740992.
        out[i] = 1. + (-noise + (noise - (-noise)) * u)
    return out, index
