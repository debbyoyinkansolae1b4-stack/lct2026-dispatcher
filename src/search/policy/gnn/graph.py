"""Граф состояния для сети: узлы исполнителей, маршрутов и заявок, связи, подсказки, маски.

graph() строит признаки узлов и подсказки эвристики по плану-родителю; enrich_graph()
добавляет точные допустимые переносы (через сервис маршрута — кэш восстановления),
рёбра для обмена сообщениями и маску переноса. После события граф строится по проекции
на подвижную часть (projection.py): закреплённых заявок среди узлов нет, g['request_ids']
(только при истории) переводит индекс узла в полный индекс inst.requests.
"""
# refactor: Claude, 21.09.2026 — из src/alt/population_gnn.py и planner_population.py; стиль, логика без изменений
# refactor: Claude, 21.09.2026 — шаг 5б: вставки после события из нативного кэша
# refactor: Claude, 21.09.2026 — шаг 5г: история из RouteState, один сервис вставок
# refactor: Claude, 21.09.2026 — шаг 5г.3: граф по проекции на подвижную часть
# refactor: Claude, 22.09.2026 — шаг 8а: EQUIPMENT из features.py (policy_net уходит)
import torch

from src.engine.route_state import RouteState
from src.search.policy.gnn.projection import project
from src.search.policy.gnn.features import EQUIPMENT
from src.plan.resources import equipment_reason
from src.plan.scheduling import day_start, static_reason
from src.search.policy.gnn.context import encode_context

SKILLS = ('local', 'connect', 'emergency')
TRANSPORTS = ('car', 'public', 'foot', 'bike')

# Расстояние до рекорда, покомпонентно. Сеть видит только текущий план, а он ходит
# по отжигу (anneal принимает ухудшения) и может быть заметно хуже лучшего найденного.
# Решение «рвать осторожно или ломать сильно» зависит именно от этого, а вход в обоих
# случаях был одинаковый.
RECORD_GAP = ('gap_emergencies', 'gap_connections', 'gap_assigned', 'gap_urgent_late',
              'gap_fleet', 'gap_km', 'gap_delay')

# Типы связей графа: e — исполнитель, r — маршрут, q — заявка.
RELATIONS = [('e', 'owns', 'r'), ('r', 'owner', 'e'), ('q', 'assigned', 'r'),
             ('r', 'contains', 'q'), ('q', 'eligible', 'e'), ('e', 'can_serve', 'q'),
             ('q', 'near', 'q'), ('q', 'next', 'q'), ('q', 'previous', 'q')]


def _normalize_rows(a):
    return a / a.sum(1, keepdim=True).clamp_min(1)


def _request_features(r, entry, depot, detour):
    """Признаки заявки. entry — (исполнитель, позиция, визит) для назначенной, иначе None."""
    dep_lat, dep_lon = depot
    stop = entry[2] if entry else None
    return [*[float(r.skill == k) for k in SKILLS], float(r.priority), float(r.duration_min),
            float(r.window_start.hour), (r.window_end - r.window_start).total_seconds() / 3600,
            r.lat - dep_lat, r.lon - dep_lon, float(entry is not None),
            (stop.start - r.window_start).total_seconds() / 60 if stop else 0.,
            (r.window_end - stop.start).total_seconds() / 60 if stop else 0.,
            stop.travel_km if stop else 0., float(entry[1]) if entry else 0.,
            *detour,
            *[float(r.equipment.get(k, 0)) for k in EQUIPMENT], float(bool(r.required_transport))]


def _engineer_features(e):
    return [*[float(k in e.skills) for k in SKILLS],
            *[float(e.transport == m) for m in TRANSPORTS], float(e.available),
            float(e.shift_start.hour), (e.shift_end - e.shift_start).total_seconds() / 3600,
            *[float(e.equipment.get(k, 0)) for k in EQUIPMENT]]


def _geographic_neighbours(reqs):
    """Пять ближайших заявок по прямой с весом exp(-d/0.1), без петель."""
    coords = torch.tensor([[r.lat, r.lon] for r in reqs])
    d = torch.cdist(coords, coords)
    geo = torch.zeros_like(d)
    nearest = d.topk(min(5, len(reqs)), largest=False).indices
    geo.scatter_(1, nearest, torch.exp(-d.gather(1, nearest) / .1))
    geo.fill_diagonal_(0)
    return geo


def _heuristic_route_hint(routes):
    # Колонки routes (policy_net.encode + context.ROUTE_CONTEXT): 0 — число остановок,
    # 1 — километры, 23 — минимальный запас окна по маршруту (минуты).
    return 1 / (routes[:, 0] + 1) + routes[:, 1].clamp_min(0) / 50 + 1 / (1 + routes[:, 23].clamp_min(0) / 120)


def _heuristic_request_hint(reqs, owner, compatible):
    # Срочные и неназначенные вперёд, узкое окно и мало подходящих исполнителей — тоже.
    return torch.tensor([(1 + 2 * r.priority) * (2 if r.id not in owner else 1)
                         / (1 + (r.window_end - r.window_start).total_seconds() / 3600)
                         for r in reqs]) / compatible.sum(0).clamp_min(1)


def graph(inst, plan, epoch=0, epochs=100, stall=0, best=None, penalty_hints=False,
          penalty_scope='both', detour_feature=False, state=None):
    """Граф плана-родителя для сети и масок выбора действия (патч learned_full: зрение цели, подсказки по строкам).

    penalty_hints — подсказки из разложения plan_scalar (penalty_attribution) вместо
    эвристических формул; penalty_scope='aim' оставляет цель головы на прежней формуле.
    best — добавить в глобальные признаки расстояние до рекорда (RECORD_GAP).
    Признаки сырые: нормирует их политика сети по самой задаче (normalization.py).
    """
    state = state or RouteState(inst)
    view = project(inst, plan, state)
    routes, _, glob = encode_context(view.inst, view.plan, epoch, epochs, stall)
    if state.has_history:
        # Глобальные — по полному плану: это состояние критерия, вклад истории в нём постоянен.
        glob = encode_context(inst, plan, epoch, epochs, stall)[2]
        # Колонка 4 — «исполнитель в составе»: очистка хвоста не убирает уже задействованного.
        used = torch.tensor([e.id in view.already_used for e in inst.engineers], dtype=routes.dtype)
        routes[:, 4] = torch.maximum(routes[:, 4], used)
    # Критерий — по полному плану: разрыв до рекорда сравнивает полное с полным (Н33).
    full_cost = plan.cost_tuple()
    # База задержки аварий — начало смен исходной задачи, как у цели: в проекции shift_start —
    # готовность хвостов после события (Н59).
    since = day_start(inst)
    inst, plan = view.inst, view.plan
    reqs, engs = inst.requests, inst.engineers
    if not reqs or not engs:
        raise ValueError('Empty instance')
    owner = {s.request_id: (i, j, s) for i, e in enumerate(engs) for j, s in enumerate(plan.routes[e.id].stops)}
    depot = inst.matrix.points[inst.index['depot']]
    detours = request_detours(inst, plan, origin=view.origin) if detour_feature else {}
    indices = {r.id: i for i, r in enumerate(reqs)}
    # Патч learned_full (Михаил 25.09, «зрение»): признаки в терминах цели. Ожидание аварии — от max(начало окна,
    # начало смен), как в цели (scheduling.urgent_delays); ориентир — строка «сверх ориентира». Крюк — всегда.
    from src.plan import solver as _solver
    target = _solver.URGENT_TARGET_MIN
    goal_detour = request_detours(inst, plan, origin=view.origin)
    waits = {}
    for r in reqs:
        entry = owner.get(r.id)
        if r.urgent and entry is not None:
            base = max(r.window_start, since) if since is not None else r.window_start
            waits[r.id] = max(0., (entry[2].start - base).total_seconds() / 60)

    assignment = torch.zeros(len(engs), len(reqs))
    req_features = []
    for i, r in enumerate(reqs):
        entry = owner.get(r.id)
        if entry:
            assignment[entry[0], i] = 1
        wait = waits.get(r.id)
        goal = [wait if wait is not None else 0., float(wait is not None and wait > target),
                (target - wait) if wait is not None else 0., goal_detour.get(r.id, 0.)]
        req_features.append(_request_features(r, entry, depot, [detours.get(r.id, 0.)] if detour_feature else []) + goal)
    sequence = torch.zeros(len(reqs), len(reqs))
    for route in plan.routes.values():
        for a, b in zip(route.stops, route.stops[1:]):
            # Завершённые заявки остаются в маршруте, но не в активных: у них нет узла.
            if a.request_id in indices and b.request_id in indices:
                sequence[indices[a.request_id], indices[b.request_id]] = 1
                sequence[indices[b.request_id], indices[a.request_id]] = 1
    engineers = torch.tensor([_engineer_features(e) for e in engs])
    compatible = torch.tensor([[float(e.available and static_reason(r, e) is None
                                      and equipment_reason([r], e) is None) for r in reqs] for e in engs])

    # Проблемные места в терминах цели — для подсказок «куда бить» и для замера прицела (патч learned_full).
    over_route = torch.zeros(len(engs))
    near = 0.75 * target
    detour_values = sorted(goal_detour.values())
    detour_cut = detour_values[int(.75 * (len(detour_values) - 1))] if detour_values else float('inf')
    problem_requests = torch.zeros(len(reqs), dtype=torch.bool)
    request_goal = torch.zeros(len(reqs))
    for i, r in enumerate(reqs):
        entry, wait = owner.get(r.id), waits.get(r.id)
        if entry is None:
            request_goal[i] = 3 + r.priority                     # строки 1–3: неназначенная, по приоритету
            problem_requests[i] = True
        elif wait is not None and wait > target:
            request_goal[i] = 2 + min(1., (wait - target) / target)   # строка 4: авария сверх ориентира
            over_route[entry[0]] += 1
            problem_requests[i] = True
        elif wait is not None and wait >= near:
            request_goal[i] = 1 + (wait - near) / (target - near)     # строка 4: авария у порога
            problem_requests[i] = True
        else:
            d = goal_detour.get(r.id, 0.)
            request_goal[i] = d / (1 + d)                             # строка 6: крюк
            problem_requests[i] = d >= detour_cut and d > 0
    stops = torch.tensor([float(len(plan.routes[e.id].stops)) for e in engs])
    used_routes = stops > 0
    route_goal = (2 * over_route + 1 / (1 + stops)) * used_routes     # строка 4, затем строка 5 (почти пустой — к роспуску)
    problem_routes = used_routes & ((over_route > 0) | (stops <= 2))
    routes = torch.cat([routes, over_route[:, None]], 1)
    glob = torch.cat([glob, torch.tensor([float(over_route.sum())])])
    # Подсказки: при penalty_hints — по строкам цели (патч learned_full; прежние — по старым весам 10000/300).
    if penalty_hints:
        route_hint, request_hint = route_goal.clamp_min(1e-6), request_goal.clamp_min(1e-6)
        if penalty_scope == 'aim':
            # Штраф только в наводку: цель головы остаётся прежней формулой,
            # иначе замер смешивает два разных утверждения.
            head_route, head_request = _heuristic_route_hint(routes), _heuristic_request_hint(reqs, owner, compatible)
        else:
            head_route, head_request = route_hint, request_hint
    else:
        route_hint = _heuristic_route_hint(routes)
        request_hint = _heuristic_request_hint(reqs, owner, compatible)
        head_route, head_request = route_hint, request_hint
    # Колонки routes: 25 — запас смены (минуты), 3 — доля смены, занятая работой.
    engineer_hint = (1 + routes[:, 25].clamp_min(0) / 720) / (1 + routes[:, 3].clamp_min(0))
    if best is not None:
        gap = [float(a - b) for a, b in zip(full_cost, best.cost_tuple())]
        glob = torch.cat([glob, torch.tensor(gap, dtype=torch.float32)])

    g = dict(hints=dict(route=route_hint, request=request_hint, engineer=engineer_hint),
             head_hints=dict(route=head_route, request=head_request, engineer=engineer_hint),
             # Без подсказок: «куда бить» выбирает только сеть (Михаил 25.09; подсказки глушили обучение прицела, Н80).
             blend=0.,
             engineers=engineers.float(), routes=routes, requests=torch.tensor(req_features).float(), glob=glob,
             assignment=_normalize_rows(assignment), ownership=_normalize_rows(assignment.T),
             compatible=_normalize_rows(compatible), reverse_compatible=_normalize_rows(compatible.T),
             neighbors=_normalize_rows(sequence + _geographic_neighbours(reqs)),
             assigned=torch.tensor([r.id in owner for r in reqs]), owner=owner,
             problem_requests=problem_requests, problem_routes=problem_routes)
    if best is not None:
        g['glob_extra_names'] = RECORD_GAP
    if state.has_history:
        # Ключи только при истории — граф утренней задачи остаётся побайтово прежним
        # (эталон хеширует весь словарь). Утром узлы заявок — все inst.requests по порядку.
        g['request_ids'] = view.request_ids
        # Маршрут с историей распустить нельзя: исполнитель уже в работе.
        g['locked_routes'] = torch.tensor([e.id in view.already_used for e in engs])
    return g


def request_detours(inst, plan, normalized=True, origin=None):
    """Крюк каждой назначенной заявки: prev->r->next минус prev->next.

    Возврата в офис нет, поэтому у последней остановки крюк это входящее плечо.
    Нормируем на среднюю длину плеча в этом же плане: признак становится
    безразмерным («этот заезд во столько-то раз длиннее обычного») и не требует
    сохранённых констант, то есть переносится на другой район без перекалибровки.
    origin — точка старта маршрута по исполнителю (проекция на хвост); по умолчанию офис.
    """
    index = inst.index; depot = index['depot']
    raw = {}; legs = []
    for eng in inst.engineers:
        stops = plan.routes[eng.id].stops
        for i, stop in enumerate(stops):
            prev = (origin[eng.id] if origin else depot) if i == 0 else index[stops[i - 1].request_id]
            here = index[stop.request_id]
            leg = inst.matrix.distance(prev, here)
            legs.append(leg)
            gain = leg
            if i + 1 < len(stops):
                nxt = index[stops[i + 1].request_id]
                gain += inst.matrix.distance(here, nxt) - inst.matrix.distance(prev, nxt)
            raw[stop.request_id] = max(0., gain)
    if not normalized or not legs:
        return raw
    scale = sum(legs) / len(legs)
    if scale <= 0:
        return raw
    return {k: v / scale for k, v in raw.items()}


def penalty_attribution(inst, plan, reqs, engs, owner, view=None, since=None):
    """Разложение plan_scalar по носителям: (вес маршрута, вес заявки).

    На маршруте висит W_ENGINEER за сам факт занятости плюс его километры.
    На заявке — либо штраф за неназначенность по её приоритету, либо её собственный
    крюк prev->r->next минус prev->next, плюс просрочка, если заявка срочная.
    Возврата в офис нет, поэтому у последней остановки крюк — это входящее плечо.

    Пол в единицу оставлен намеренно и по той же причине, что в Planner: сущности
    с нулевым штрафом тоже должны иногда выбираться, иначе поиск слепнет к ним
    целиком. Никаких других коэффициентов здесь нет.

    view — проекция на подвижную часть: маршрут — хвост от своей точки старта, а
    W_ENGINEER не висит на уже задействованном историей (очистка хвоста его не сэкономит).
    since — база задержки аварий, начало смен исходной задачи (как в solver.plan_scalar);
    без него — начало смен inst (для задачи без событий это одно и то же).
    """
    from src.plan.solver import W_UNASSIGNED, W_ENGINEER, delay_cost
    first = since if since is not None else day_start(inst)

    index = inst.index
    depot = index['depot']
    radix = len(inst.by_id) + 1
    request_weight = {}
    for u in plan.unassigned:
        req = inst.by_id.get(u.request_id)
        if req is not None:
            request_weight[req.id] = W_UNASSIGNED * radix ** req.priority
    route_weight = []
    already_used = view.already_used if view is not None else frozenset()
    for eng in engs:
        route = plan.routes[eng.id]
        stops = route.stops
        start = view.origin[eng.id] if view is not None else depot
        route_weight.append((W_ENGINEER if route.used and eng.id not in already_used else 0.) + route.distance_km)
        for i, stop in enumerate(stops):
            req = inst.by_id[stop.request_id]
            prev = start if i == 0 else index[stops[i - 1].request_id]
            here = index[req.id]
            detour = inst.matrix.distance(prev, here)
            if i + 1 < len(stops):
                nxt = index[stops[i + 1].request_id]
                detour += inst.matrix.distance(here, nxt) - inst.matrix.distance(prev, nxt)
            delay = (stop.start - max(req.window_start, first)).total_seconds() / 60 if req.urgent else 0.
            request_weight[req.id] = max(0., detour) + delay_cost([max(0., delay)])
    return (torch.tensor(route_weight, dtype=torch.float32) + 1.,
            torch.tensor([request_weight.get(r.id, 0.) for r in reqs], dtype=torch.float32) + 1.)


def _potential(inst, plan, g, pair, feasible, reqs):
    """Потенциал (Михаил 25.09, проба «куда бить»: ×2.5–3.6 к доле улучшений): выигрыш переноса заявки = её крюк (км) минус
    цена лучшей вставки к другой бригаде (из уже посчитанных pair, без открытия новой). Признаки: у заявки — выигрыш и флаг
    «перенос выгоден», у маршрута — сумма положительных выигрышей, в общем состоянии — число выгодных переносов.
    potential_hint — те же выигрыши для прицела uniform (U2n): «куда бить» частью пропорционально потенциалу."""
    raw = request_detours(inst, plan, normalized=False)
    n, m = len(reqs), len(inst.engineers)
    owner = {s.request_id: ei for ei, e in enumerate(inst.engineers) for s in plan.routes[e.id].stops}
    gain = torch.zeros(n)
    for qi, r in enumerate(reqs):
        own = owner.get(r.id)
        if own is None:
            continue
        costs = [float(pair[ei, qi, 1]) * 100. for ei in range(m) if ei != own and feasible[ei, qi] and pair[ei, qi, 1] * 100. < 1e5]
        if costs:
            gain[qi] = raw.get(r.id, 0.) - min(costs)
    positive = gain > 0
    route_gain = torch.zeros(m)
    for qi, r in enumerate(reqs):
        own = owner.get(r.id)
        if own is not None and positive[qi]:
            route_gain[own] += gain[qi]
    g['requests'] = torch.cat([g['requests'], gain[:, None], positive.float()[:, None]], 1)
    g['routes'] = torch.cat([g['routes'], route_gain[:, None]], 1)
    g['glob'] = torch.cat([g['glob'], torch.tensor([float(positive.sum())])])
    g['request_extra_names'] = (*g.get('request_extra_names', ()), 'potential_gain', 'potential_positive')
    g['route_extra_names'] = (*g.get('route_extra_names', ()), 'potential_route')
    g['glob_extra_names'] = (*g.get('glob_extra_names', ()), 'potential_count')
    g['potential_hint'] = dict(request=gain.clamp_min(0), route=route_gain)
    g['problem_requests'] = positive
    g['problem_routes'] = route_gain > 0


def enrich_graph(inst, plan, g, cache, rehomable=False, balance_hint=False, potential=False):
    """Точные допустимые переносы, рёбра для обмена сообщениями и маска переноса.

    Маска переноса полная; рёбра eligible/can_serve прорежены до четырёх самых дешёвых
    адресатов на заявку. Прореживание сокращает сообщения, но не убирает ни одного действия.
    """
    # Заявки графа — проекция без закреплённых; вставка считается в полный маршрут.
    reqs = inst.requests if 'request_ids' not in g else [inst.requests[i] for i in g['request_ids']]
    n, m = len(reqs), len(inst.engineers)
    feasible = torch.zeros(m, n, dtype=torch.bool)
    pair = torch.zeros(m, n, 4)      # [есть, цена вставки/100, позиция/длина маршрута, 1]
    orders = {e.id: [inst.by_id[s.request_id] for s in plan.routes[e.id].stops] for e in inst.engineers}
    for ei, e in enumerate(inst.engineers):
        for qi, req in enumerate(reqs):
            if g['ownership'][qi, ei] > 0:
                continue
            # Недоступный исполнитель новой работы не получает.
            if not e.available:
                continue
            # Лучшая вставка без шума после закреплённого начала — из сервиса маршрута.
            option = cache.best_insertion(req, ei, orders[e.id])
            if option is not None:
                feasible[ei, qi] = True
                # Позиция — относительно подвижного хвоста: длина истории в признак не протекает (Н34).
                lock = cache.state.tails[e.id].lock
                pair[ei, qi] = torch.tensor([1., option[0] / 100., (option[1] - lock) / max(1, len(orders[e.id]) - lock),
                                             1.])
    sparse = torch.zeros_like(feasible)
    for qi in range(n):
        ids = feasible[:, qi].nonzero().flatten()
        if len(ids):
            sparse[ids[pair[ids, qi, 1].argsort()[:4]], qi] = True

    edges, attrs = {}, {}

    def add(rel, src, dst, features=None):
        edges[rel] = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
        attrs[rel] = (torch.tensor(features, dtype=torch.float32).reshape(-1, 4) if features is not None
                      else torch.ones(len(src), 4))

    add(('e', 'owns', 'r'), list(range(m)), list(range(m)))
    add(('r', 'owner', 'e'), list(range(m)), list(range(m)))
    ei, qi = g['assignment'].nonzero(as_tuple=True)
    add(('q', 'assigned', 'r'), qi.tolist(), ei.tolist())
    add(('r', 'contains', 'q'), ei.tolist(), qi.tolist())
    ei, qi = sparse.nonzero(as_tuple=True)
    f = pair[ei, qi].tolist()
    add(('q', 'eligible', 'e'), qi.tolist(), ei.tolist(), f)
    add(('e', 'can_serve', 'q'), ei.tolist(), qi.tolist(), f)

    idx = {r.id: i for i, r in enumerate(reqs)}
    src, dst, features = [], [], []
    for route in plan.routes.values():
        for a, b in zip(route.stops, route.stops[1:]):
            if a.request_id not in idx or b.request_id not in idx:
                continue
            src.append(idx[a.request_id])
            dst.append(idx[b.request_id])
            features.append([b.travel_km / 100., (b.arrive - a.end).total_seconds() / 3600., 0., 1.])
    add(('q', 'next', 'q'), src, dst, features)
    add(('q', 'previous', 'q'), dst, src, features)

    src, dst, features = [], [], []
    for i, req in enumerate(reqs):
        distance = lambda j: inst.matrix.distance(inst.index[req.id], inst.index[reqs[j].id])
        for j in sorted((j for j in range(n) if j != i), key=distance)[:4]:
            src.append(i)
            dst.append(j)
            features.append([distance(j) / 100., 0., 0., 1.])
    add(('q', 'near', 'q'), src, dst, features)

    g.update(edge_index=edges, edge_attr=attrs, transfer_mask=feasible,
             transfer_hint=transfer_weights(inst, plan, pair, feasible, balance_hint, reqs),
             graph_stats=dict(edges={str(k): v.shape[1] for k, v in edges.items()},
                              feasible_pairs=int(feasible.sum()), message_pairs=int(sparse.sum())))
    if potential:
        _potential(inst, plan, g, pair, feasible, reqs)
    if rehomable:
        g['routes'] = torch.cat([g['routes'], rehomable_share(inst, plan, feasible, reqs=reqs)], 1)
        # Имя объявляем явно: check_names сверяет по этому списку, а не по ширине.
        g['route_extra_names'] = (*g.get('route_extra_names', ()), 'rehomable_share')
    return g


def transfer_weights(inst, plan, pair, feasible, balance_hint, reqs=None):
    """Наводка переноса: по цене вставки (как было) или по выгоде хода.

    Выгода = крюк донора минус цена вставки у получателя. Стенд показал, что
    выбор цели по крюку поднимает долю удачных переносов с 13.3% до 32.1%,
    а по полной выгоде — до 100%, но последнее делает оператор детерминированным
    жадным шагом, поэтому берём положительную часть и оставляем её распределением.
    Единиц измерения не добавляем: и крюк, и цена вставки в километрах.
    """
    if not balance_hint:
        return torch.exp(-pair[:, :, 1].clamp_min(0)) * feasible
    detour = request_detours(inst, plan, normalized=False)
    gain = torch.tensor([detour.get(r.id, 0.) for r in (reqs if reqs is not None else inst.requests)],
                        dtype=torch.float32)
    # pair[..., 1] — цена вставки, делённая на 100 при сборке рёбер
    balance = gain[None, :] - pair[:, :, 1] * 100.
    return balance.clamp_min(0.) * feasible


def rehomable_share(inst, plan, feasible, reqs=None):
    """Доля заявок маршрута, у которых есть хотя бы один допустимый чужой хозяин.

    route_head выбирает, какой маршрут распускать, но до сих пор не видел главного:
    разойдутся заявки или повиснут. Стенд diagnose_feature_value дал AUC 0.873 и 0.921
    на отказ «потерял заявки» при избыточности 0.28-0.65 с имеющимися признаками.
    Предсказывает он именно провал, а не выигрыш (AUC 0.62-0.65), и это честная,
    более узкая польза: потеря заявки бьёт по третьей компоненте cost_tuple,
    которая старше состава.

    Матрица feasible уже посчитана выше для transfer, поэтому признак достаётся даром.
    """
    # reqs — заявки графа (проекция): закреплённые и завершённые — без узла, в долю не входят.
    index = {r.id: i for i, r in enumerate(reqs if reqs is not None else inst.requests)}
    values = torch.zeros(len(inst.engineers), 1)
    for ei, eng in enumerate(inst.engineers):
        stops = plan.routes[eng.id].stops
        if not stops:
            continue
        rows = torch.ones(len(inst.engineers), dtype=torch.bool)
        rows[ei] = False
        active = [s for s in stops if s.request_id in index]
        if not active:
            continue
        values[ei, 0] = float(sum(bool(feasible[rows, index[s.request_id]].any()) for s in active)) / len(active)
    return values
