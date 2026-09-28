"""Операторы разрушения и мутация плана: разрушить по действию, восстановить regret-вставкой.

Реестр DESTROY сопоставляет имени оператора функцию, которая возвращает снимаемые
заявки. Порядок обращений к генератору внутри мутации — часть контракта: от него
зависит траектория поиска, и эталон шага 0 сверяет её побайтово.
"""
# refactor: Claude, 21.09.2026 — из mutate() в src/alt/population_gnn.py
# refactor: Claude, 21.09.2026 — шаг 5г: история из RouteState, перенос через сервис маршрута
import random
from dataclasses import dataclass

from src.engine.repair import NativeRepair as NativeRepairType, exact_best_position
from src.engine.route_state import RouteState
from src.search.actions import CHAIN_EXTRA, OPERATORS, SCALES, Action
from src.plan.solver import construct

NOISE = .15                # шум вставки при восстановлении (по умолчанию; задаёт конфигурация поиска)


@dataclass
class Destroy:
    """Всё, что нужно оператору: задача, заказы по исполнителям, действие, генератор, снимок истории."""
    inst: object
    orders: dict
    action: Action
    rng: random.Random
    state: RouteState

    @property
    def frozen(self):
        """Закреплённые заявки: начатые и завершённые до события. Их не снимают."""
        return self.state.pinned

    def route_of(self, index):
        """Подвижная часть маршрута: без закреплённого начала (утром — весь маршрут)."""
        eid = self.inst.engineers[index].id
        return self.state.movable(eid, self.orders[eid])

    def share(self, seq):
        """Сколько снять из маршрута при выбранном масштабе: хотя бы одну, не больше всех."""
        return min(len(seq), max(1, round(len(seq) * SCALES[self.action.scale])))

    def assigned(self, excluding=()):
        return [r for seq in self.orders.values() for r in seq if r.id not in excluding and r.id not in self.frozen]


def neighbourhood(inst, assigned, anchor, operator, count, rng):
    """count назначенных заявок, ближайших к якорю: по расстоянию (geographic) или по окну.

    Шум 0.1 разбивает ничьи случайно; одно обращение к генератору на каждую заявку
    в порядке списка — так их делает sorted с ключом.
    """
    if OPERATORS[operator] == 'geographic':
        score = lambda r: inst.matrix.distance(inst.index[anchor.id], inst.index[r.id])
    else:
        score = lambda r: abs((r.window_start - anchor.window_start).total_seconds()) / 60
    return sorted(assigned, key=lambda r: score(r) + rng.random() * .1)[:count]


def route_segment(d):
    """Подряд идущий кусок маршрута, начиная с заявки-якоря, с переходом через конец."""
    seq = d.route_of(d.action.route)
    anchor = d.inst.requests[d.action.request]
    pos = next(i for i, r in enumerate(seq) if r.id == anchor.id)
    return [seq[(pos + j) % len(seq)] for j in range(d.share(seq))]


def route_pair(d):
    """Случайные заявки из двух маршрутов: своего и маршрута-адресата."""
    victims = []
    for index in (d.action.route, d.action.engineer):
        seq = d.route_of(index)
        victims.extend(d.rng.sample(seq, d.share(seq)))
    return victims


def cross(d):
    """geographic и time_window: соседство якоря по всем маршрутам сразу."""
    # Без масштаба — самый мелкий, как action.get('scale', 0) в прежнем коде.
    count = max(2, round(len(d.inst.requests) * SCALES[d.action.scale if d.action.scale is not None else 0]))
    return neighbourhood(d.inst, d.assigned(), d.inst.requests[d.action.request], d.action.operator, count, d.rng)


def transfer(d):
    """Одна заявка, которую затем вставят к выбранному исполнителю (см. mutate)."""
    return [d.inst.requests[d.action.request]]


def dissolve_route(d):
    """Маршрут целиком. Восстановлению запрещено открыть исполнителя заново (forbid).

    Состав исполнителей — четвёртая компонента cost_tuple, важнее пробега, но ни один
    другой оператор её прицельно не трогает: разрушение должно выесть у исполнителя
    все заявки разом, а восстановление не должно открыть его обратно.
    """
    return list(d.route_of(d.action.route))


DESTROY = {'route_segment': route_segment, 'route_pair': route_pair, 'geographic': cross,
           'time_window': cross, 'transfer': transfer, 'dissolve_route': dissolve_route}
assert tuple(DESTROY) == OPERATORS


def chain_links(d, victims):
    """Поперечные разрушения поверх выбранного хода, до длины цепочки.

    Отбор идёт по лучшему в популяции, а рекорд хранится отдельно, поэтому рискованный
    ход ничего не стоит: плохой кандидат просто не будет выбран. Звенья, не выбранные
    сетью, тянутся случайно. Цепочка обрывается, когда нетронутых назначенных меньше двух.
    """
    inst, rng, links = d.inst, d.rng, d.action.links
    added = 0
    for _ in range(max(1, d.action.chain or 1) - 1):
        assigned = d.assigned(excluding={v.id for v in victims})
        if len(assigned) < 2:
            break
        if added < len(links):
            link = links[added]
            operator, anchor = link.operator, inst.requests[link.anchor]
            scale = link.scale
        else:
            operator = rng.choice(CHAIN_EXTRA)
            anchor = rng.choice(assigned)
            scale = rng.randrange(len(SCALES))
        count = max(2, round(len(inst.requests) * SCALES[scale]))
        # Звено не geographic — значит по окну, как и в прежнем коде.
        kind = operator if OPERATORS[operator] == 'geographic' else OPERATORS.index('time_window')
        victims = victims + neighbourhood(inst, assigned, anchor, kind, count, rng)
        added += 1
    return victims, added


def best_position(state, eng, seq, req):
    """Самая короткая по километрам допустимая вставка заявки в маршрут, или None.

    Питоновский эталон — точный перебор (engine.repair.exact_best_position): не раньше
    закреплённого начала и с учётом момента готовности. Нативный путь —
    NativeRepair.best_position с тем же результатом.
    """
    return exact_best_position(state, eng, seq, req)


def mutate(inst, plan, action, seed, constructor=None, state=None, noise=NOISE):
    """Кандидат-потомок: разрушение по действию, затем regret-восстановление.

    action — Action или словарь журнала. constructor — восстановление (RouteCache,
    NativeRepair); по умолчанию solver.construct. state — снимок истории маршрутов
    (по умолчанию — у восстановления или новый). noise — шум вставки. Возвращает
    (план, трасса мутации).
    """
    action = Action.of(action)
    rng = random.Random(seed)
    state = state or getattr(constructor, 'state', None) or RouteState(inst)
    orders = {eid: [inst.by_id[s.request_id] for s in r.stops] for eid, r in plan.routes.items()}
    requests = inst.requests
    frozen = state.pinned
    destroy = Destroy(inst, orders, action, rng, state)
    victims = DESTROY[OPERATORS[action.operator]](destroy)
    victims, links = chain_links(destroy, victims)

    removed = {r.id for r in victims if r.id not in frozen}
    orders = {e: [r for r in seq if r.id not in removed] for e, seq in orders.items()}
    transferred = False
    moving = OPERATORS[action.operator] == 'transfer' and action.engineer is not None
    if moving:
        eng = inst.engineers[action.engineer]
        # Нативный сервис маршрута, если он есть у восстановления; иначе — питоновский эталон.
        position = getattr(constructor, 'best_position', None)
        best = (position(eng, orders[eng.id], requests[action.request]) if position is not None
                else best_position(state, eng, orders[eng.id], requests[action.request]))
        if best is not None:
            orders[eng.id] = best[1]
            transferred = True
    held = {r.id for seq in orders.values() for r in seq}
    dissolved = inst.engineers[action.route].id if OPERATORS[action.operator] == 'dissolve_route' else None
    # forbid передаём только когда он есть: сторонний конструктор без этого параметра
    # должен продолжать работать на остальных пяти операторах.
    extra = {'forbid': {dissolved}} if dissolved else {}
    candidate = (constructor or construct)(inst, orders=orders, pool=[r for r in requests if r.id not in held],
                                           regret=action.repair % 3 + 1, noise=noise, rng=rng,
                                           **extra, **({'mode': action.repair // 3} if isinstance(constructor, NativeRepairType) else {}))
    target = inst.engineers[action.engineer].id if action.engineer is not None else None
    return candidate, dict(
        removed=sorted(removed), transfer_applied=transferred, dissolved=dissolved, chain_links=links,
        dissolve_kept=(not candidate.routes[dissolved].stops) if dissolved else None,
        transfer_retained=(any(s.request_id == requests[action.request].id for s in candidate.routes[target].stops)
                           if moving else None),
        target_engineer=target)
