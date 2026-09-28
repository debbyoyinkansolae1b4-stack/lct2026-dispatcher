"""Восстановление плана после разрушения: regret-вставка с кэшем в пределах одной задачи.

Две реализации с одним интерфейсом `repair(inst, pool, orders, regret, locked, noise,
rng, forbid) -> Plan`:

- RouteCache — питоновская: solver.construct с запоминающими расчётом маршрута
  и вставкой, переданными явными аргументами. Она же — эталон для нативной;
- NativeRepair — те же решения через скомпилированные ядра (src/engine/kernels.py).

Обе дают одинаковые планы и одинаково расходуют генератор случайных чисел
(tests/test_native_repair.py, tests/test_population_edges.py). Кэш принадлежит одной
задаче и одной популяции: данные задачи во время поиска не меняются.

История маршрутов (закреплённое начало, момент готовности, расход оборудования) —
из одного снимка RouteState (src/engine/route_state.py). С историей NativeRepair повторяет
питоновский solver.construct (точную вставку) вплоть до сумм пробега (Н28):
tests/test_native_events.py. Шум вставки задаёт вызывающий: ядро его не переопределяет.
"""
# refactor: Claude, 21.09.2026 — из src/alt/cached_population.py и native_repair.py; bind() заменён явными аргументами
# refactor: Claude, 21.09.2026 — шаг 5б: закреплённое начало маршрутов после события
# refactor: Claude, 21.09.2026 — шаг 5г: история из RouteState, кэш маршрутов и с историей
# patch urgent_insertion: Claude, 25.09.2026 — вставка с узкими окнами аварий, второй проход с обычными (Н77)
import random
from collections import OrderedDict
from functools import partial

import numpy as np

from src.plan import solver
from src.engine.compiled import CompiledInstance
from src.engine.kernels import history_deltas, random_jitter, route_column, route_total, route_view, select_regret
from src.engine.route_state import RouteState
from src.plan.fastroute import build_view, insertion_delta
from src.model import Unassigned
from src.plan.scheduling import evaluate, static_reason

# Предел памяти кэша колонок нативного восстановления.
COLUMN_BYTES = 64 * 1024 * 1024


def exact_best_position(state, eng, seq, req, evaluate_route=evaluate):
    """Самая короткая по пробегу допустимая вставка заявки в маршрут: (пробег, новый порядок) или None.

    Точный перебор: каждая позиция после закреплённого начала проверяется расчётом
    маршрута уже со вставкой. Маршрут до вставки может быть недопустим (после разрушения
    на нетреугольной матрице) — вставка способна это исправить (Н31).
    """
    inst = state.inst
    best = None
    for pos in range(state.tails[eng.id].lock, len(seq) + 1):
        trial = seq[:pos] + [req] + seq[pos:]
        route, bad = evaluate_route(eng, trial, inst.matrix, inst.index, **state.evaluate_kwargs(eng))
        if bad is None and (best is None or route.distance_km < best[0]):
            best = (route.distance_km, trial)
    return best


class RouteCache:
    """Запоминание расчёта маршрута и вариантов вставки в пределах одной задачи."""

    def __init__(self, inst, limit=50000, compiled=None, defer_reasons=False, state=None, product_insertion=True,
                 open_cost=None):
        self.inst = inst
        # Цена открытия новой бригады в продуктовой вставке — настройка участника (SearchConfig.open_cost);
        # None — solver.OPEN_ENGINEER_PENALTY (так строится стартовый план).
        self.open_cost = open_cost
        # Вставка: продуктовая (+OPEN_ENGINEER_PENALTY за новую бригаду, способ — только regret) или вставка сети b
        # («плотная» — новая бригада, только если некуда; «свободная» — без штрафа). Выбирает участник (SearchConfig.insertion).
        self.product_insertion = product_insertion
        self.state = state or RouteState(inst)
        self.compiled = compiled
        self.defer_reasons = defer_reasons
        self.limit = limit
        self.routes = OrderedDict()
        self.evaluations = OrderedDict()
        self._counts = [0, 0, 0, 0]
        self.cached_reason = partial(solver._reason_for, evaluate_route=self.evaluate)
        self.rebuild = partial(solver._rebuild, evaluate_route=self.evaluate)
        self.constructor = partial(solver.construct, evaluate_route=self.evaluate, insertion=self.insertion,
                                   reason=self.deferred_reason if defer_reasons else self.cached_reason,
                                   rebuild=self.rebuild)

    def deferred_reason(self, req, inst, orders):
        """Объяснение отказа откладывается до итога; допустимость и состав неназначенных те же."""
        return Unassigned(req.id, 'deferred until result materialization', 'deferred')

    def materialize(self, plan):
        """Заменяет отложенные объяснения настоящими — для итогового плана."""
        orders = {eid: [self.inst.by_id[s.request_id] for s in route.stops] for eid, route in plan.routes.items()}
        plan.unassigned = [self.cached_reason(self.inst.by_id[u.request_id], self.inst, orders)
                           if u.reason_code == 'deferred' else u for u in plan.unassigned]
        return plan

    @property
    def stats(self):
        return dict(zip(('evaluate_calls', 'evaluate_hits', 'insertion_calls', 'insertion_hits'), self._counts))

    def best_insertion(self, req, ei, order):
        """Лучшая вставка без шума после закреплённого начала: (прирост, позиция) или None.

        Питоновский эталон: с историей — точная вставка, утром — быстрая по развёртке.
        NativeRepair даёт то же на ядрах.
        """
        eng = self.inst.engineers[ei]
        if self.state.has_history:
            return solver._insertion_candidates(req, eng, order, self.inst, min_pos=self.state.tails[eng.id].lock,
                                                opening=self.opening())
        return self.insertion(req, eng, order, self.inst, {})

    def opening(self):
        """Цена открытия нового исполнителя на питоновском пути (читается в момент вызова: снимок весов)."""
        if not self.product_insertion:
            return solver.OPEN_LAST
        return solver.OPEN_ENGINEER_PENALTY if self.open_cost is None else self.open_cost

    def _check_static(self, inst, locked=None):
        if inst is not self.inst:
            raise ValueError('Cache belongs to a different instance')
        # Питоновский эталон — только для утренней задачи; снимок сверяется на каждом вызове.
        if self.state.has_history or locked or RouteState._signature(inst) != self.state.signature:
            raise ValueError('Route cache is for static immutable scenarios only')

    def __call__(self, inst, **kwargs):
        self._check_static(inst, kwargs.get('locked'))
        return self.constructor(inst, opening=self.opening(), **kwargs)

    def evaluate(self, eng, order, matrix, index, **kwargs):
        if matrix is not self.inst.matrix or index is not self.inst.index:
            raise ValueError('Changed matrix/index')
        # Ключ — исполнитель и порядок: история и момент события постоянны в пределах снимка.
        # Вызов с другими параметрами истории — мимо кэша.
        if not self.state.owns(eng, kwargs):
            return evaluate(eng, order, matrix, index, **kwargs)
        key = (eng.id, tuple(r.id for r in order))
        self._counts[0] += 1
        if key in self.evaluations:
            self._counts[1] += 1
            self.evaluations.move_to_end(key)
            return self.evaluations[key]
        result = evaluate(eng, order, matrix, index, **kwargs)
        self.evaluations[key] = result
        if len(self.evaluations) > self.limit:
            self.evaluations.popitem(last=False)
        return result

    def insertion(self, req, eng, order, inst, view_cache, noise=None, equip_cache=None):
        """Лучшая вставка заявки в маршрут: (прирост с шумом, позиция) или None."""
        self._counts[2] += 1
        # Конструктор сбрасывает эту запись, как только маршрут исполнителя меняется.
        entry = view_cache.get(eng.id)
        if entry is None:
            key = (eng.id, tuple(r.id for r in order))
            entry = self.routes.get(key)
            if entry is None:
                entry = {'view': None, 'built': False, 'options': {}, 'order': tuple(order),
                         'equipment': solver._route_equipment(eng, order, {})}
                self.routes[key] = entry
                if len(self.routes) > self.limit:
                    self.routes.popitem(last=False)
            else:
                self.routes.move_to_end(key)
            view_cache[eng.id] = entry
        options = entry['options'].get(req.id)
        if options is None:
            if static_reason(req, eng) or not solver._equipment_fits(req, eng, entry['equipment']):
                options = ()
            else:
                if not entry['built']:
                    entry['view'] = build_view(eng, order, inst, eng.transport,
                                               earliest=inst.earliest_start, exempt=inst.earliest_exempt)
                    entry['built'] = True
                view = entry['view']
                values = []
                if view is not None:
                    opening = self.opening() if not order else 0
                    if self.compiled is not None:
                        values = [(delta + opening, pos) for delta, pos in self.compiled.options(entry, view, req, eng)]
                    else:
                        for pos in range(len(order) + 1):
                            delta = insertion_delta(view, pos, req, inst, eng, eng.transport)
                            if delta is not None:
                                values.append((delta + opening, pos))
                options = tuple(values)
            entry['options'][req.id] = options
        else:
            self._counts[3] += 1
        best = None
        for delta, pos in options:
            if noise is not None:
                delta *= noise()
            if best is None or delta < best[0]:
                best = (delta, pos)
        return best


class NativeRepair(RouteCache):
    """Regret-вставка на скомпилированных ядрах; решения и расход генератора — как у RouteCache."""

    def __init__(self, inst, **kwargs):
        super().__init__(inst, **kwargs)
        if self.compiled is None:
            self.compiled = CompiledInstance(inst)
        c = self.compiled
        # Колонка — варианты вставки всех заявок в один маршрут; ключ (исполнитель, порядок).
        self.columns = OrderedDict()
        self.column_bytes = 0
        n, m = len(inst.requests), len(inst.engineers)
        self.value_buffer = np.empty((n, m, n + 1))
        self.position_buffer = np.empty((n, m, n + 1), dtype=np.int64)
        self.count_buffer = np.empty((n, m), dtype=np.int64)
        # Хвосты из снимка — в массивы для ядер, по индексу исполнителя.
        tails = [self.state.tails[e.id] for e in inst.engineers]
        item = {k: i for i, k in enumerate(c.items)}
        self.history_used = np.zeros((m, len(c.items)))
        for ei, tail in enumerate(tails):
            for k, quantity in tail.used:
                if k in item:
                    self.history_used[ei, item[k]] += quantity
        self.history_km = [np.asarray(tail.km, dtype=np.float64) for tail in tails]
        self.tails = tails
        self.eindex = {e.id: ei for ei, e in enumerate(inst.engineers)}
        self.infeasible = set()     # ключи колонок с недопустимым хвостом
        # Патч urgent_insertion: при включённой строке «аварии сверх ориентира» вставка сначала ищет места с узкими
        # окнами аварий (не выталкивать аварию за ориентир); заявки, которым так места нет, — вторым проходом
        # с обычными окнами (иначе остались бы неназначенными — строка «назначено» выше).
        self.plain_pass = False
        # Способ вставки (патч learned_full): 0 — «плотная» (новая бригада — только если некуда), 1 — «свободная»
        # (новая бригада без сдвига, если так короче по км). Выбирает политика: repair // 3.
        self.mode = 0
        # С историей прирост считается суммой пробега полного маршрута, как в питоновском
        # пути (Н28); утром — формулой прироста, как в эталоне. Совместимость до шага 9.
        self.exact_sums = self.state.has_history

    def _displace_infeasible(self, inst, orders):
        """Снятие визита не всегда безопасно (нетреугольные матрицы): недопустимый хвост в очередь."""
        displaced = []
        for ei, eng in enumerate(inst.engineers):
            tail = self.tails[ei]
            order = orders.setdefault(eng.id, [inst.by_id[rid] for rid in tail.fixed_ids])
            while order:
                route, bad = self.evaluate(eng, order, inst.matrix, inst.index, **self.state.evaluate_kwargs(eng))
                if route is not None:
                    break
                if len(order) <= tail.lock:
                    raise RuntimeError(f"Недопустимая фиксированная история у {eng.id}: {bad}")
                displaced.append(order.pop())
        return displaced

    def _tight(self):
        return solver.URGENT_TARGET_FIRST and not self.plain_pass

    def _key(self, ei, order):
        return (ei, tuple(r.id for r in order), self._tight(), self.mode)

    def _column(self, ei, eng, order):
        """Колонка маршрута из кэша или заново; кэш вытесняет старые колонки сверх COLUMN_BYTES."""
        c = self.compiled
        key = self._key(ei, order)
        column = self.columns.get(key)
        if column is not None:
            self.columns.move_to_end(key)
            return column
        # Ядра видят только подвижный хвост: старт в точке origin в момент ready.
        state = self.tails[ei]
        lock = state.lock
        if lock and tuple(r.id for r in order[:lock]) != state.fixed_ids:
            raise ValueError(f'Изменён закреплённый маршрут у {eng.id}')
        tail = order[lock:]
        order_indices = np.asarray([c.qindex[r.id] for r in tail], dtype=np.int64)
        windows = c.tight_windows(solver.URGENT_TARGET_MIN)[ei] if key[2] else c.windows[ei]
        valid, indices, ends, latest = route_view(order_indices, c.nodes, windows, c.durations,
                                                  state.ready, c.shifts[ei, 1], state.origin,
                                                  c.tables[eng.transport], c.step)
        if not valid and key[2]:
            # Авария в маршруте уже за ориентиром — узкие окна маршрут не держат: обычные окна для этого маршрута.
            windows = c.windows[ei]
            valid, indices, ends, latest = route_view(order_indices, c.nodes, windows, c.durations,
                                                      state.ready, c.shifts[ei, 1], state.origin,
                                                      c.tables[eng.transport], c.step)
        width = len(tail) + 1
        requests = len(self.inst.requests)
        if not valid:
            # Недопустимый хвост: колонка пуста, но это не «вставлять некуда» (см. best_position).
            self.infeasible.add(key)
            column = (np.zeros((requests, width)), np.zeros((requests, width), dtype=np.int64),
                      np.zeros(requests, dtype=np.int64))
        else:
            used = c.need[order_indices].sum(0) if tail else np.zeros(len(c.items))
            if lock:
                used = used + self.history_used[ei]
            # Штраф открытия — по всему маршруту: исполнитель с историей уже задействован.
            # Продуктовая вставка: +OPEN_ENGINEER_PENALTY за новую бригаду, способ вставки — только regret (Н83).
            if self.product_insertion:
                opening = 0. if order or state.already_used else self.opening()
            else:
                opening = 0. if order or state.already_used or self.mode == 1 else solver.OPEN_LAST
            column = route_column(indices, ends, latest, c.shifts[ei, 1], state.floor, c.nodes, windows,
                                  c.durations, c.eligible[ei], c.need, c.stocks[ei], used, state.origin,
                                  c.km, c.tables[eng.transport], c.step, opening)
            if self.exact_sums:
                # Позиции в полном маршруте: при равенстве прироста сравнение идёт по позиции.
                column = (column[0], column[1] + lock, column[2])
                history_deltas(column[0], column[1], column[2], self.history_km[ei], indices, state.origin,
                               c.nodes, c.km, opening, lock)
        self.columns[key] = column
        self.column_bytes += sum(a.nbytes for a in column)
        while self.column_bytes > COLUMN_BYTES and len(self.columns) > 1:
            old_key, old = self.columns.popitem(last=False)
            self.column_bytes -= sum(a.nbytes for a in old)
            self.infeasible.discard(old_key)     # пометка живёт столько же, сколько колонка (Н32)
        return column

    def _check_call(self, inst, locked=None):
        # Снимок истории сверяется в месте подключения (RouteState.check), не на каждом кандидате.
        if inst is not self.inst:
            raise ValueError('Cache belongs to a different instance')
        if locked:
            raise ValueError('locked is not supported: fixed history comes from the route state')

    def best_position(self, eng, seq, req):
        """Самая короткая по пробегу полного маршрута допустимая вставка: (пробег, новый порядок) или None.

        То же, что exact_best_position: пробег — сумма полного маршрута, при равенстве —
        первая позиция. Быстрый путь — колонка допустимого хвоста; если хвост до вставки
        недопустим, колонка ничего не говорит, и позиции проверяются точным перебором (Н31).
        """
        ei = self.eindex[eng.id]
        values, positions, counts = self._column(ei, eng, seq)
        if self._key(ei, seq) in self.infeasible:
            return exact_best_position(self.state, eng, seq, req, evaluate_route=self.evaluate)
        c, tail = self.compiled, self.tails[ei]
        qi = c.qindex[req.id]
        if not counts[qi]:
            return None
        lock = tail.lock
        indices = np.asarray([c.nodes[c.qindex[r.id]] for r in seq[lock:]], dtype=np.int64)
        best = None
        for pos in sorted(int(p) for p in positions[qi, :counts[qi]]):
            total = route_total(self.history_km[ei], indices, tail.origin, c.nodes[qi], pos - lock, c.km)
            if best is None or total < best[0]:
                best = (total, pos)
        if best is None:
            return None
        pos = best[1]
        return best[0], seq[:pos] + [req] + seq[pos:]

    def best_insertion(self, req, ei, order):
        """Лучшая вставка без шума (прирост со штрафом открытия, позиция) или None — как _insertion_candidates."""
        values, positions, counts = self._column(ei, self.inst.engineers[ei], order)
        qi = self.compiled.qindex[req.id]
        if not counts[qi]:
            return None
        j = int(np.argmin(values[qi, :counts[qi]]))
        return float(values[qi, j]), int(positions[qi, j])

    def __call__(self, inst, pool=None, orders=None, regret=2, locked=None, noise=0., rng=None, forbid=None, mode=0):
        self._check_call(inst, locked)
        if mode not in (0, 1):
            raise ValueError('mode вставки — 0 (плотная) или 1 (свободная)')
        # У продуктовой вставки способа нет: ключ кэша колонок не должен делиться по нему (лишние промахи).
        self.mode = 0 if self.product_insertion else mode
        try:
            return self._call(inst, pool, orders, regret, noise, rng, forbid)
        finally:
            self.mode = 0

    def _call(self, inst, pool, orders, regret, noise, rng, forbid):
        if regret < 1:
            raise ValueError('regret must be positive')
        c = self.compiled
        orders = {eid: list(v) for eid, v in (orders or {}).items()}
        displaced = self._displace_infeasible(inst, orders)
        assigned = {r.id for seq in orders.values() for r in seq}
        pending = list(pool if pool is not None else [r for r in inst.requests if r.id not in assigned])
        pending = list({r.id: r for r in pending + displaced if r.id not in assigned}.values())
        pending.sort(key=lambda r: (-r.priority, r.window_start, -r.duration_min))

        n, m = len(pending), len(inst.engineers)
        values, positions = self.value_buffer[:n], self.position_buffer[:n]
        counts = self.count_buffer[:n]
        counts.fill(0)
        active = np.ones(n, dtype=np.bool_)
        priorities = np.asarray([r.priority for r in pending], dtype=np.int64)
        urgent = np.asarray([r.urgent for r in pending], dtype=np.bool_)
        pending_indices = np.asarray([c.qindex[r.id] for r in pending], dtype=np.int64)

        # Генератор Python продолжается в ядре: состояние вынимается и возвращается обратно.
        rng = rng or random.Random(0)
        rng_version, rng_tuple, rng_gauss = rng.getstate()
        mt = np.asarray(rng_tuple[:-1], dtype=np.uint64)
        mt_index = rng_tuple[-1]

        def refresh(ei):
            eng = inst.engineers[ei]
            column = self._column(ei, eng, orders[eng.id])
            width = column[0].shape[1]
            values[:, ei, :width] = column[0][pending_indices]
            positions[:, ei, :width] = column[1][pending_indices]
            counts[:, ei] = column[2][pending_indices]

        for ei in range(m):
            refresh(ei)
        # Запрет открывать исполнителя держим здесь, а не в eligible: колонки кешируются
        # по (исполнитель, маршрут) без учёта запрета, и правка eligible отравила бы кеш.
        # Обнулённый счётчик позиций означает «вставлять некуда».
        if forbid:
            for ei, eng in enumerate(inst.engineers):
                if eng.id in forbid:
                    counts[:, ei] = 0
        while active.any():
            total = int(counts[active].sum())
            if noise > 0:
                jitter, mt_index = random_jitter(mt, mt_index, total, noise)
            else:
                jitter = np.empty(0)
            qi, ei, pos, consumed = select_regret(values, positions, counts, active, priorities, urgent,
                                                  c.eid_rank, regret, jitter)
            assert consumed == total
            if qi < 0:
                break
            orders[inst.engineers[ei].id].insert(pos, pending[qi])
            active[qi] = False
            refresh(ei)
        if active.any() and self._tight():
            # Второй проход: оставшимся заявкам — обычные окна (авария может выйти за ориентир, но заявка назначена).
            self.plain_pass = True
            try:
                for ei in range(m):
                    refresh(ei)
                if forbid:
                    for ei, eng in enumerate(inst.engineers):
                        if eng.id in forbid:
                            counts[:, ei] = 0
                while active.any():
                    total = int(counts[active].sum())
                    if noise > 0:
                        jitter, mt_index = random_jitter(mt, mt_index, total, noise)
                    else:
                        jitter = np.empty(0)
                    qi, ei, pos, consumed = select_regret(values, positions, counts, active, priorities, urgent,
                                                          c.eid_rank, regret, jitter)
                    if qi < 0:
                        break
                    orders[inst.engineers[ei].id].insert(pos, pending[qi])
                    active[qi] = False
                    refresh(ei)
            finally:
                self.plain_pass = False
        if noise > 0:
            rng.setstate((rng_version, tuple(int(x) for x in mt) + (int(mt_index),), rng_gauss))
        reason = self.deferred_reason if self.defer_reasons else self.cached_reason
        unassigned = [reason(r, inst, orders) for i, r in enumerate(pending) if active[i]]
        return self.rebuild(inst, orders, unassigned)
