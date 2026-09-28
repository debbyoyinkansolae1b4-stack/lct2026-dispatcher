"""Задача, развёрнутая в массивы для скомпилированных ядер (src/engine/kernels.py).

Строится один раз на задачу и дальше не меняется: матрицы расстояний и времени
по корзинам суток, окна, длительности, допуски исполнителей к заявкам, потребность
и запас оборудования, смены. Соответствие ID ↔ индекс хранится здесь же.

Пока только статическая задача: закреплённые визиты и время события (перепланирование)
появятся на шаге 5 вместе с состоянием маршрутов (REFACTOR-PLAN.md).
"""
# refactor: Claude, 21.09.2026 — массивы из src/alt/native_insertion.py и native_repair.py
# refactor: Claude, 22.09.2026 — Н30: форма окон при пустом списке заявок
# patch urgent_insertion: Claude, 25.09.2026 — узкие окна аварий для вставки (Н77)
import math

import numpy as np

from src.engine.kernels import insertion_options
from src.plan.fastroute import day_zero, to_min, window_min
from src.plan.scheduling import static_reason


class CompiledInstance:
    """Неизменяемый снимок задачи в массивах. Поддерживаются часовые профили пробок."""

    def __init__(self, inst):
        self.inst = inst
        self.km = np.asarray(inst.matrix.km, dtype=np.float64)
        traffic = inst.matrix.traffic
        if traffic.layers and (not isinstance(traffic.layer_step, int) or traffic.layer_step <= 0
                               or 1440 % traffic.layer_step):
            raise ValueError('Native time tables require a layer step dividing one day')
        self.step = math.gcd(60, traffic.layer_step) if traffic.layers else 60
        points = len(inst.matrix.points)
        self.tables = {mode: np.asarray([[[inst.matrix.minutes(i, j, mode, bucket) for j in range(points)]
                                          for i in range(points)]
                                         for bucket in range(0, 1440, self.step)], dtype=np.float64)
                       for mode in {e.transport for e in inst.engineers}}
        self.depot = inst.index['depot']

        requests, engineers = inst.requests, inst.engineers
        self.qindex = {r.id: i for i, r in enumerate(requests)}
        self.items = sorted({k for r in requests for k in r.equipment} | {k for e in engineers for k in e.equipment})
        self.need = np.asarray([[r.equipment.get(k, 0) for k in self.items] for r in requests],
                               dtype=np.float64).reshape(len(requests), len(self.items))
        self.nodes = np.asarray([inst.index[r.id] for r in requests], dtype=np.int64)
        self.durations = np.asarray([r.duration_min for r in requests], dtype=np.float64)
        self.eligible = np.asarray([[e.available and static_reason(r, e) is None for r in requests] for e in engineers])
        self.stocks = np.asarray([[e.equipment.get(k, 0) for k in self.items] for e in engineers],
                                 dtype=np.float64).reshape(len(engineers), len(self.items))
        # Явная форма: без активных заявок (всё выполнено) массив иначе схлопывается в двумерный.
        self.windows = np.asarray([[window_min(r, day_zero(e)) for r in requests] for e in engineers],
                                  dtype=np.float64).reshape(len(engineers), len(requests), 2)
        self.shifts = np.asarray([[to_min(e.shift_start, day_zero(e)), to_min(e.shift_end, day_zero(e))]
                                  for e in engineers], dtype=np.float64)
        # Порядок исполнителей при равенстве — по строковому ID, как в сортировке solver.construct.
        ids = sorted(e.id for e in engineers)
        self.eid_rank = np.asarray([ids.index(e.id) for e in engineers], dtype=np.int64)

        # Компиляция ядра вставки здесь, а не на первом кандидате: время настройки
        # честно попадает в подготовку, а не в первую эпоху поиска.
        table = next(iter(self.tables.values()))
        insertion_options(np.empty(0, dtype=np.int64), np.array([600.]), np.empty(0), 1440., -np.inf,
                          0, 0., 1440., 0., 0, self.km, table, self.step)

    def tight_windows(self, target_min):
        """Патч urgent_insertion (Михаил 25.09): окна, где конец окна аварии — не позже max(начало окна, начало смен) +
        target_min, как отсчёт ожидания в цели (scheduling.urgent_delays). Вставка с такими окнами не выталкивает аварию
        за ориентир. Кэш по target_min."""
        cached = getattr(self, '_tight', None)
        if cached is not None and cached[0] == target_min:
            return cached[1]
        from src.plan.scheduling import day_start
        inst = self.inst
        tight = self.windows.copy()
        start = day_start(inst)
        for ei, e in enumerate(inst.engineers):
            base_day = to_min(start, day_zero(e)) if start is not None else -np.inf
            for qi, r in enumerate(inst.requests):
                if r.urgent:
                    ws, we = tight[ei, qi]
                    tight[ei, qi, 1] = min(we, max(ws, base_day) + target_min)
        self._tight = (target_min, tight)
        return tight

    def options(self, entry, view, req, eng):
        """Допустимые вставки заявки в развёрнутый маршрут (fastroute.RouteView): [(прирост км, позиция)].

        entry — запись кэша маршрута; массивы развёртки кладутся в неё и переиспользуются.
        """
        arrays = entry.get('native_arrays')
        if arrays is None:
            arrays = (np.asarray(view.idx, dtype=np.int64), np.asarray(view.ends, dtype=np.float64),
                      np.asarray(view.latest, dtype=np.float64))
            entry['native_arrays'] = arrays
        ws, we = window_min(req, view.day)
        floor = float(view.floor_min) if view.floor_min is not None else -np.inf
        deltas, valid = insertion_options(*arrays, float(view.shift_end), floor, self.inst.index[req.id],
                                          float(ws), float(we), float(req.duration_min), self.depot,
                                          self.km, self.tables[eng.transport], self.step)
        return [(float(deltas[pos]), pos) for pos in range(len(valid)) if valid[pos]]
