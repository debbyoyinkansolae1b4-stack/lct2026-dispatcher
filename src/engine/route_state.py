"""Закреплённое начало маршрутов и откуда стартует подвижный хвост: один снимок на запуск поиска.

Единственное место, где разбирается история маршрутов (inst.committed, момент события).
Поиск, граф, операторы и восстановление читают отсюда: какие заявки закреплены, сколько
визитов в начале маршрута нельзя трогать, из какой точки и в какой момент исполнитель
готов к новой работе, сколько оборудования уже израсходовано.

Снимок фиксирован на время жизни кэша поиска: история и момент события во время поиска
не меняются. Новое событие — новая задача, новый снимок, новый кэш (Н12, Н29).
Проверка совместимости — в местах подключения (check), а не на каждом кандидате.

Утренняя задача — тот же снимок с пустой историей: хвост = весь маршрут, старт из офиса
в начало смены. Отдельных веток «статика/событие» у потребителей нет — есть данные.
"""
# refactor: Claude, 21.09.2026 — шаг 5г.1 (EVENTS-CLEANUP-PLAN-CODEX.md)
from dataclasses import dataclass

from src.plan.fastroute import day_zero, to_min


@dataclass(frozen=True)
class Tail:
    """Граница закреплённого и подвижного у одного исполнителя."""
    fixed: tuple            # закреплённые Stop, как в плане
    fixed_ids: tuple        # их заявки по порядку
    origin: int             # точка матрицы, откуда стартует хвост
    ready: float            # когда готов к новой работе: минуты от полуночи дня исполнителя, точно
    floor: float            # раньше этого момента новую работу не начинать; -inf — без ограничения
    used: tuple             # расход оборудования историей: ((предмет, количество), ...)
    km: tuple               # пробег истории по остановкам, как записан в Stop
    available: bool         # может ли получать новую работу
    already_used: bool      # уже задействован историей: штрафа открытия нет, в составе считается

    @property
    def lock(self):
        return len(self.fixed)


class RouteState:
    """Снимок истории маршрутов задачи. has_history — есть ли что закреплять или момент, раньше которого нельзя."""

    def __init__(self, inst):
        self.inst = inst
        self.signature = self._signature(inst)
        self.has_history = (bool(inst.committed) or inst.earliest_start is not None
                            or bool(inst.completed) or bool(inst.cancelled))
        depot = inst.index['depot']
        self.tails = {}
        for eng in inst.engineers:
            day = day_zero(eng)
            fixed = tuple(inst.committed.get(eng.id, ()))
            # Начало смены — как в утренних массивах ядра (целые минуты): эталон побайтово.
            ready = float(to_min(eng.shift_start, day))
            floor = -float('inf')
            if fixed:
                ready = _minutes(fixed[-1].end, day)
            if inst.earliest_start is not None:
                floor = _minutes(inst.earliest_start, day)
                ready = max(ready, floor)
            used = {}
            for stop in fixed:
                # Завершённые заявки не входят в активные — расход берётся из самих заявок.
                for item, quantity in inst.by_id[stop.request_id].equipment.items():
                    used[item] = used.get(item, 0) + quantity
            self.tails[eng.id] = Tail(
                fixed=fixed, fixed_ids=tuple(s.request_id for s in fixed),
                origin=inst.index[fixed[-1].request_id] if fixed else depot, ready=ready, floor=floor,
                used=tuple(sorted(used.items())), km=tuple(s.travel_km for s in fixed),
                available=eng.available, already_used=bool(fixed))
        self.pinned = frozenset(rid for tail in self.tails.values() for rid in tail.fixed_ids)

    @staticmethod
    def _signature(inst):
        """Всё, от чего зависит снимок: момент события, исключения, история, доступность."""
        return (inst.earliest_start, frozenset(inst.earliest_exempt or ()),
                tuple((eid, tuple((s.request_id, s.end) for s in stops)) for eid, stops in sorted(inst.committed.items())),
                tuple(e.available for e in inst.engineers))

    def check(self, inst):
        """Место подключения: задача та же и её история не менялась с момента снимка."""
        if inst is not self.inst:
            raise ValueError('Route state belongs to a different instance')
        if self._signature(inst) != self.signature:
            raise ValueError('Route history changed since the snapshot: build a new route state and cache')

    def movable(self, eid, order):
        """Подвижный хвост маршрута: без закреплённого начала."""
        return order[self.tails[eid].lock:]

    def locked(self):
        """Сколько визитов в начале маршрута закреплено — только у исполнителей с историей."""
        return {eid: tail.lock for eid, tail in self.tails.items() if tail.lock}

    def evaluate_kwargs(self, eng):
        """Аргументы точного расчёта маршрута (scheduling.evaluate) для этого снимка."""
        inst = self.inst
        return dict(earliest=inst.earliest_start, exempt=inst.earliest_exempt,
                    committed=inst.committed.get(eng.id, []))

    def owns(self, eng, kwargs):
        """Вызов evaluate с параметрами именно этого снимка — результат можно кэшировать по порядку."""
        inst = self.inst
        committed = kwargs.get('committed') or ()
        fixed = self.tails[eng.id].fixed
        return (kwargs.get('earliest') == inst.earliest_start
                and (kwargs.get('exempt') or set()) == (inst.earliest_exempt or set())
                and len(committed) == len(fixed) and all(a is b for a, b in zip(committed, fixed)))


def _minutes(moment, day):
    """Точные минуты от полуночи дня исполнителя, без округления (дробные — от секунд)."""
    return (moment - day).total_seconds() / 60
