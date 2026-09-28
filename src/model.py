"""Объекты предметной области: заявка, инженер, маршрут, план."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Request:
    id: str                     # ID из синтетических данных
    bk_type: str                # Подключение / Локальная заявка / Дозаказ / Глобальная проблема
    hd_type: str                # подтип работы, из него норматив длительности
    window_start: datetime      # обещанное клиенту окно прибытия
    window_end: datetime
    district: str
    address: str                # как в файле
    address_query: str          # нормализованный для геокодера
    lat: float | None = None
    lon: float | None = None
    skill: str = "local"        # local / connect / emergency
    duration_min: int = 60
    urgent: bool = False
    required_transport: str | None = None   # None = ограничения нет
    gigabit: bool = False
    status: str | None = None   # Статус BK из контрольного файла, если известен
    control_engineer: str | None = None     # бригада в контрольной выборке

    equipment: dict[str, int] = field(default_factory=dict)

    @property
    def priority(self) -> int:
        # Qualification and priority are different: an add-on uses connect skill
        # but belongs to the lowest priority class.
        return 2 if self.urgent else (1 if self.bk_type == 'Подключение' else 0)

    @property
    def geocoded(self) -> bool:
        return self.lat is not None and self.lon is not None


@dataclass
class Engineer:
    id: str
    name: str
    skills: set[str]
    transport: str              # car / public / foot / bike
    shift_start: datetime
    shift_end: datetime
    depot_lat: float
    depot_lon: float
    available: bool = True      # снимается событием «инженер недоступен»
    equipment: dict[str, int] = field(default_factory=dict)


@dataclass
class Stop:
    """Один визит в маршруте: когда выехали, когда приехали, когда начали и закончили."""
    request_id: str
    arrive: datetime            # прибытие на адрес
    start: datetime             # начало работы (не раньше окна)
    end: datetime
    travel_min: float           # время в пути от предыдущей точки
    travel_km: float


@dataclass
class Route:
    engineer_id: str
    stops: list[Stop] = field(default_factory=list)

    @property
    def distance_km(self) -> float:
        return sum(s.travel_km for s in self.stops)

    @property
    def travel_min(self) -> float:
        return sum(s.travel_min for s in self.stops)

    @property
    def used(self) -> bool:
        return bool(self.stops)


@dataclass
class Unassigned:
    request_id: str
    reason: str                 # человеческим языком, для диспетчера
    reason_code: str            # skill / transport / window / shift / no_engineer


@dataclass
class Plan:
    routes: dict[str, Route]
    unassigned: list[Unassigned] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def used_engineers(self) -> int:
        return sum(1 for r in self.routes.values() if r.used)

    @property
    def assigned_count(self) -> int:
        return sum(len(r.stops) for r in self.routes.values())

    @property
    def total_km(self) -> float:
        return sum(r.distance_km for r in self.routes.values())

    urgent_delay_min: float = 0.0   # заполняется при подсчёте метрик
    delay_cost: float = 0.0         # цена задержек аварий в км по режиму (solver.delay_cost)

    priority_counts: tuple[int, int] = (0, 0)
    urgent_late: int = 0            # аварий дольше ориентира; 0, если компонента выключена (solver.plan_scalar)

    def cost_tuple(self) -> tuple:
        """Цель: аварии → подключения → назначено → аварии сверх ориентира → исполнители → хвост → задержка.
        Аварии сверх ориентира — число назначенных аварий, ждущих начала работ дольше ориентира
        (config/weights.json, по умолчанию выключено — тогда 0). Хвост — км плюс цена задержек аварий
        (линейно или квадратично): задержка соревнуется с пробегом; последнее место — сумма задержек."""
        return (-self.priority_counts[0], -self.priority_counts[1],
                -self.assigned_count, self.urgent_late, self.used_engineers,
                round(self.total_km + self.delay_cost, 3), round(self.urgent_delay_min))
