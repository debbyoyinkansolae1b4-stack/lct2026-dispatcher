"""Сборка инстанса участка: заявки, исполнители, матрица расстояний, индексы."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from src import travel
from src.data.loading import load_region
from src.model import Engineer, Request

CONFIG = Path(__file__).resolve().parents[1] / "config"


@dataclass
class Instance:
    region: str
    requests: list[Request]
    engineers: list[Engineer]
    depot: dict
    matrix: travel.TravelMatrix
    index: dict                      # id заявки → номер точки в матрице, 'depot' → 0
    by_id: dict = field(default_factory=dict)
    nogeo: list[Request] = field(default_factory=list)
    # Момент события при перепланировании: раньше него начинать работу нельзя,
    # иначе система «назначает в прошлое» — план выглядит допустимым, но неисполним
    earliest_start = None
    # Заявки, на которые запрет «не раньше события» не распространяется: их уже начали
    # или выполнили до события, и переносить их вперёд бессмысленно
    earliest_exempt: set = field(default_factory=set)
    # Заявки, отменённые в течение дня: в планирование больше не попадают,
    # но остаются в by_id, чтобы история и объяснения не потеряли ссылку
    cancelled: set = field(default_factory=set)
    completed: set = field(default_factory=set)
    # Exact committed prefix: departed travel, waiting and service cannot be rewritten.
    committed: dict = field(default_factory=dict)
    # После события: кто вёз каждую заявку в плане до него — цель берёт цену за смену исполнителя
    # (solver.W_EVENT_MOVE, «цена переназначения после события»). Пусто у плана дня.
    previous_owner: dict = field(default_factory=dict)

    def request(self, rid: str) -> Request:
        return self.by_id[rid]


def _matrix_key(region: str) -> str:
    """Ключ кэша матрицы: вариант набора делит её с базовым (те же точки)."""
    from src.data import datasets
    return datasets.root_of(region).replace(" ", "_").replace("~", "_")


def build(region: str, use_osrm: bool = True) -> Instance:
    data = load_region(region)
    reqs_all, engs, depot = data["requests"], data["engineers"], data["depot"]
    asm = json.loads((CONFIG / "assumptions.json").read_text(encoding="utf-8"))

    if depot["lat"] is None:
        raise RuntimeError(f"Не геокодирован офис участка {region}: {depot['address']}")

    # Заявки без координат в маршруты не ставим — их нельзя ни доехать, ни показать на карте.
    # Они попадут в неназначенные с отдельной причиной, это честнее выдуманной точки.
    reqs = [r for r in reqs_all if r.geocoded]
    nogeo = [r for r in reqs_all if not r.geocoded]

    points = [(depot["lat"], depot["lon"])] + [(r.lat, r.lon) for r in reqs]
    index = {"depot": 0}
    for i, r in enumerate(reqs, start=1):
        index[r.id] = i

    matrix = travel.build(
        points,
        speeds=asm["travel"]["speed_kmh"],
        overhead=asm["travel"]["fixed_overhead_min"],
        cache_key=_matrix_key(region),
        use_osrm=use_osrm,
        traffic=asm["travel"].get("traffic"),
    )
    if reqs:
        matrix.weekday = reqs[0].window_start.weekday()      # 17.08.2026 — понедельник
    return Instance(region=region, requests=reqs, engineers=engs, depot=depot,
                    matrix=matrix, index=index,
                    by_id={r.id: r for r in reqs_all}, nogeo=nogeo)
