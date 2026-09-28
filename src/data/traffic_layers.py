"""Слои пробок: матрица времён на машине по часам для каждой пары точек набора (формат файла «Файл со слоями»).

Демонстрационные слои строятся от свободной дороги набора: городская кривая профиля (config/assumptions.json) с поправкой
на день недели, утром дорога в центр тяжелее, вечером — из центра, у каждой пары своя постоянная «тяжесть» и небольшой
почасовой шум (с сидом — воспроизводимо), пробки — только на городской части плеча. Это не настоящие пробки: ими
пользуются scripts/build_traffic_layers.py (демо-файлы участков) и админка («скачать пример» для выбранного набора —
образец формата, который можно заменить своими матрицами и загрузить).
"""
# refactor: Claude, 26.09.2026 — из scripts/build_traffic_layers.py: пример слоёв для любого набора в админке
import math
import random

from src import settings
from src.travel import TravelMatrix, haversine_km

CENTER = (55.7520, 37.6175)          # Кремль
MORNING, EVENING = range(6, 12), range(15, 21)
DIRECTION = 0.5                      # насколько направление усиливает или ослабляет пробку в пик
ROAD_SIGMA, HOUR_SIGMA = 0.15, 0.04


def layers_for(inst, seed: int) -> dict:
    m = inst.matrix
    traffic = settings._load("assumptions")["travel"]["traffic"]
    hourly = traffic["hourly"]
    day = traffic.get("weekday", {}).get(str(m.traffic.weekday), 1.0)
    n = len(m.points)
    to_center = [haversine_km(p, CENTER) for p in m.points]
    rng = random.Random(seed)
    road = [[math.exp(rng.gauss(0, ROAD_SIGMA)) for _ in range(n)] for _ in range(n)]
    layers = {}
    for h in range(24):
        excess = hourly[h] * day - 1.0
        sign = 1 if h in MORNING else -1 if h in EVENING else 0
        layer = []
        for i in range(n):
            row = []
            for j in range(n):
                free, km = m.car_min[i][j], m.km[i][j]
                if i == j or free <= 0:
                    row.append(0.0)
                    continue
                inbound = max(-1., min(1., (to_center[i] - to_center[j]) / km)) if km > 0 else 0.
                e = excess * (1 + DIRECTION * sign * inbound) if excess > 0 else excess
                e *= road[i][j] * math.exp(rng.gauss(0, HOUR_SIGMA))
                city = min(1., TravelMatrix.CITY_LIMIT_KM / km) if km > 0 else 1.
                row.append(round(free * (1 + max(-0.5, e) * city), 2))
            layer.append(row)
        layers[str(h * 60)] = layer
    return layers




NOTE = ("демонстрационные слои: не настоящие пробки — городская кривая профиля, утром в центр тяжелее, вечером из центра, "
        "у каждой дороги своя тяжесть. Формат: layers[минуты от полуночи] — матрица минут на машине, "
        "строки и столбцы — точки набора в порядке задачи (0 — офис)")


def sample_blob(region: str, inst, seed: int = 1, note: str = NOTE) -> dict:
    """Файл слоёв для набора region в формате, который принимает загрузка."""
    return {"region": region, "points": len(inst.matrix.points), "step_min": 60, "seed": seed, "note": note,
            "layers": layers_for(inst, seed)}
