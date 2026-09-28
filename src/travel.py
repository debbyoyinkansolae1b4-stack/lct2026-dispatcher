"""Матрица расстояний между точками участка и время в пути по типу транспорта.

Загруженность дорог живёт отдельно, в traffic.py: здесь только география и скорости.

Расстояние берём по дорогам из OSRM — постановщик прямо просил не по прямой [17:49].
Со временем сложнее: публичный OSRM отдаёт только профиль driving, а у нас четыре режима
передвижения. Поэтому для машины берём duration из OSRM, а для пешехода, велосипеда
и общественного транспорта считаем от того же дорожного расстояния и средней скорости
режима плюс фиксированная надбавка (парковка, ожидание транспорта, подъём к клиенту —
всё это постановщик разрешил не моделировать отдельно [12:18]).

Если OSRM недоступен, падаем на гаверсинус × 1.35 (типичный коэффициент извилистости улиц
в городе) и честно помечаем это в матрице — допущение идёт в README.
"""

import json
import math
import time
import urllib.request
from pathlib import Path

from src.traffic import TrafficModel

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "interim"
OSRM = "https://router.project-osrm.org/table/v1/driving/"
DETOUR = 1.35        # во сколько раз путь по улицам длиннее прямой
MAX_POINTS = 95      # предел публичного OSRM на размер таблицы


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


class TravelMatrix:
    """km[i][j] — дорожное расстояние, minutes(i, j, mode, выезд) — время в пути."""

    # За городом скорость другая: до Каширы 110 км, и городские 18 км/ч там бессмысленны
    CITY_LIMIT_KM = 15.0
    INTERCITY_KMH = 45.0
    # Пешком и на велосипеде дальние заявки не делают: человек садится на транспорт
    SELF_POWERED_LIMIT_KM = {"foot": 5.0, "bike": 15.0}

    def __init__(self, points: list[tuple[float, float]], speeds: dict, overhead: dict,
                 km: list[list[float]], car_min: list[list[float]], source: str,
                 traffic: dict | None = None):
        self.points = points
        self.speeds = speeds
        self.overhead = overhead
        self._km = km
        self._car_min = car_min
        self.source = source          # osrm | haversine
        self.traffic = TrafficModel(traffic)
        # Когда загруженность не учитывается, время в пути зависит только от пары точек
        # и режима — значит его можно посчитать один раз на режим. Поиск обращается
        # к матрице сотни тысяч раз за секунду, и разница между «вычислить» и «взять
        # из таблицы» определяет, сколько итераций поместится в бюджет.
        self._mode_cache: dict[str, list[list[float]]] = {}

    @property
    def weekday(self) -> int:
        return self.traffic.weekday

    @weekday.setter
    def weekday(self, value: int) -> None:
        self.traffic.weekday = value

    # Базовые матрицы отдаём через свойства: их подменяют и в тестах, и при добавлении
    # точки срочной заявкой. Любая подмена обнуляет предвычисленные таблицы режимов,
    # иначе поиск продолжит считать по устаревшим числам и будет прав только на вид.
    @property
    def km(self) -> list[list[float]]:
        return self._km

    @km.setter
    def km(self, value) -> None:
        self._km = value
        self._mode_cache.clear()

    @property
    def car_min(self) -> list[list[float]]:
        return self._car_min

    @car_min.setter
    def car_min(self, value) -> None:
        self._car_min = value
        self._mode_cache.clear()

    def _plain_matrix(self, mode: str) -> list[list[float]] | None:
        """Готовая таблица минут для режима, если загруженность не учитывается.

        Таблица пересобирается, когда в матрице появились новые точки: срочная заявка
        в течение дня добавляет свою, и старая таблица становится короче матрицы.
        Проверка по размеру дешевле, чем помнить об очистке кэша в чужом коде.
        """
        if self.traffic.enabled or self.traffic.layers:
            return None
        n = len(self.points)
        cached = self._mode_cache.get(mode)
        if cached is None or len(cached) != n:
            cached = [[self._compute_minutes(i, j, mode) for j in range(n)] for i in range(n)]
            self._mode_cache[mode] = cached
        return cached

    def minutes(self, i: int, j: int, mode: str, depart_min: float | None = None) -> float:
        if i == j:
            return 0.0
        table = self._plain_matrix(mode)
        if table is not None:
            return table[i][j]
        return self._compute_minutes(i, j, mode, depart_min)

    def _compute_minutes(self, i: int, j: int, mode: str, depart_min: float | None = None) -> float:
        if i == j:
            return 0.0
        km = self.km[i][j]
        extra = self.overhead.get(mode, 0.0)
        # Время из внешнего источника уже учитывает загруженность этой дороги,
        # коэффициент к нему применять нельзя — иначе пробка посчитается дважды
        from_source = self.traffic.layer_minutes(i, j, depart_min)
        factor = 1.0 if from_source is not None else self.traffic.factor(depart_min, mode)

        if mode == "car" or self.speeds.get(mode) is None:
            return (from_source if from_source is not None else self.car_min[i][j]) * factor + extra
        if from_source is not None and self.car_min[i][j] > 0:
            # Слой задаёт время на машине; прочий транспорт страдает от той же загруженности в меру своей
            # чувствительности — как у профиля (Н90). refactor: Claude, 26.09.2026
            ratio = from_source / self.car_min[i][j]
            factor = 1.0 + (ratio - 1.0) * self.traffic.sensitivity.get(mode, 1.0)

        limit = self.SELF_POWERED_LIMIT_KM.get(mode)
        if limit is not None and km > limit:
            mode = "public"          # дальше своих сил — общественным транспортом
            extra = self.overhead.get("public", extra)

        speed = self.speeds.get(mode) or self.INTERCITY_KMH
        if km <= self.CITY_LIMIT_KM:
            return km / speed * 60.0 * factor + extra
        # За городом пробки почти не действуют — трасса есть трасса
        city_part = self.CITY_LIMIT_KM / speed * 60.0 * factor
        return city_part + (km - self.CITY_LIMIT_KM) / self.INTERCITY_KMH * 60.0 + extra

    def load_layers(self, layers: dict[int, list[list[float]]], step_min: int = 60) -> None:
        """Подключить внешний источник времён с учётом пробок. См. traffic.py."""
        self.traffic.load_layers(layers, step_min, expected_size=len(self.points))

    @property
    def time_source(self) -> str:
        return self.traffic.describe()

    def worst_minutes(self, i: int, j: int, mode: str, from_min: float, to_min: float) -> float:
        """Самая долгая дорога при выезде в интервале [from_min, to_min].

        Нужна обратному проходу: момент выезда там ещё неизвестен, а оценка обязана быть
        осторожной. Взять время на границе интервала нельзя — пик может быть внутри,
        и проверка вставки начнёт пропускать недопустимые варианты.
        """
        if i == j:
            return 0.0
        if not self.traffic.enabled and not self.traffic.layers:
            return self.minutes(i, j, mode)
        lo, hi = int(min(from_min, to_min)), int(max(from_min, to_min))
        probes = {lo, hi} | {h * 60 for h in range(lo // 60, hi // 60 + 2) if lo <= h * 60 <= hi}
        return max(self.minutes(i, j, mode, p) for p in probes)

    def distance(self, i: int, j: int) -> float:
        return self.km[i][j]


def _osrm_table(points: list[tuple[float, float]], tries: int = 3) -> tuple[list, list] | None:
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    url = f"{OSRM}{coords}?annotations=duration,distance"
    raw = None
    for attempt in range(tries):
        try:
            raw = urllib.request.urlopen(url, timeout=120).read()
            break
        except Exception as e:
            # Публичный OSRM прикрывается при частых запросах подряд; три участка
            # считаются один за другим, поэтому пауза и повтор, а не сразу запасной путь
            print(f"[travel] попытка {attempt + 1}/{tries} не удалась ({type(e).__name__}: {e})")
            time.sleep(3.0 * (attempt + 1))
    if raw is None:
        print("[travel] OSRM недоступен, падаем на гаверсинус")
        return None
    data = json.loads(raw)
    if data.get("code") != "Ok":
        print(f"[travel] OSRM вернул {data.get('code')}, падаем на гаверсинус")
        return None
    km = [[(d or 0.0) / 1000.0 for d in row] for row in data["distances"]]
    minutes = [[(t or 0.0) / 60.0 for t in row] for row in data["durations"]]
    return km, minutes


def build(points: list[tuple[float, float]], speeds: dict, overhead: dict,
          cache_key: str | None = None, use_osrm: bool = True,
          traffic: dict | None = None) -> TravelMatrix:
    cache_file = CACHE_DIR / f"travel_{cache_key}.json" if cache_key else None
    if cache_file and cache_file.exists():
        blob = json.loads(cache_file.read_text())
        fresh = blob["points"] == [list(p) for p in points]
        if fresh and not (use_osrm and blob.get("source") != "osrm"):
            return TravelMatrix(points, speeds, overhead, blob["km"], blob["car_min"],
                                blob["source"], traffic)

    table = _osrm_table(points) if (use_osrm and len(points) <= MAX_POINTS) else None
    if table:
        km, car_min = table
        source = "osrm"
    else:
        n = len(points)
        km = [[haversine_km(points[i], points[j]) * DETOUR for j in range(n)] for i in range(n)]
        # 24 км/ч — средняя скорость машины по городу с учётом светофоров
        car_min = [[km[i][j] / 24.0 * 60.0 for j in range(n)] for i in range(n)]
        source = "haversine"

    # Запасную матрицу на диск не пишем: один сорвавшийся запрос иначе навсегда
    # подменяет дороги прямыми линиями, и об этом уже не вспомнишь
    if cache_file and source == "osrm":
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(
            {"points": [list(p) for p in points], "km": km, "car_min": car_min, "source": source}))
    return TravelMatrix(points, speeds, overhead, km, car_min, source, traffic)
