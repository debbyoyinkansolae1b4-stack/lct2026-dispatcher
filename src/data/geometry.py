"""Настоящая геометрия маршрута для карты.

Расстояния мы и так считаем по дорогам, но на карте рисовали прямые между точками —
маршрут шёл через реки и кварталы. Здесь запрашиваем у OSRM линию проезда по всему
маршруту сразу (один запрос на исполнителя, а не на каждый отрезок) и кэшируем на диск,
чтобы на демонстрации не зависеть от сети.

Если OSRM недоступен — возвращаем None, и карта рисует прямые как раньше. Сеть не должна задерживать ответ:
на все запросы одного ответа — общий короткий бюджет (deadline), а после сбоя сеть не трогаем 10 минут.
"""
# refactor: Claude, 26.09.2026 — замечание Codex: был таймаут 60 с на каждый маршрут, ответ мог ждать сеть минутами

import json
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parents[2] / "data" / "interim" / "geometry.json"
OSRM = "https://router.project-osrm.org/route/v1/driving/"
REQUEST_TIMEOUT = 3.0      # с, на один запрос
OFFLINE_PAUSE = 600.0      # с: после сбоя сети не ждём её снова
_offline_until = 0.0


def _load() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


_cache = _load()


def _save() -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(_cache), encoding="utf-8")


def route_line(points: list[tuple[float, float]], deadline: float | None = None) -> list[list[float]] | None:
    """Ломаная проезда через заданные точки. Ключ кэша — сами точки, порядок важен.

    deadline — момент (time.monotonic), позже которого в сеть не идём: бюджет на весь ответ, а не на один маршрут."""
    global _offline_until
    if len(points) < 2:
        return None
    key = ";".join(f"{lat:.5f},{lon:.5f}" for lat, lon in points)
    if key in _cache:
        return _cache[key]
    now = time.monotonic()
    timeout = REQUEST_TIMEOUT if deadline is None else min(REQUEST_TIMEOUT, deadline - now)
    if now < _offline_until or timeout <= 0.2:
        return None                                   # сети нет или бюджет ответа исчерпан — прямые линии

    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    url = f"{OSRM}{coords}?overview=full&geometries=geojson"
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=timeout).read())
    except Exception:
        _offline_until = time.monotonic() + OFFLINE_PAUSE
        return None
    if data.get("code") != "Ok" or not data.get("routes"):
        return None

    line = [[lat, lon] for lon, lat in data["routes"][0]["geometry"]["coordinates"]]
    _cache[key] = line
    _save()
    return line
