"""Геокодирование адресов с общим кэшем.

Основной источник — Photon (тот же OSM, но без жёсткого рейт-лимита), резервный — Nominatim.
Nominatim в лоб не годится: он режет на 1 запрос в секунду с IP, а по этим адресам
параллельно ходит и вторая реализация — ловим 429 и получаем пустые ответы.

Важно: ошибку сети и «адрес не найден» различаем. Ошибку не кэшируем никогда, иначе
один 429 навсегда превращает нормальный адрес в промах.

Кэш: data/interim/geocode.json — общий, координаты это данные, а не проектное решение.

    python -m src.data.geocode --stats
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from src.data.addresses import normalize

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "interim" / "geocode.json"
UA = "lct2026-beeline-router/0.1 (hackathon prototype)"

MOSCOW = (55.7558, 37.6173)
MAX_KM_FROM_MOSCOW = 160.0     # в наборе есть Кашира и Ступино (~110 км), но не Урал


class Miss(Exception):
    """Адрес честно не найден — в отличие от сетевой ошибки, это кэшируем."""


def _get(url: str, tries: int = 3) -> dict:
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"не достучались: {last}")


def _far_from_moscow(lat: float, lon: float) -> bool:
    from src.travel import haversine_km
    return haversine_km((lat, lon), MOSCOW) > MAX_KM_FROM_MOSCOW


def _photon(query: str) -> dict:
    data = _get("https://photon.komoot.io/api?limit=1&q=" + quote(query))
    feats = data.get("features") or []
    if not feats:
        raise Miss(query)
    props, (lon, lat) = feats[0]["properties"], feats[0]["geometry"]["coordinates"]
    if _far_from_moscow(lat, lon):
        raise Miss(f"{query} → промах по географии ({lat:.3f},{lon:.3f})")
    # Номер дома держим отдельным полем, а не в склеенной строке: сравнивать его
    # надо точно, иначе «83с4» примет «83к1» за тот же дом
    return {"lat": lat, "lon": lon, "source": "photon", "query": query,
            "precision": props.get("type", ""),
            "housenumber": str(props.get("housenumber", "") or ""),
            "street": str(props.get("street", "") or ""),
            "matched": " ".join(str(props.get(k, "")) for k in ("city", "street", "housenumber")).strip()}


def _nominatim(query: str) -> dict:
    data = _get("https://nominatim.openstreetmap.org/search?format=json&addressdetails=1"
                "&limit=1&q=" + quote(query))
    time.sleep(1.1)                       # их правило: не чаще одного запроса в секунду
    if not data:
        raise Miss(query)
    hit = data[0]
    lat, lon = float(hit["lat"]), float(hit["lon"])
    if _far_from_moscow(lat, lon):
        raise Miss(f"{query} → промах по географии")
    addr = hit.get("address", {}) or {}
    return {"lat": lat, "lon": lon, "source": "nominatim", "query": query,
            "precision": hit.get("addresstype", ""),
            "housenumber": str(addr.get("house_number", "") or ""),
            "street": str(addr.get("road", "") or ""),
            "matched": hit.get("display_name", "")}


def load_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def geocode(raw_address: str, cache: dict) -> dict | None:
    """Варианты запроса × два сервиса. Первый дом побеждает; улица без дома — последняя надежда."""
    if raw_address in cache and cache[raw_address].get("lat") is not None:
        return cache[raw_address]

    queries = normalize(raw_address)["queries"]
    errors = []
    for level, q in enumerate(queries):
        for provider in (_photon, _nominatim):
            try:
                hit = provider(q)
                hit["fallback_level"] = level      # 0 — попали с первого варианта
                cache[raw_address] = hit
                return hit
            except Miss:
                continue
            except Exception as e:
                errors.append(f"{provider.__name__}: {type(e).__name__}")

    if errors:                                     # сеть подвела — промах не фиксируем
        print(f"  [сеть] {raw_address[:60]}: {', '.join(errors[:3])}", file=sys.stderr)
        return None

    cache[raw_address] = {"lat": None, "lon": None, "tried": queries, "status": "not_found"}
    return None


if __name__ == "__main__":
    c = load_cache()
    hits = sum(1 for v in c.values() if v.get("lat") is not None)
    print(f"{CACHE}: {len(c)} адресов, найдено {hits}, промахов {len(c) - hits}")
