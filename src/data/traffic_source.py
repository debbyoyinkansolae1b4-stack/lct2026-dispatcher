"""Откуда брать времена в пути с учётом пробок.

Реальность: посегментных пробок бесплатно не отдаёт никто. TomTom, HERE, Яндекс и 2ГИС
требуют ключ (проверено — без него 401), OSRM считает свободный поток. Поэтому здесь
не имитация интеграции, а разъём под неё:

  профиль по часам — работает сразу, без ключей, типовая кривая города;
  файл со слоями   — оператор подгружает заранее выкачанные матрицы; этим же путём
                     заходит история поездок компании и любой прогноз;
  внешний сервис   — адрес и ключ задаются в настройках; пока ключа нет, источник
                     честно сообщает об этом, а не подсовывает выдуманные числа.

Формат файла: {"step_min": 60, "layers": {"480": [[мин, ...], ...], "540": [...]}},
ключ — минута начала интервала от полуночи, значение — матрица времён на машине
в том же порядке точек, что и в матрице расстояний.
"""

import json
from pathlib import Path


class TrafficSourceError(Exception):
    pass


def load_from_file(path: str | Path, expected_size: int) -> tuple[dict, int]:
    """Читает слои из файла. Возвращает ({минута: матрица}, шаг в минутах)."""
    p = Path(path)
    if not p.exists():
        raise TrafficSourceError(f"файл не найден: {p}")
    return parse(p.read_text(encoding="utf-8"), expected_size)


def parse(text: str, expected_size: int) -> tuple[dict, int]:
    """Слои из текста файла (JSON в формате выше): ({минута: матрица}, шаг в минутах)."""
    try:
        blob = json.loads(text)
    except json.JSONDecodeError as e:
        raise TrafficSourceError(f"файл не читается как JSON: {e}") from e
    if not isinstance(blob, dict):
        raise TrafficSourceError("в файле нет объекта layers с матрицами")

    raw = blob.get("layers")
    if not isinstance(raw, dict) or not raw:
        raise TrafficSourceError("в файле нет объекта layers с матрицами")

    layers = {}
    for key, matrix in raw.items():
        try:
            start = int(key)
        except (TypeError, ValueError):
            raise TrafficSourceError(f"ключ слоя «{key}» не минута от полуночи") from None
        if len(matrix) != expected_size or any(len(row) != expected_size for row in matrix):
            raise TrafficSourceError(
                f"слой {key}: матрица {len(matrix)}×{len(matrix[0]) if matrix else 0}, "
                f"а точек в участке {expected_size}")
        layers[start % (24 * 60)] = matrix
    return layers, int(blob.get("step_min", 60))


# refactor: Claude, 26.09.2026 — шаг 8: готовые файлы слоёв в проекте (scripts/build_traffic_layers.py → data/traffic/)
LAYERS_DIR = Path(__file__).resolve().parents[2] / "data" / "traffic"


def list_files(region: str, points: int) -> list[dict]:
    """Файлы слоёв из data/traffic/ (демо) и data/traffic/uploads/ (загруженные), подходящие набору: тот же набор и то же
    число точек."""
    out = []
    for path in sorted(LAYERS_DIR.glob("*.json")) + sorted(LAYERS_DIR.glob("uploads/*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if blob.get("region") == region and blob.get("points") == points:
            out.append({"path": str(path), "name": path.name, "note": blob.get("note", ""),
                        "layers": len(blob.get("layers", {})), "step_min": blob.get("step_min", 60)})
    return out


def save_upload(region: str, filename: str, text: str, expected_size: int) -> Path:
    """Файл слоёв, загруженный в админке: проверяется как при чтении и кладётся в data/traffic/uploads/ — дальше он в списке
    набора, как демонстрационные. Имя — «набор-имя файла.json»; набор и число точек дописываются в файл."""
    parse(text, expected_size)
    blob = json.loads(text)
    blob.update(region=region, points=expected_size)
    blob.setdefault("note", f"загружен в админке: {filename}")
    stem = "".join(ch for ch in Path(filename).stem if ch.isalnum() or ch in "-_ ")[:60].strip() or "слои"
    path = LAYERS_DIR / "uploads" / f"{region}-{stem}.json"      # загруженные — отдельно от демо, вне git
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def fetch_from_provider(provider: str, api_key: str, points, hours) -> tuple[dict, int]:
    """Забор матриц у платного сервиса. Без ключа не работает — так и сообщаем."""
    if not api_key:
        raise TrafficSourceError(
            f"для источника «{provider}» нужен ключ API. Бесплатного доступа к пробкам "
            f"нет ни у одного сервиса: TomTom, HERE, Яндекс и 2ГИС отвечают 401 без ключа. "
            f"У TomTom и HERE есть бесплатный тариф разработчика — ключ выдаётся при "
            f"регистрации, покрытие России стоит проверить до подключения.")
    raise TrafficSourceError(
        f"адаптер для «{provider}» не реализован: писать запросы к платному API, "
        f"не имея возможности их проверить, значит везти на защиту непроверенный код. "
        f"Рабочий путь сейчас — выкачать матрицы отдельно и подгрузить файлом.")


def describe_sources(has_key: bool) -> list[dict]:
    return [
        {"id": "profile", "title": "Профиль по часам",
         "note": "Работает сразу. Типовая кривая города: утренний и вечерний пик, "
                 "пятница тяжелее, выходные легче. Один коэффициент на весь участок."},
        {"id": "file", "title": "Файл со слоями",
         "note": "Матрицы времён по интервалам, выкачанные заранее. У каждой пары точек "
                 "свой профиль — въезд в центр и выезд из него считаются по-разному."},
        {"id": "provider", "title": "Внешний сервис",
         "note": "Ключ задан, можно забирать матрицы" if has_key else
                 "Нужен ключ API: бесплатного доступа к пробкам нет ни у одного сервиса"},
    ]
