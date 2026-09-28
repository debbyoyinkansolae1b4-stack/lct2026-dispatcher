"""Реестр наборов данных: встроенные участки (эталон) и добавленные — варианты со сгенерированными
бригадами и загруженные CSV.

Встроенные три участка — синтетические файлы организаторов, бригады восстановлены по контрольному
дню (допущение команды; доступный состав участка — оттуда же). Их идентификаторы — названия
участков: на них опираются замеры, эталон и тесты, менять нельзя. Добавленные наборы записываются
в config/datasets.json: вариант ссылается на заявки базового набора и задаёт бригады параметрами
с сидом (воспроизводимо); загруженный — на свой файл в data/uploads/.
"""
# refactor: Claude, 23.09.2026 — наборы данных: реестр, генерация бригад, загрузка CSV (решение Михаила)
import base64
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parents[2] / "config"
REGISTRY = CONFIG / "datasets.json"
UPLOADS = ROOT / "data" / "uploads"
BUILTIN = ("Восток", "Юго-восток", "Югоцентр")


def _added() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return json.loads(REGISTRY.read_text(encoding="utf-8")).get("datasets", [])


def _save(entries: list[dict]) -> None:
    REGISTRY.write_text(json.dumps({"_comment": __doc__.strip().splitlines()[0], "datasets": entries},
                                   ensure_ascii=False, indent=2), encoding="utf-8")


def entries() -> list[dict]:
    """Все наборы: сначала встроенные, затем добавленные — в порядке добавления."""
    builtin = [{"id": name, "title": name, "builtin": True, "brigades": {"source": "control"}} for name in BUILTIN]
    return builtin + _added()


def ids() -> list[str]:
    return [e["id"] for e in entries()]


def get(dataset_id: str) -> dict:
    for entry in entries():
        if entry["id"] == dataset_id:
            return entry
    raise KeyError(f"Нет набора «{dataset_id}»")


def root_of(dataset_id: str) -> str:
    """Набор, чьи заявки и матрицу расстояний использует этот (вариант — базовый участок)."""
    entry = get(dataset_id)
    return root_of(entry["base"]) if entry.get("base") else entry["id"]


def _unique(slug: str) -> str:
    taken, base, n = set(ids()), slug, 2
    while slug in taken:
        slug, n = f"{base}-{n}", n + 1
    return slug


def check_brigades(count, emergency, connect, local, seed) -> list[str]:
    integer = lambda v: isinstance(v, int) and not isinstance(v, bool)
    problems = []
    if not integer(count) or not 1 <= count <= 200:
        problems.append("бригад — целое от 1 до 200")
    for name, value in (("с аварийным допуском", emergency), ("с подключениями", connect), ("с ремонтом", local)):
        if not integer(value) or value < 0 or (integer(count) and value > count):
            problems.append(f"бригад {name} — целое от 0 до числа бригад")
    if not integer(seed):
        problems.append("сид — целое")
    if not problems and emergency + connect + local < count:
        problems.append("навыков меньше, чем бригад: у бригады без навыков нет работы")
    return problems


def generate(base: str, count: int, emergency: int, connect: int, local: int, seed: int,
             title: str | None = None) -> dict:
    """Вариант набора: заявки базового, бригады — по параметрам с сидом. Базовый не меняется."""
    get(base)
    problems = check_brigades(count, emergency, connect, local, seed)
    if problems:
        raise ValueError("; ".join(problems))
    entry = {"id": _unique(f"{base}~бригады-{count}-{emergency}ав-с{seed}"),
             "title": title or f"{base}: {count} бригад, {emergency} с аварийным допуском (сид {seed})",
             "base": base, "created": f"{datetime.now():%d.%m.%Y %H:%M}",
             "brigades": {"source": "generated", "count": count, "emergency": emergency, "connect": connect,
                          "local": local, "seed": seed}}
    _save(_added() + [entry])
    return entry


def remove(dataset_id: str) -> None:
    entry = get(dataset_id)
    if entry.get("builtin"):
        raise ValueError("Встроенный набор удалить нельзя")
    if any(e.get("base") == dataset_id for e in _added()):
        raise ValueError("Сначала удалите варианты этого набора")
    if entry.get("file"):
        # Загруженный набор уносит с собой файл и свой кэш матрицы (вариант делит матрицу с базовым — её не трогаем).
        from src import travel
        from src.instance import _matrix_key
        (travel.CACHE_DIR / f"travel_{_matrix_key(dataset_id)}.json").unlink(missing_ok=True)
    _save([e for e in _added() if e["id"] != dataset_id])
    if entry.get("file"):
        (ROOT / entry["file"]).unlink(missing_ok=True)


def upload(filename: str, content_b64: str, office: str | None = None, brigades: dict | None = None,
           title: str | None = None) -> tuple[dict, dict]:
    """Сохранить загруженный CSV, разобрать его и зарегистрировать набор. Возвращает (набор, отчёт).

    Бригады — сгенерированные (по умолчанию 12, из них 4 с аварийным допуском, все с подключениями
    и ремонтом, сид 1): у загруженного файла нет контрольного дня.
    """
    from src.data.loading import read_requests_file
    raw = base64.b64decode(content_b64)
    if not raw:
        raise ValueError("Пустой файл")
    brigades = brigades or {"count": 12, "emergency": 4, "connect": 12, "local": 12, "seed": 1}
    problems = check_brigades(brigades.get("count"), brigades.get("emergency"), brigades.get("connect"),
                              brigades.get("local"), brigades.get("seed"))
    if problems:
        raise ValueError("; ".join(problems))
    stem = re.sub(r"[^\w.-]+", "_", Path(filename or "набор.csv").stem, flags=re.UNICODE)[:40] or "набор"
    dataset_id = _unique(f"загружен~{stem}")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{dataset_id.replace('~', '_')}.csv"
    path.write_bytes(raw)
    try:
        report = read_requests_file(path, office=office, geocode_missing=True)["report"]
    except Exception:
        path.unlink(missing_ok=True)
        raise
    stored = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    entry = {"id": dataset_id, "title": title or Path(filename or "набор.csv").name, "file": stored,
             "office": office, "created": f"{datetime.now():%d.%m.%Y %H:%M}",
             "brigades": {"source": "generated", **brigades}}
    _save(_added() + [entry])
    return entry, report
