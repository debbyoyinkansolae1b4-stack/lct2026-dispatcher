"""Длительность работы на адресе по официальной таблице нормативов.

Базовый норматив организаторов включает резерв на дорогу (20 минут). При распределении
резерв заменяется рассчитанной дорогой, поэтому в заявку идёт только время на адресе:
техническая работа плюс документы. Дорогу добавляет расписание маршрута по матрице.
Пример из разъяснения: авария с дорогой 5 минут занимает 85 минут, с дорогой 30 — 110.

Класс работы определяется по «Типу заявки BK». Срочная заявка всегда считается
аварией. Неизвестный тип — ошибка: молча считать его ремонтом нельзя.
"""

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "config" / "normatives.json"


def load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def on_site_minutes(norms: dict, work_class: str) -> int:
    """Время на адресе для класса: техническая работа плюс документы."""
    try:
        cls = norms["classes"][work_class]
    except KeyError as exc:
        raise ValueError(f"Нет норматива для класса работ «{work_class}»") from exc
    return int(cls["technical_minutes"] + cls["documents_minutes"])


def base_minutes(norms: dict, work_class: str) -> int:
    """Официальный базовый норматив: время на адресе плюс резерв на дорогу."""
    return on_site_minutes(norms, work_class) + int(norms["travel_reserve_minutes"])


def work_class(norms: dict, bk_type: str, urgent: bool = False) -> str:
    if urgent:
        return "emergency"
    try:
        return norms["class_by_bk_type"][bk_type]
    except KeyError as exc:
        raise ValueError(f"Тип заявки BK «{bk_type}» не сопоставлен ни с одним классом "
                         "нормативов (config/normatives.json, class_by_bk_type)") from exc


def duration(norms: dict, bk_type: str, urgent: bool = False) -> int:
    """Длительность заявки в минутах — только время на адресе, без дороги."""
    return on_site_minutes(norms, work_class(norms, bk_type, urgent))
