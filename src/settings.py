"""Чтение и правка справочников из административного экрана.

Справочники живут в config/*.json, а админка меняет именно их, чтобы правка
нормативов или допущений не превращалась в правку кода. Поля с подчёркиванием
(комментарии) не отдаём и не даём затирать.
"""

import json
import math
from pathlib import Path

from src.data import normatives

CONFIG = Path(__file__).resolve().parents[1] / "config"

# Что администратору разрешено менять. Всё остальное — только через конфиг руками:
# смена справочника навыков или способа вывода навыков бригад меняет постановку,
# а не настройку, и должна оставаться осознанным решением.
EDITABLE = {
    # В нормативах меняются только минуты классов. Резерв дороги и соответствие типов BK
    # классам — часть официальной постановки, их правят в конфиге осознанно.
    "normatives": ["classes"],
    "assumptions": ["engineers.shift_start", "engineers.shift_end",
                    "engineers.transport_mix", "engineers.skills_mode",
                    "requests.require_car_for_hd", "requests.urgent_bk_types",
                    "travel.speed_kmh", "travel.fixed_overhead_min"],
}


# Заводские умолчания — копии настроек поставки: кнопка «сбросить» в админке работает и без git (сервер, Docker).
# Меняете умолчание осознанно — меняйте и config/, и config/defaults/ (тест test_defaults_match_committed_config).
DEFAULTS = CONFIG / "defaults"
RESETTABLE = {"goal": ("weights",), "search": ("search",), "norms": ("normatives", "assumptions")}


def changed() -> dict:
    """Какие вкладки отличаются от заводских умолчаний: {вкладка: True/False}."""
    differs = lambda name: json.loads((CONFIG / f"{name}.json").read_text(encoding="utf-8")) != \
        json.loads((DEFAULTS / f"{name}.json").read_text(encoding="utf-8"))
    return {tab: any(differs(n) for n in names) for tab, names in RESETTABLE.items()}


def reset(scope: str) -> list[str]:
    """Вернуть заводские умолчания вкладки (goal / search / norms) или всех (all). Возвращает сброшенные файлы."""
    if scope != "all" and scope not in RESETTABLE:
        raise ValueError("Сбрасывается вкладка «Цель», «Поиск», «Нормативы и допущения» или всё")
    names = [n for tab, group in RESETTABLE.items() if scope in ("all", tab) for n in group]
    for name in names:
        (CONFIG / f"{name}.json").write_text((DEFAULTS / f"{name}.json").read_text(encoding="utf-8"), encoding="utf-8")
    from src.plan import solver
    solver.load_weights()
    return names


def _load(name: str) -> dict:
    return json.loads((CONFIG / f"{name}.json").read_text(encoding="utf-8"))


def _save(name: str, data: dict) -> None:
    (CONFIG / f"{name}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_path(data: dict, path: str):
    node = data
    for part in path.split("."):
        node = node[part]
    return node


def _set_path(data: dict, path: str, value) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def read() -> dict:
    """Текущие справочники + пояснения, которые видит администратор."""
    norms, asm = _load("normatives"), _load("assumptions")
    return {
        "normatives": {
            "travel_reserve_minutes": norms["travel_reserve_minutes"],
            "classes": {key: {**cls,
                              "on_site_minutes": normatives.on_site_minutes(norms, key),
                              "base_minutes": normatives.base_minutes(norms, key)}
                        for key, cls in norms["classes"].items()},
            "class_by_bk_type": norms["class_by_bk_type"],
            "note": norms.get("_comment", ""),
        },
        "assumptions": {
            "skills_mode": asm["engineers"].get("skills_mode", "history"),
            "skills_mode_comment": asm["engineers"].get("skills_mode_comment", ""),
            "shift_start": asm["engineers"]["shift_start"],
            "shift_end": asm["engineers"]["shift_end"],
            "transport_mix": {k: v for k, v in asm["engineers"]["transport_mix"].items()
                              if not k.startswith("_")},
            "require_car_for_hd": asm["requests"]["require_car_for_hd"],
            "urgent_bk_types": asm["requests"]["urgent_bk_types"],
            "speed_kmh": asm["travel"]["speed_kmh"],
            "fixed_overhead_min": asm["travel"]["fixed_overhead_min"],
            "note": asm["requests"].get("_comment", ""),
        },
    }


CLASS_PARTS = ("technical_minutes", "documents_minutes")


def _class_problem(current: dict | None, parts) -> str | None:
    """Почему правку класса нельзя принять. Класс принимается или отклоняется целиком.

    Отдельная составляющая может быть нулём (у аварии нет документов), но время
    на адресе в сумме должно быть положительным: заявка нулевой длительности
    не создаётся. Минуты целые — так их хранит заявка.
    """
    if current is None:
        return "неизвестный класс"
    if not isinstance(parts, dict) or not parts:
        return "ожидается словарь составляющих"
    unknown = sorted(set(parts) - set(CLASS_PARTS))
    if unknown:
        return f"неизвестные поля {unknown}"
    for part, minutes in parts.items():
        if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) \
                or not math.isfinite(minutes) or minutes != int(minutes) or not 0 <= minutes <= 480:
            return f"{part}: нужно целое число минут от 0 до 480"
    merged = {part: parts.get(part, current[part]) for part in CLASS_PARTS}
    if sum(merged.values()) <= 0:
        return "время на адресе должно быть больше нуля"
    return None


def update(changes: dict) -> dict:
    """changes: {"normatives": {...}, "assumptions": {...}} — только разрешённые поля."""
    applied, rejected = [], []

    norms = _load("normatives")
    for key, value in (changes.get("normatives") or {}).items():
        if key == "classes" and isinstance(value, dict):
            for name, parts in value.items():
                problem = _class_problem(norms["classes"].get(name), parts)
                if problem:
                    rejected.append(f"normatives.classes.{name}: {problem}")
                    continue
                for part, minutes in parts.items():
                    norms["classes"][name][part] = int(minutes)
                    applied.append(f"normatives.classes.{name}.{part}")
        else:
            rejected.append(f"normatives.{key}")
    _save("normatives", norms)

    asm = _load("assumptions")
    alias = {"shift_start": "engineers.shift_start", "shift_end": "engineers.shift_end",
             "transport_mix": "engineers.transport_mix", "skills_mode": "engineers.skills_mode",
             "require_car_for_hd": "requests.require_car_for_hd",
             "urgent_bk_types": "requests.urgent_bk_types",
             "speed_kmh": "travel.speed_kmh", "fixed_overhead_min": "travel.fixed_overhead_min"}
    for key, value in (changes.get("assumptions") or {}).items():
        path = alias.get(key)
        if path and path in EDITABLE["assumptions"]:
            _set_path(asm, path, value)
            applied.append(f"assumptions.{path}")
        else:
            rejected.append(f"assumptions.{key}")
    _save("assumptions", asm)

    return {"applied": applied, "rejected": rejected}


WEIGHTS_FILE = CONFIG / "weights.json"


def read_weights() -> dict:
    """Приоритеты планирования плюс проверка, что иерархия не сломана."""
    cfg = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
    values = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return {"values": values, "checks": check_weights(values),
            "note": cfg.get("_comment", "")}


def check_weights(v: dict) -> list[dict]:
    """Проверки смысла, а не формата. Каждая объясняет, чем грозит нарушение."""
    out = []
    req, eng = float(v["unassigned_request"]), float(v["engineer_used"])
    open_pen = float(v.get("open_engineer_penalty", eng))
    delay = float(v["urgent_delay_minute"])
    square = v.get("urgent_delay_mode", "linear") == "square"
    ten_hours = float(v.get("urgent_delay_square", 0.)) * 600 ** 2 if square else delay * 600

    out.append({
        "ok": req > eng * 5,
        "text": "Заявка должна быть дороже исполнителя с запасом",
        "why": (f"сейчас заявка {req:.0f}, исполнитель {eng:.0f} — "
                + ("порядок верный" if req > eng * 5 else
                   "планировщик начнёт бросать заявки ради экономии людей")),
    })
    out.append({
        "ok": abs(open_pen - eng) < 1e-6,
        "text": "Штраф за открытие исполнителя равен его цене",
        "why": ("совпадают" if abs(open_pen - eng) < 1e-6 else
                f"открытие стоит {open_pen:.0f}, а сам исполнитель {eng:.0f}: поиск будет "
                f"распускать маршрут и тут же открывать заново"),
    })
    out.append({
        "ok": eng >= 50,
        "text": "Исполнитель дороже обычного дневного пробега",
        "why": (f"маршрут за день — это десятки километров; при цене {eng:.0f} "
                + ("экономия людей перевешивает" if eng >= 50 else
                   "планировщик предпочтёт лишнего человека нескольким километрам")),
    })
    out.append({
        "ok": 0 < ten_hours < eng,
        "text": "Задержка аварии влияет, но не перевешивает исполнителя",
        "why": (f"десять часов задержки одной аварии стоят {ten_hours:.0f} км"
                + (" (квадратично)" if square else " (линейно)")
                + (" — соревнуется с пробегом, как и задумано" if 0 < ten_hours < eng else
                   " — срочность начнёт спорить с числом исполнителей")),
    })
    return out


def update_weights(values: dict) -> dict:
    """Сохраняет приоритеты и перечитывает их в решателе."""
    cfg = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
    applied = []
    for key in ("unassigned_request", "engineer_used", "urgent_delay_minute",
                "open_engineer_penalty", "urgent_delay_square", "urgent_target_minutes", "event_move_km"):
        if key in values:
            value = values[key]
            # До записи: вес — конечное неотрицательное число, иначе файл и настройки не меняются (Н60).
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or value < 0:
                raise ValueError(f"Вес «{key}» — конечное неотрицательное число, получено {value!r}")
            cfg[key] = float(value)
            applied.append(key)
    if "urgent_delay_mode" in values:
        if values["urgent_delay_mode"] not in ("linear", "square"):
            raise ValueError("Режим задержки аварий — linear или square")
        cfg["urgent_delay_mode"] = values["urgent_delay_mode"]
        applied.append("urgent_delay_mode")
    if "urgent_target_first" in values:
        if not isinstance(values["urgent_target_first"], bool):
            raise ValueError("Аварии сверх ориентира — да или нет")
        cfg["urgent_target_first"] = values["urgent_target_first"]
        applied.append("urgent_target_first")
    WEIGHTS_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    from src.plan import solver
    solver.load_weights()
    return {"applied": applied, "checks": check_weights(cfg)}


def traffic_key() -> str:
    return _load("assumptions")["travel"].get("traffic", {}).get("api_key", "")


def traffic_provider() -> str:
    return _load("assumptions")["travel"].get("traffic", {}).get("provider", "не выбран")


# --------------------------------------------------------------------- поиск (шаг 6в)
# refactor: Claude, 22.09.2026 — настройки портфеля поиска для приложения
# перенос: Claude, 26.09.2026 — два режима одной машины: learned — сеть b, uniform — потомок U2n (PORT-PLAN.md)

SEARCH_MODES = ("uniform", "learned")
SEARCH_STOPS = ("time", "patience")


# Поля участника портфеля: режим, сид, сеть (learned) и параметры поиска у каждого свои (Михаил, 23.09).
MEMBER_FIELDS = ("mode", "seed", "hidden", "experts", "lr", "population", "chain_max", "schedule_epochs", "repair_noise",
                 "chain_learned", "rehomable_feature", "normalize_features", "dedup", "insertion", "target_potential",
                 "open_cost")


def check_search(cfg: dict) -> list[str]:
    """Что не так с настройками поиска. Пустой список — можно сохранять.

    Проверки те же, что у портфеля и поиска до запуска: режимы, целые сиды, параметры
    сети, участников не больше процессов, процессов не больше ядер.
    """
    import os
    integer = lambda v: isinstance(v, int) and not isinstance(v, bool)
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    problems = []
    workers = cfg.get("workers")
    if not integer(workers) or not 1 <= workers <= (os.cpu_count() or 1):
        problems.append(f"число процессов — целое от 1 до {os.cpu_count() or 1} (ядер на машине)")
    if not isinstance(cfg.get("urgent_front", False), bool):
        problems.append("постобработка «аварии вперёд» — да или нет")
    plan, event = cfg.get("plan"), cfg.get("event")
    if not isinstance(plan, dict) or not isinstance(event, dict):     # Н42: форма — до обращений
        return problems + ["план и событие — разделы настроек"]
    if plan.get("stop") not in SEARCH_STOPS:
        problems.append("остановка расчёта плана — «до времени» или «после N эпох без улучшения»")
    if not integer(plan.get("patience")) or plan["patience"] < 1:
        problems.append("N эпох без улучшения — целое ≥ 1")
    if not number(event.get("seconds")) or not 1 <= event["seconds"] <= 60:
        problems.append("время перепланирования — от 1 до 60 секунд")
    for name, block in (("план", plan), ("событие", event)):
        members = block.get("members")
        if not isinstance(members, list) or not members:
            problems.append(f"{name}: нужен хотя бы один поиск")
            continue
        if integer(workers) and len(members) > workers:
            problems.append(f"{name}: поисков {len(members)} больше, чем процессов {workers}")
        for i, m in enumerate(members, 1):
            if not isinstance(m, dict):                    # Н42: не участник — отказ, а не падение
                problems.append(f"{name}, поиск {i}: ожидается описание поиска")
                continue
            unknown = set(m) - set(MEMBER_FIELDS)
            if unknown:
                problems.append(f"{name}, поиск {i}: неизвестные поля {sorted(unknown)}")
            if m.get("mode") not in SEARCH_MODES:
                problems.append(f"{name}, поиск {i}: режим — «равномерный» или «сеть»")
            if not integer(m.get("seed")):
                problems.append(f"{name}, поиск {i}: сид — целое")
            # Параметры сети проверяются у любого режима, если заданы: сохранённое должно
            # запускаться (Н43); у равномерного они не используются.
            if not integer(m.get("hidden", 256)) or not 8 <= m.get("hidden", 256) <= 1024:
                problems.append(f"{name}, поиск {i}: ширина сети — целое от 8 до 1024")
            if not integer(m.get("experts", 3)) or not 1 <= m.get("experts", 3) <= 8:
                problems.append(f"{name}, поиск {i}: экспертов — целое от 1 до 8")
            if not number(m.get("lr", .001)) or not 0 < m.get("lr", .001) <= .1:
                problems.append(f"{name}, поиск {i}: шаг обучения — от 0 до 0.1")
            if not integer(m.get("population", 64)) or not 4 <= m.get("population", 64) <= 256:
                problems.append(f"{name}, поиск {i}: кандидатов за эпоху — целое от 4 до 256")
            if not integer(m.get("chain_max", 3)) or not 1 <= m.get("chain_max", 3) <= 4:
                problems.append(f"{name}, поиск {i}: звеньев цепочки — целое от 1 до 4")
            if not integer(m.get("schedule_epochs", 500)) or not 20 <= m.get("schedule_epochs", 500) <= 10000:
                problems.append(f"{name}, поиск {i}: горизонт расписания — целое от 20 до 10000 эпох")
            for flag in ("chain_learned", "rehomable_feature", "normalize_features"):
                if not isinstance(m.get(flag, True), bool):
                    problems.append(f"{name}, поиск {i}: {flag} — да или нет")
            if m.get("dedup") is not None and not isinstance(m["dedup"], bool):
                problems.append(f"{name}, поиск {i}: без повторов — да, нет или по режиму")
            if m.get("insertion") not in (None, "six", "product"):
                problems.append(f"{name}, поиск {i}: вставка — сети (six), продуктовая (product) или по режиму")
            share = m.get("target_potential")
            if share is not None and (not number(share) or not 0 <= share <= 1):
                problems.append(f"{name}, поиск {i}: прицел по потенциалу — доля от 0 до 1 или по режиму")
            cost = m.get("open_cost")
            if cost is not None and (not number(cost) or not 0 <= cost <= 100000):
                problems.append(f"{name}, поиск {i}: цена открытия бригады — от 0 до 100000 км или по режиму")
            noise = m.get("repair_noise")
            if noise is not None and (not number(noise) or not 0 <= noise <= .5):
                problems.append(f"{name}, поиск {i}: шум вставки — от 0 до 0.5 или пусто")
    if not problems:
        # Последняя проверка — та же, что у поиска перед запуском: сохранённое обязано запускаться.
        from src.search.portfolio import Member, search_config
        for name, block in (("план", plan), ("событие", event)):
            for i, m in enumerate(block["members"], 1):
                try:
                    search_config(Member(**m), 1.)
                except (TypeError, ValueError) as exc:
                    problems.append(f"{name}, поиск {i}: {exc}")
    return problems


def read_search() -> dict:
    cfg = _load("search")
    return {k: v for k, v in cfg.items() if not k.startswith("_")} | {"note": cfg.get("_comment", ""),
                                                                      "cores": __import__("os").cpu_count()}


def update_search(values: dict) -> dict:
    """Сохранить настройки поиска целиком после проверки. Действуют со следующего расчёта."""
    cfg = _load("search")
    new = {**cfg, **{k: values[k] for k in ("workers", "plan", "event", "urgent_front") if k in values}}
    problems = check_search(new)
    if problems:
        raise ValueError("; ".join(problems))
    _save("search", new)
    return read_search()


def search_members(kind: str, snapshot: dict | None = None):
    """Участники портфеля для расчёта плана (kind='plan') или перепланирования ('event').

    snapshot — уже прочитанные настройки (read_search): все поля запроса берутся из одного
    снимка, а не из нескольких чтений файла (Н46)."""
    from src.search.portfolio import Member
    fields = MEMBER_FIELDS
    cfg = snapshot if snapshot is not None else _load("search")
    return [Member(**{k: m[k] for k in fields if k in m}) for m in cfg[kind]["members"]]
