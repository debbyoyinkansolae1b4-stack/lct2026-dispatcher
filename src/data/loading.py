"""Чтение выгрузки «Обезличивание» и достройка всего, чего в ней нет.

Файлы: cp1251, разделитель `;`, по два на участок. В синтетическом файле последняя строка —
не заявка, а адрес офиса участка (подтверждено постановщиком [7:12]), он же стартовая точка.

Того, что ТЗ называет минимальными полями, в данных нет: ни координат, ни длительностей,
ни навыков, ни транспорта, ни смен, ни приоритетов. Всё это достраивается здесь по
config/assumptions.json и config/normatives.json — одним местом, чтобы допущения можно было
перечислить в README. Длительности — по официальной таблице нормативов, src/data/normatives.py.
"""

import json
import random
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.addresses import normalize
from src.data.geocode import load_cache
from src.data import normatives
from src.model import Engineer, Request
from src.plan.resources import request_equipment

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "obezlichivanie" / "Обезличивание"
CONFIG = Path(__file__).resolve().parents[2] / "config"

REGIONS = ["Восток", "Юго-восток", "Югоцентр"]


def _cfg(name: str) -> dict:
    return json.loads((CONFIG / f"{name}.json").read_text(encoding="utf-8"))


def _files(region: str) -> tuple[Path, Path | None]:
    """Файл заявок набора и контрольный файл (None — у загруженного набора его нет)."""
    from src.data import datasets
    root = datasets.get(datasets.root_of(region))
    if root.get("file"):
        return datasets.ROOT / root["file"], None
    synth = next(DATA.glob(f"{root['id']} Синтетические*.csv"))
    ctrl = next(DATA.glob(f"{root['id']} Контрольное*.csv"))
    return synth, ctrl


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="cp1251").dropna(how="all")


# Колонки формата организаторов и их близкие названия в загружаемых файлах (регистр не важен).
COLUMNS = {"Заявка": ("заявка", "номер заявки", "номер", "id"),
           "Тип заявки BK": ("тип заявки bk", "тип bk", "bk", "тип заявки", "тип"),
           "Тип заявки HD": ("тип заявки hd", "тип hd", "hd", "подтип"),
           "Начало": ("начало", "начало окна", "window_start", "from"),
           "Окончание": ("окончание", "конец окна", "конец", "window_end", "to"),
           "Район": ("район", "district"), "Адрес": ("адрес", "address"),
           "lat": ("lat", "latitude", "широта"), "lon": ("lon", "lng", "longitude", "долгота")}


def _read_any(path: Path) -> pd.DataFrame:
    """Загруженный файл: UTF-8 или cp1251, разделитель ; , или табуляция; колонки — к формату организаторов."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1251")
        except UnicodeDecodeError:
            raise ValueError("Файл не читается как текст (UTF-8 или cp1251) — нужна таблица CSV") from None
    head = text.splitlines()[0] if text else ""
    sep = max((";", ",", "\t"), key=head.count)
    import io
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str).dropna(how="all")
    rename = {}
    for column in df.columns:
        key = str(column).strip().casefold()
        for name, aliases in COLUMNS.items():
            if key == name.casefold() or key in aliases:
                rename[column] = name
                break
    df = df.rename(columns=rename)
    missing = [c for c in ("Заявка", "Тип заявки BK", "Начало", "Окончание", "Адрес") if c not in df.columns]
    if missing:
        raise ValueError(f"В файле нет колонок: {', '.join(missing)} (формат — как у организаторов)")
    return df


def _dt(value: str) -> datetime:
    text = str(value).strip()
    for fmt in ("%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"не разобрать время «{text}»")


def _is_request_row(row) -> bool:
    return bool(re.fullmatch(r"\d+", str(row["Заявка"]).strip()))


def _office_address(df: pd.DataFrame) -> str:
    """Строка офиса: в колонке «Заявка» текст, сам адрес уехал в следующую колонку."""
    for _, row in df.iterrows():
        if "офис" in str(row["Заявка"]).lower():
            return str(row["Тип заявки BK"]).strip()
    raise ValueError("В файле нет строки с адресом офиса")


def load_requests(region: str) -> tuple[list[Request], dict]:
    """Заявки набора + координаты офиса."""
    from src.data import datasets
    synth, _ = _files(region)
    root = datasets.get(datasets.root_of(region))
    if root.get("file"):
        out = read_requests_file(synth, office=root.get("office"))
        return out["requests"], {**out["depot"], "region": region}
    return _parse(_read(synth), region)


def read_requests_file(path: Path, office: str | None = None, geocode_missing: bool = False) -> dict:
    """Загруженный файл заявок: заявки, офис и отчёт (сколько принято, что пропущено и почему).

    Координаты — из колонок lat/lon, если они есть, иначе из кэша геокодера; geocode_missing —
    дозапросить недостающие адреса онлайн (при загрузке). Офис — строка «офис» в файле или office.
    """
    from src.data import geocode
    df = _read_any(path)
    if not any(_is_request_row(row) for _, row in df.iterrows()):
        raise ValueError("В файле нет строк заявок: номер заявки — целое число, как в файлах организаторов")
    cache = load_cache()
    report = {"rows": 0, "accepted": 0, "skipped": [], "from_file": 0, "from_cache": 0, "online": 0, "not_found": 0}
    if geocode_missing:
        for _, row in df.iterrows():
            if _is_request_row(row) and not _row_coords(row) and str(row["Адрес"]).strip() not in cache:
                geocode.geocode(str(row["Адрес"]).strip(), cache)
                report["online"] += 1
        geocode.save_cache(cache)
    requests, depot = _parse(df, "загружен", cache=cache, strict=False, report=report, office=office)
    # Номер заявки — ключ задачи (by_id, матрица): повтор молча затёр бы одну из строк (Н66).
    from collections import Counter
    repeated = sorted(rid for rid, k in Counter(r.id for r in requests).items() if k > 1)
    if repeated:
        raise ValueError(f"Номера заявок повторяются: {', '.join(repeated[:10])}"
                         f"{' и ещё ' + str(len(repeated) - 10) if len(repeated) > 10 else ''} — номер должен быть уникальным")
    return {"requests": requests, "depot": depot, "report": report}


def _row_coords(row):
    try:
        lat, lon = float(row.get("lat")), float(row.get("lon"))
    except (TypeError, ValueError):
        return None
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _parse(df: pd.DataFrame, region: str, cache: dict | None = None, strict: bool = True, report: dict | None = None,
           office: str | None = None) -> tuple[list[Request], dict]:
    """Строки выгрузки → заявки и офис. strict — формат организаторов: ошибка в строке — ошибка
    загрузки; иначе строка пропускается с причиной в report."""
    norms, asm = _cfg("normatives"), _cfg("assumptions")
    geo = load_cache() if cache is None else cache

    def coords(addr: str):
        hit = geo.get(addr)
        return (hit["lat"], hit["lon"]) if hit and hit.get("lat") is not None else (None, None)

    requests = []
    for number, (_, row) in enumerate(df.iterrows(), start=2):
        if not _is_request_row(row):
            continue
        if report is not None:
            report["rows"] += 1
        if not strict:
            try:
                bk = str(row["Тип заявки BK"]).strip()
                normatives.work_class(norms, bk, bk in asm["requests"]["urgent_bk_types"])
                _dt(row["Начало"]), _dt(row["Окончание"])
            except (ValueError, KeyError) as exc:
                report["skipped"].append({"row": number, "id": str(row["Заявка"]).strip(), "reason": str(exc)})
                continue
        status = str(row.get('Статус', row.get('Статус BK', ''))).strip()
        if status.casefold() in {'завершено','завершена','закрыто','закрыта','отменена','отменено'}:
            continue
        hd = str(row.get("Тип заявки HD", "")).strip()
        hd = "" if hd.casefold() == "nan" else hd
        bk = str(row["Тип заявки BK"]).strip()
        addr = str(row["Адрес"]).strip()
        own = None if strict else _row_coords(row)
        lat, lon = own if own else coords(addr)
        if report is not None:
            if own:
                report["from_file"] += 1
            elif lat is not None:
                report["from_cache"] += 1
            else:
                report["not_found"] += 1
            report["accepted"] += 1
        gigabit = str(row.get("Гигабитное подключение", "Нет")).strip().lower() == "да"

        requests.append(Request(
            id=str(row["Заявка"]).strip(),
            bk_type=bk,
            hd_type=hd,
            window_start=_dt(row["Начало"]),
            window_end=_dt(row["Окончание"]),
            district=re.sub(r"^GPON\s+", "", str(row.get("Район", "—")).strip()),   # технология в поле района
            address=addr,
            address_query=normalize(addr)["queries"][0] if normalize(addr)["queries"] else addr,
            lat=lat, lon=lon,
            skill=asm["skills"]["by_bk_type"].get(bk, "local"),
            duration_min=normatives.duration(norms, bk, bk in asm["requests"]["urgent_bk_types"]),
            status=status or None,
            equipment=request_equipment(bk,hd,asm["equipment"]),
            urgent=bk in asm["requests"]["urgent_bk_types"],
            required_transport="car" if hd in asm["requests"]["require_car_for_hd"] else None,
            gigabit=gigabit,
        ))

    try:
        office_addr = _office_address(df)
    except ValueError:
        if strict or not office:
            raise ValueError("В файле нет строки с адресом офиса — укажите адрес офиса при загрузке")
        office_addr = office
    point = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", office_addr)
    if point:
        o_lat, o_lon = float(point.group(1)), float(point.group(2))
    else:
        if not strict and office_addr not in geo:
            from src.data import geocode
            geocode.geocode(office_addr, geo)
            geocode.save_cache(geo)
        o_lat, o_lon = coords(office_addr)
    if not strict and o_lat is None:
        raise ValueError(f"Не найден на карте адрес офиса «{office_addr}» — укажите координаты «широта, долгота»")
    depot = {"address": office_addr, "lat": o_lat, "lon": o_lon, "region": region}
    return requests, depot


def load_control(region: str) -> pd.DataFrame:
    """Контрольное распределение: доступный состав участка на день — ориентир, не эталон [24:06]."""
    _, ctrl = _files(region)
    if ctrl is None:
        raise FileNotFoundError(f"У набора «{region}» нет контрольного дня")
    df = _read(ctrl)
    df = df[df.apply(_is_request_row, axis=1)].copy()
    df["Начало_dt"] = df["Начало"].map(_dt)
    df["Адрес_без_кв"] = df["Адрес"].map(lambda a: re.sub(r",?\s*кв\.?\s*\S+", "", str(a)).strip())
    return df


def generated_engineers(params: dict, day, depot: dict) -> list[Engineer]:
    """Бригады по параметрам набора (сколько всего, сколько с каждым навыком, сид) — без контрольного дня.

    Навыки раздаются по сиду ровно в заданных количествах, и у каждой бригады есть хотя бы один: сначала навык
    получают бригады, у которых навыков ещё нет, остальные — случайно среди прочих (Н67). Навыков меньше, чем
    бригад, — отказ в datasets.check_brigades.
    Транспорт — доли из допущений, как у встроенных наборов. Одинаковые параметры — одинаковые бригады.
    """
    asm = _cfg("assumptions")
    eng_cfg = asm["engineers"]
    rng = random.Random(params["seed"])
    count = params["count"]
    skills = [set() for _ in range(count)]
    for skill in ("emergency", "connect", "local"):
        empty = [i for i in range(count) if not skills[i]]
        rng.shuffle(empty)
        chosen = empty[:params[skill]]
        rest = [i for i in range(count) if i not in chosen]
        chosen += rng.sample(rest, params[skill] - len(chosen))
        for i in chosen:
            skills[i].add(skill)
    mix = eng_cfg["transport_mix"]
    modes = [m for m in mix if not m.startswith("_")]
    transports = rng.choices(modes, weights=[mix[m] for m in modes], k=count)

    def at(hhmm: str) -> datetime:
        h, m = map(int, hhmm.split(":"))
        return datetime(day.year, day.month, day.day, h, m)
    return [Engineer(id=f"eng{i + 1:02d}", name=f"Бригада {i + 1:02d}", skills=skills[i],
                     equipment=dict(asm["equipment"]["daily_stock"]), transport=transports[i],
                     shift_start=at(eng_cfg["shift_start"]), shift_end=at(eng_cfg["shift_end"]),
                     depot_lat=depot["lat"], depot_lon=depot["lon"]) for i in range(count)]


def build_engineers(region: str, depot: dict, requests: list | None = None) -> list[Engineer]:
    """Пул исполнителей: имена и навыки — из контрольного файла, транспорт и смена — допущение.
    У набора со сгенерированными бригадами — generated_engineers (день — по заявкам).

    Навык бригады выводим из того, какие типы работ она фактически делала за день.
    Сколько исполнителей задействовать — решает алгоритм, в этом одна из двух метрик.
    """
    from src.data import datasets
    brigades = datasets.get(region)["brigades"]
    if brigades["source"] == "generated":
        day = min(r.window_start for r in requests).date() if requests else datetime.now().date()
        return generated_engineers(brigades, day, depot)
    asm = _cfg("assumptions")
    eng_cfg = asm["engineers"]
    ctrl = load_control(region)
    day = ctrl["Начало_dt"].min().date()

    by_brigade: dict[str, set[str]] = {}
    for _, row in ctrl.iterrows():
        name = str(row["Бригада"]).strip()
        if not name or name.lower() == "nan":
            continue
        skill = asm["skills"]["by_bk_type"].get(str(row["Тип заявки BK"]).strip(), "local")
        by_brigade.setdefault(name, set()).add(skill)

    # Транспорт раздаём детерминированно: доли из конфига, порядок по имени — прогоны воспроизводимы
    names = sorted(by_brigade)
    rng = random.Random(eng_cfg["seed"])
    mix = eng_cfg["transport_mix"]
    modes = [m for m in mix if not m.startswith("_")]
    weights = [mix[m] for m in modes]
    transports = rng.choices(modes, weights=weights, k=len(names))

    def at(hhmm: str) -> datetime:
        h, m = map(int, hhmm.split(":"))
        return datetime(day.year, day.month, day.day, h, m)

    universal = eng_cfg.get("skills_mode") == "universal"
    engineers = []
    for i, name in enumerate(names):
        skills = {"local", "connect", "emergency"} if universal else by_brigade[name]
        # Аварийный навык в контроле встречается редко: кто делал аварии, тот их и умеет,
        # остальным его не приписываем — иначе ограничение по квалификации станет фиктивным.
        engineers.append(Engineer(
            id=f"eng{i + 1:02d}",
            name=name,
            skills=skills,
            equipment=dict(asm["equipment"]["daily_stock"]),
            transport=transports[i],
            shift_start=at(eng_cfg["shift_start"]),
            shift_end=at(eng_cfg["shift_end"]),
            depot_lat=depot["lat"], depot_lon=depot["lon"],
        ))
    return engineers


def load_region(region: str) -> dict:
    requests, depot = load_requests(region)
    engineers = build_engineers(region, depot, requests)
    return {"region": region, "requests": requests, "engineers": engineers, "depot": depot}
