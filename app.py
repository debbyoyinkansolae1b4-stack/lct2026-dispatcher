"""Веб-прототип помощника диспетчера.

    .venv/bin/uvicorn app:api --port 8000     → http://127.0.0.1:8000

Состояние держим в памяти процесса: пользователь один — диспетчер [42:50], и прототипу
база не нужна. Планы по участкам считаются лениво и кэшируются до следующего пересчёта.

Поиск — портфель популяционного поиска (src/search/portfolio.py, шаг 7 REFACTOR-PLAN.md):
рабочие процессы поднимаются и прогреваются при старте, состав участников — из настроек
(config/search.json, админка). Расчёты идут по очереди — пул один на приложение.
"""
# refactor: Claude, 22.09.2026 — шаг 7: приложение на портфеле популяционного поиска вместо ALNS
# refactor: Claude, 22.09.2026 — Н45–Н47: замок расчёта, один снимок настроек, прогрев поколений
# refactor: Claude, 22.09.2026 — упрощение: единственный замок расчёта — COMPUTE; портфель последовательный

import sys
import threading
import time
from contextlib import asynccontextmanager
from uuid import uuid4
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, StrictBool, StrictFloat, StrictInt

sys.path.insert(0, str(Path(__file__).parent))

from src import settings

from src.data import traffic_source

from src.plan import events, solver, validate

from src.report import analysis, bounds, explain, metrics
from src.plan.manual_assignment import assign
from src.data.geometry import route_line
from src.instance import build
from src.data import datasets
from src.data.loading import REGIONS
from src.search.portfolio import Member, Portfolio, Weights
from src.search.report import OPERATOR_TITLES, search_report
from src.plan.solver import baseline_greedy, construct

WEB = Path(__file__).parent / "web"
DOCS = Path(__file__).parent / "docs"        # сопроводительные материалы: документация и схема движка
STATE: dict[str, dict] = {}       # region → {inst, plan, baseline, report, events}
PORTFOLIO: Portfolio | None = None
# Весь расчёт приложения — одна транзакция под этим замком: снимок настроек и весов, пул,
# поиск, сохранение результата. Диспетчер один; параллельные запросы ждут очереди (Н45).
COMPUTE = threading.RLock()
# Перепланирование после события — без шума вставки: прежнее поведение (FINDINGS Н26),
# задаётся явно, а не умолчанием ядра.
EVENT_REPAIR_NOISE = 0.


def portfolio(snapshot: dict | None = None) -> Portfolio:
    """Пул рабочих процессов под число процессов из снимка настроек.

    Число поменяли в админке — старый пул закрывается только после текущего расчёта
    (его замок), новый прогревается перед первым расчётом (Н45, Н47).
    """
    global PORTFOLIO
    with COMPUTE:
        workers = (snapshot or settings.read_search())["workers"]
        if PORTFOLIO is not None and PORTFOLIO.workers != workers:
            # Под COMPUTE: ни один расчёт сейчас не идёт — пул закрывается между расчётами.
            PORTFOLIO.close()
            PORTFOLIO = None
        if PORTFOLIO is None:
            PORTFOLIO = Portfolio(workers=workers)
            PORTFOLIO.set_warmup(*_warmup_task())
        return PORTFOLIO


def _warmup_task():
    """Задача прогрева — самый маленький участок: короткий поиск learned в каждом процессе."""
    inst = build(REGIONS[-1])
    return inst, construct(inst)


def warm_up():
    """Прогрев при старте (Н15): каждый процесс выполнил эпоху поиска до первого запроса —
    в режимах, которые есть в настройках. Не хватает памяти — приложение стартует, а расчёт
    откажет с понятной причиной; прогрев повторится перед первым расчётом."""
    with COMPUTE:
        search = settings.read_search()
        modes = {m.mode for kind in ("plan", "event") for m in settings.search_members(kind, search)}
        pool = portfolio(search)
        try:
            warning = pool.check_memory(pool._warmup[0], pool.warm_members(modes))
            if warning:
                print(f"Прогрев: {warning}", file=sys.stderr)
            pool.prepare(modes)
        except (ValueError, RuntimeError) as exc:
            print(f"Прогрев отложен: {exc}", file=sys.stderr)


def _snapshot():
    """Один снимок настроек поиска и весов критерия на запрос (Н46). Веса выставляются и
    в координаторе: стартовый план и проверка считаются по тем же весам, что и поиск."""
    weights = Weights.read()
    weights.apply()
    return settings.read_search(), weights


@asynccontextmanager
async def lifespan(app):
    warm_up()
    yield
    if PORTFOLIO is not None:
        PORTFOLIO.close()


api = FastAPI(title="Планировщик выездных инженеров", lifespan=lifespan)


def _state(region: str) -> dict:
    if region not in STATE:
        if region not in datasets.ids():
            raise HTTPException(status_code=404, detail=f"Нет набора «{region}»")
        inst = build(region)
        base = baseline_greedy(inst)
        # origin — задача дня до событий: «Построить план» считает день заново по ней (Н63).
        STATE[region] = {"origin": inst, "inst": inst, "plan": None, "baseline": base, "report": None, "events": []}
    return STATE[region]


GEOMETRY_BUDGET = 3.0      # с на линии маршрутов одного ответа: дальше — прямые, план не ждёт сеть


def _route_with_line(plan, inst, engineer_id: str, deadline: float | None = None) -> dict:
    """Маршрут плюс линия проезда по дорогам. Нет геометрии — карта рисует прямые."""
    summary = explain.route_summary(plan, inst, engineer_id)
    points = [(inst.depot["lat"], inst.depot["lon"])] + [(s["lat"], s["lon"]) for s in summary["stops"]]
    summary["line"] = route_line(points, deadline)
    return summary


def _plan_payload(region: str) -> dict:
    st = _state(region)
    return _payload(region, st["inst"], st["plan"], st["baseline"], st["events"], "plan")


def _payload(region, inst, plan, base, events_log, mode) -> dict:
    """Ответ интерфейсу по плану: рабочему или сверочному (у сверки свои задача, план и базовый вариант)."""
    deadline = time.monotonic() + GEOMETRY_BUDGET
    summary = metrics.summarize(plan, inst, "Текущий план")
    base_sum = metrics.summarize(base, inst, "Базовый вариант ТЗ")
    return {
        "region": region,
        "available_engineers": [{"id": e.id, "name": e.name} for e in inst.engineers if e.available],
        "depot": {"lat": inst.depot["lat"], "lon": inst.depot["lon"],
                  "address": inst.depot["address"]},
        "summary": summary,
        "baseline": base_sum,
        "comparison": metrics.compare(base_sum, summary),
        "control": metrics.control_reference(inst),
        "advice": metrics.capacity_advice(plan, inst),
        "routes": [_route_with_line(plan, inst, eid, deadline)
                   for eid, r in plan.routes.items() if r.used],
        "idle": [e.name for e in inst.engineers if not plan.routes[e.id].used],
        "unassigned": [{**u.__dict__,
                        "district": inst.by_id[u.request_id].district,
                        "hd_type": inst.by_id[u.request_id].hd_type,
                        "address": inst.by_id[u.request_id].address,
                        "lat": inst.by_id[u.request_id].lat,
                        "lon": inst.by_id[u.request_id].lon}
                       for u in plan.unassigned if u.request_id in inst.by_id],
        "validation": validate.validate(plan, inst),
        "meta": plan.meta,
        "events": events_log,
        "travel_source": inst.matrix.source,
        "mode": mode,
    }


class PlanRequest(BaseModel):
    region: str = "Восток"
    seconds: float = 20.0
    mode: str | None = None       # converge — до N эпох без улучшения, time — весь бюджет; None — из настроек
    ortools: bool = False         # сверка с OR-Tools: общие упрощённые рамки, три плана с одним бюджетом (шаг 7)


class EventRequest(BaseModel):
    region: str
    kind: str                      # urgent | cancel | engineer_off
    at: str = "13:00"
    request_id: str | None = None
    engineer_id: str | None = None
    lat: float | None = None
    lon: float | None = None
    address: str | None = None     # альтернатива точке на карте
    district: str | None = None
    hd_type: str = "Авария"
    duration_min: int | None = None  # по умолчанию — норматив аварии на адресе
    reserve: bool = False            # эскалация: можно вызвать бригаду, не вышедшую утром (message687)


class GeocodeRequest(BaseModel):
    address: str


@api.post("/api/geocode")
def geocode_address(req: GeocodeRequest):
    """Адрес в координаты — для срочной заявки, которую диспетчеру диктуют голосом."""
    from src.data.geocode import geocode, load_cache, save_cache
    from src.data.geoquality import classify

    cache = load_cache()
    hit = geocode(req.address, cache)
    save_cache(cache)
    if not hit:
        return {"ok": False, "error": "адрес не найден, попробуйте уточнить улицу и дом "
                                      "или укажите точку на карте"}
    quality = classify(req.address, hit)
    return {"ok": True, "lat": hit["lat"], "lon": hit["lon"],
            "matched": hit.get("matched", ""),
            "quality": quality,
            "warning": "" if quality == "house" else
                       "координата определена только до улицы — проверьте точку на карте"}


@api.get("/api/regions")
def regions():
    """Наборы для выбора на главной: встроенные участки, затем добавленные (реестр config/datasets.json)."""
    items = datasets.entries()
    return {"regions": [e["id"] for e in items], "titles": {e["id"]: e["title"] for e in items},
            "loaded": list(STATE)}


# --------------------------------------------------------------------- наборы данных (вкладка «Данные»)

class GenerateRequest(BaseModel):
    base: str
    count: StrictInt
    emergency: StrictInt
    connect: StrictInt
    local: StrictInt
    seed: StrictInt
    title: str | None = None


class UploadRequest(BaseModel):
    filename: str
    content: str                    # содержимое файла в base64: без зависимости от multipart
    office: str | None = None       # адрес офиса или «широта, долгота», если в файле нет строки офиса
    brigades: dict | None = None    # count / emergency / connect / local / seed; по умолчанию — 12 бригад, 4 аварийных
    title: str | None = None        # название набора; по умолчанию — имя файла


@api.get("/api/datasets")
def list_datasets():
    return {"datasets": datasets.entries()}


@api.post("/api/datasets/generate")
def generate_dataset(req: GenerateRequest):
    """Вариант набора со сгенерированными бригадами; базовый набор не меняется."""
    with COMPUTE:
        try:
            entry = datasets.generate(req.base, req.count, req.emergency, req.connect, req.local, req.seed, req.title)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc).strip("'\"")) from exc
    return {"dataset": entry}


SAMPLE_CSV = Path(__file__).parent / "data" / "samples" / "пример_заявок.csv"


@api.get("/api/datasets/sample")
def dataset_sample():
    """Пример файла заявок в формате организаторов (scripts/build_sample_csv.py): скачать или взять в форму «Новый набор»."""
    return FileResponse(SAMPLE_CSV, media_type="text/csv; charset=utf-8", filename=SAMPLE_CSV.name)


@api.post("/api/datasets/upload")
def upload_dataset(req: UploadRequest):
    """Загрузка CSV заявок (формат организаторов или близкий); отчёт — что принято и что пропущено."""
    with COMPUTE:
        try:
            entry, report = datasets.upload(req.filename, req.content, req.office, req.brigades, req.title)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:                     # binascii, pandas: файл не читается как CSV
            raise HTTPException(status_code=422, detail=f"Файл не прочитан: {type(exc).__name__}: {exc}") from exc
    return {"dataset": entry, "report": report}


@api.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    with COMPUTE:
        try:
            datasets.remove(dataset_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc).strip("'\"")) from exc
        STATE.pop(dataset_id, None)
    return {"ok": True}


@api.get("/api/dataset/{region}")
def dataset(region: str):
    """Что у диспетчера на день — до всякого планирования.

    ТЗ задаёт первым шагом демонстрации «загрузите или откройте тестовый набор»
    и только вторым — запуск планирования. Значит заявки должны быть видны сразу.
    """
    from collections import Counter

    inst = _state(region)["inst"]
    reqs = inst.requests
    by_skill = Counter(r.skill for r in reqs)
    titles = {"local": "Локальные работы", "connect": "Подключения и дозаказы",
              "emergency": "Аварийные работы"}
    windows = Counter(f"{r.window_start:%H:%M}" for r in reqs)
    return {
        "region": region,
        "depot": {"lat": inst.depot["lat"], "lon": inst.depot["lon"],
                  "address": inst.depot["address"]},
        "day": f"{reqs[0].window_start:%d.%m.%Y}" if reqs else "",
        "counts": {"requests": len(reqs), "engineers": len(inst.engineers),
                   "urgent": sum(1 for r in reqs if r.urgent),
                   "districts": len({r.district for r in reqs}),
                   "work_hours": round(sum(r.duration_min for r in reqs) / 60, 1)},
        "by_skill": [{"skill": titles[k], "count": v} for k, v in by_skill.most_common()],
        "windows": [{"start": k, "count": v} for k, v in sorted(windows.items())],
        "travel_source": inst.matrix.source,
        "time_source": inst.matrix.time_source,
        "requests": [{"id": r.id, "lat": r.lat, "lon": r.lon, "district": r.district,
                      "hd_type": r.hd_type, "urgent": r.urgent,
                      "window": f"{r.window_start:%H:%M}–{r.window_end:%H:%M}",
                      "duration": r.duration_min, "skill": titles[r.skill]}
                     for r in reqs],
        "engineers": [{"id": e.id, "name": e.name, "transport": e.transport,
                       "skills": sorted(titles[s] for s in e.skills),
                       "shift": f"{e.shift_start:%H:%M}–{e.shift_end:%H:%M}"}
                      for e in inst.engineers],
    }


@api.post("/api/plan")
def make_plan(req: PlanRequest):
    """План дня портфелем поиска. Состав участников и N эпох без улучшения — из настроек."""
    if req.mode not in (None, "converge", "time"):
        raise HTTPException(status_code=422, detail="Режим расчёта — converge или time")
    arrived = time.perf_counter()
    with COMPUTE:
        queued = time.perf_counter() - arrived
        st = _state(req.region)
        search, weights = _snapshot()
        # Режим не передан — из настроек. «До времени» — весь бюджет; «до сходимости» —
        # N эпох без рекорда или время, что раньше.
        mode = req.mode or ("time" if search["plan"]["stop"] == "time" else "converge")
        patience = None if mode == "time" else search["plan"]["patience"]
        # План дня — с нуля по исходной задаче: события прошлого плана (выбывший инженер, поступившие
        # аварии) в новый день не переносятся, как и список событий (Н63).
        if req.ortools:
            return _ortools_check(req, st, search, weights, patience, queued)
        inst = st["origin"]
        setup = _plan_setup(search, req.seconds, patience)
        try:
            result = portfolio(search).run(inst, construct(inst), list(setup["members"]),
                                           seconds=req.seconds, patience=patience, weights=weights,
                                           urgent_front=setup["urgent_front"])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        report = search_report(result, patience)
        report["queued"] = round(queued, 1)      # очередь за другими расчётами приложения
        plan = result.best
        plan.meta = {"seconds": report["seconds"], "iterations": report["iterations"], "engine": "population"}
        st["inst"], st["plan"], st["report"], st["events"] = inst, plan, report, []
        st["setup"] = setup                      # с чем построен план: «что если» в полном режиме сравнивает с ним
        payload = _plan_payload(req.region)
        payload["search"] = report
        # Границы — в той же транзакции: задача участка не может смениться между планом и оценкой (Н48).
        payload["bounds"] = bounds.gap_report(plan, st["inst"])
    return payload


# refactor: Claude, 26.09.2026 — шаг 7 PORT-PLAN: сверка с OR-Tools в общих рамках (замер — RESULTS «в общих рамках»)
ORTOOLS_FRAME = ("без пробок (свободная дорога OSRM); цель — аварии → подключения → заявки → бригады → км, без «аварий "
                 "сверх 2 ч» и цены ожидания (их OR-Tools не видит); исходный день без событий; те же окна, смены, "
                 "навыки, транспорт и запасы")


def _ortools_frame(origin, weights: Weights):
    """Копия дня в рамках, общих с OR-Tools (src/reference/ortools_ref.py): без пробок и с целью без ожидания аварий."""
    import copy
    from dataclasses import replace
    from src.traffic import TrafficModel
    inst = copy.deepcopy(origin)
    inst.matrix.traffic = TrafficModel(None)
    inst.matrix._mode_cache.clear()
    return inst, replace(weights, urgent_target_first=False, urgent_delay_mode="linear", urgent_delay_minute=0.)


def _ortools_check(req, st, search, weights, patience, queued):
    """Три плана одной задачи в общих рамках с одним бюджетом: базовый вариант ТЗ, OR-Tools (портфель стратегий на
    стольких же процессах) и наш поиск. Каждый план проверяет один и тот же валидатор; показывается наш.
    Рабочий план дня и его события остаются как были."""
    try:
        from src.reference import ortools_ref
    except ImportError as exc:
        raise HTTPException(status_code=422, detail="OR-Tools не установлен: pip install ortools (requirements.txt)") from exc
    inst, frame = _ortools_frame(st["origin"], weights)
    setup = {**_plan_setup(search, req.seconds, patience), "ortools": True}
    members = list(setup["members"])
    frame.apply()
    try:
        greedy = baseline_greedy(inst)
        result = portfolio(search).run(inst, construct(inst), members, seconds=req.seconds, patience=patience,
                                       weights=frame, urgent_front=setup["urgent_front"])
        report = search_report(result, patience)
        report["queued"] = round(queued, 1)
        frame.apply()
        # OR-Tools — на стольких же процессах, сколько задействовал наш поиск: его решатель однопоточный, поэтому
        # разные стратегии параллельно, лучший план (src/ortools_ref.solve_portfolio).
        cores = min(len(members), search["workers"])
        ref = ortools_ref.solve_portfolio(inst, req.seconds, cores, weights=frame)
        rows = [_check_row("Базовый ТЗ (жадный)", greedy, inst, None, 1),
                _check_row("OR-Tools", ref["plan"], inst, ref["seconds"], ref["processes"]) if ref["ok"]
                else {"name": "OR-Tools", "error": ref["error"]},
                _check_row("Наш поиск", result.best, inst, report["seconds"], cores)]
        ranked = [r for r in rows if "cost" in r]
        best = min(r["cost"] for r in ranked)
        for r in ranked:
            r["best"] = r["cost"] == best
        plan = result.best
        plan.meta = {"seconds": report["seconds"], "iterations": report["iterations"], "engine": "population"}
        # Рабочий день (план, события, отчёт) не трогаем: сверка хранится отдельно и уходит, когда снимают галочку.
        st["check"] = {"inst": inst, "plan": plan}
        payload = _payload(req.region, inst, plan, greedy, [], "ortools")
        payload["search"] = report
        payload["bounds"] = bounds.gap_report(plan, inst)
        payload["ortools_check"] = {"frame": ORTOOLS_FRAME, "seconds": req.seconds, "rows": rows}
        return payload
    finally:
        weights.apply()                  # цель приложения — обратно: прочие запросы считают по настройкам


def _check_row(name, plan, inst, seconds, cores):
    """Строка сверки: строки общей цели, время, ядра и независимая проверка плана."""
    check = validate.validate(plan, inst)
    return {"name": name, "urgent": plan.priority_counts[0], "connect": plan.priority_counts[1],
            "assigned": plan.assigned_count, "total": len(inst.requests), "engineers": plan.used_engineers,
            "km": round(plan.total_km, 1), "seconds": None if seconds is None else round(seconds, 1), "cores": cores,
            "valid": check["ok"], "cost": [-plan.priority_counts[0], -plan.priority_counts[1], -plan.assigned_count,
                                           plan.used_engineers, round(plan.total_km, 3)]}


@api.get("/api/explain/{region}/{request_id}")
def explain_one(region: str, request_id: str, check: bool = False):
    st = _state(region)
    source = st.get("check") if check else st
    if not source or source.get("plan") is None:
        raise HTTPException(status_code=404, detail="Плана нет")
    return explain.explain_assignment(source["plan"], source["inst"], request_id)


@api.get("/api/plan/{region}")
def current_plan(region: str):
    """Рабочий план дня — например, чтобы вернуться к нему после сверки с OR-Tools. Плана нет — 404."""
    with COMPUTE:
        st = _state(region)
        st.pop("check", None)
        if st["plan"] is None:
            raise HTTPException(status_code=404, detail="План ещё не построен")
        payload = _plan_payload(region)
        payload["search"] = st.get("report")
        return payload


@api.post("/api/event")
def apply_event(req: EventRequest):
    """Событие — одна транзакция: прежний план → событие → поиск → сохранение (Н45)."""
    arrived = time.perf_counter()
    with COMPUTE:
        return _apply_event(req, time.perf_counter() - arrived)


def _apply_event(req: EventRequest, queued: float = 0.):
    st = _state(req.region)
    inst, before = st["inst"], st["plan"]
    if before is None:
        raise HTTPException(status_code=422, detail="Сначала постройте план")
    day = inst.engineers[0].shift_start
    try:
        hh, mm = map(int, req.at.split(":"))
        at = day.replace(hour=hh, minute=mm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Нужно время в формате HH:MM") from exc

    payload = None
    if req.kind == "urgent":
        if req.lat is None or req.lon is None:
            raise HTTPException(status_code=422, detail="Укажите координаты срочной заявки")
        payload = {"lat": req.lat, "lon": req.lon, "district": req.district or "—",
                   "hd_type": req.hd_type,
                   "address": req.address or f"точка {req.lat:.4f}, {req.lon:.4f}",
                   "id": f"SOS-{uuid4().hex}"}
        if req.duration_min is not None:
            payload["duration_min"] = req.duration_min
    event = events.Event(kind=req.kind, at=at, request_id=req.request_id,
                         engineer_id=req.engineer_id, payload=payload, reserve=req.reserve)

    try:
        if req.kind == "assign":
            after, inst = assign(inst, before, req.request_id, req.engineer_id, at)
        else:
            search, weights = _snapshot()
            members = settings.search_members("event", search)
            reports = []

            def replan(event_inst, warm, frozen):
                # Поиск после события: портфель в составе «событие», без шума вставки.
                result = portfolio(search).run(event_inst, warm, members, seconds=search["event"]["seconds"],
                                               repair_noise=EVENT_REPAIR_NOISE, weights=weights,
                                               urgent_front=search.get("urgent_front", False))
                reports.append(search_report(result))
                return result.best
            after, inst = events.apply(inst, before, event, search=replan)
            # Отчёт сохраняется вместе с планом — только если событие прошло проверки.
            reports[-1]["queued"] = round(queued, 1)
            st["report"] = reports[-1]
            after.meta = {"seconds": reports[-1]["seconds"], "iterations": reports[-1]["iterations"], "engine": "population"}
        if req.kind == "assign":
            after.meta = {"engine": "manual"}             # без поиска: интерфейс так и подписывает
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    st["inst"], st["plan"] = inst, after
    names = {e.id: e.name for e in inst.engineers}
    text = (f"В {at:%H:%M} заявка {req.request_id} вручную назначена: {names.get(req.engineer_id, req.engineer_id)}."
            if req.kind == "assign" else events.describe(event, inst))
    st["events"].append(text)

    out = _plan_payload(req.region)
    out["diff"] = explain.plan_diff(before, after, inst)
    out["event_text"] = st["events"][-1]
    if req.kind != "assign":
        out["search"] = st["report"]         # отчёт поиска события, в том числе предупреждение памяти (Н55)
    return out


class SettingsPatch(BaseModel):
    normatives: dict | None = None
    assumptions: dict | None = None


class WeightsPatch(BaseModel):
    # Строгие числа: true не превращается в 1.0 до проверки весов (Н60, замечание Codex).
    unassigned_request: StrictFloat | StrictInt | None = None
    engineer_used: StrictFloat | StrictInt | None = None
    urgent_delay_minute: StrictFloat | StrictInt | None = None
    open_engineer_penalty: StrictFloat | StrictInt | None = None
    urgent_delay_mode: Literal["linear", "square"] | None = None     # хвост цели
    urgent_delay_square: StrictFloat | StrictInt | None = None
    urgent_target_first: StrictBool | None = None                   # аварии сверх ориентира выше исполнителей
    urgent_target_minutes: StrictFloat | StrictInt | None = None
    event_move_km: StrictFloat | StrictInt | None = None           # цена переназначения после события


@api.get("/api/settings")
def get_settings():
    """Справочники для административного экрана; changed — какие вкладки отличаются от заводских умолчаний."""
    return {**settings.read(), "weights": settings.read_weights(), "search": settings.read_search(),
            "changed": settings.changed()}


class ResetRequest(BaseModel):
    scope: Literal["goal", "search", "norms", "all"]


@api.post("/api/settings/reset")
def reset_settings(req: ResetRequest):
    """Сброс к заводским умолчаниям (вкладка или всё). Планы сбрасываем: могли поменяться цель и допущения."""
    with COMPUTE:
        reset = settings.reset(req.scope)
        STATE.clear()
    return {"reset": reset, "changed": settings.changed()}


@api.post("/api/weights")
def post_weights(patch: WeightsPatch):
    """Правка приоритетов планирования. Планы сбрасываем: цель изменилась."""
    with COMPUTE:     # правка настроек — между расчётами, не посреди (Н45, Н46)
        try:
            result = settings.update_weights(patch.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        STATE.clear()
        return result


class SearchSettingsPatch(BaseModel):
    # Строгие типы: true не превращается в 1 (Н42). Содержимое разделов проверяет settings.check_search.
    workers: StrictInt | None = None
    plan: dict | None = None
    event: dict | None = None
    urgent_front: StrictBool | None = None      # постобработка «аварии вперёд»


@api.post("/api/search-settings")
def post_search_settings(patch: SearchSettingsPatch):
    """Состав портфеля поиска (шаг 6в). Действует со следующего расчёта; план не сбрасываем."""
    with COMPUTE:     # правка настроек — между расчётами, не посреди (Н45, Н46)
        # refactor: Claude, 22.09.2026 — шаг 6в
        try:
            return settings.update_search(patch.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@api.post("/api/settings")
def post_settings(patch: SettingsPatch):
    """Правка справочников. Инстансы сбрасываем: длительности и смены влияют на план."""
    with COMPUTE:     # между расчётами, не посреди (Н45)
        result = settings.update(patch.model_dump(exclude_none=True))
        STATE.clear()
    result["note"] = "Справочники обновлены, планы будут пересчитаны заново"
    return result


@api.get("/api/analysis/{region}")
def dataset_analysis(region: str):
    """Сводка по набору до поиска: работа по навыкам и что достижимо для аварий (вкладка «Анализ»)."""
    return {"region": region, **analysis.summary(_state(region)["origin"], solver.URGENT_TARGET_MIN)}


@api.get("/api/search-stats/{region}")
def search_stats(region: str):
    """Что делал поиск на последнем расчёте этого участка: участники и полезные операторы."""
    st = STATE.get(region)
    if not st or not st.get("report"):
        return {"available": False}
    report = st["report"]
    records, better = {}, {}
    for member in report["members"]:
        for name, count in (member.get("wins") or {}).get("records", {}).items():
            records[OPERATOR_TITLES.get(name, name)] = records.get(OPERATOR_TITLES.get(name, name), 0) + count
        for name, count in (member.get("wins") or {}).get("better", {}).items():
            better[OPERATOR_TITLES.get(name, name)] = better.get(OPERATOR_TITLES.get(name, name), 0) + count
    return {"available": True, **report,
            "records": dict(sorted(records.items(), key=lambda x: -x[1])),
            "better": dict(sorted(better.items(), key=lambda x: -x[1]))}


def _plan_setup(search: dict, seconds: float, patience: int | None) -> dict:
    """С чем строится план дня: состав, бюджет, остановка. Одинаковый setup — одинаковые условия расчёта."""
    return {"members": tuple(settings.search_members("plan", search)), "seconds": seconds, "patience": patience,
            "urgent_front": search.get("urgent_front", False)}


# refactor: Claude, 26.09.2026 — шаг 6 PORT-PLAN: «что если» быстро или как на главной, весь кортеж цели
WHATIF_CASES = ("как сейчас", "все умеют все работы", "все на автомобилях", "нормативы короче на 20%",
                "нормативы длиннее на 20%", "смена длиннее на час")
WHATIF_FAST_SECONDS = 6.0


def _whatif_setup(region: str, search: dict, full: bool):
    """Условия расчёта вариантов и готовый «как сейчас», если его можно взять с главной.

    Быстро — один равномерный поиск по 6 с на вариант: варианты сравнимы между собой, но не с планом на главной.
    Как на главной — состав и остановка из настроек, бюджет последнего плана участка (нет плана — 20 с). «Как сейчас» —
    сам план с главной, если он построен в тех же условиях и после него не было событий (иначе считается заново).
    """
    st = _state(region)
    if not full:
        return {"members": (Member("uniform", 0),), "seconds": WHATIF_FAST_SECONDS, "patience": None,
                "urgent_front": search.get("urgent_front", False)}, None
    last = st.get("setup")
    seconds = last["seconds"] if last else PlanRequest.model_fields["seconds"].default
    patience = None if search["plan"]["stop"] == "time" else search["plan"]["patience"]
    setup = _plan_setup(search, seconds, patience)
    current = st["plan"] if st["plan"] is not None and not st["events"] and last == setup else None
    return setup, current


@api.get("/api/whatif/{region}/estimate")
def whatif_estimate(region: str):
    """Сколько займёт «что если» в каждом режиме — для предупреждения до запуска."""
    search = settings.read_search()
    setup, current = _whatif_setup(region, search, full=True)
    runs = len(WHATIF_CASES) - (current is not None)
    fast = len(WHATIF_CASES) * WHATIF_FAST_SECONDS
    return {"cases": len(WHATIF_CASES), "fast_seconds": fast, "full_runs": runs, "full_case_seconds": setup["seconds"],
            "full_seconds": runs * setup["seconds"], "converge": setup["patience"] is not None,
            "members": len(setup["members"]), "current_from_plan": current is not None}


@api.get("/api/whatif/{region}")
def whatif(region: str, full: bool = False):
    """Что будет с планом, если поменять допущение. Считает варианты на лету (см. _whatif_setup)."""
    with COMPUTE:
        import copy
        from datetime import timedelta

        # «Что если» — альтернативный день по исходной задаче: после события уже начатую работу переписывать
        # нельзя (Н69), как и «Построить план» (Н63).
        base = _state(region)["origin"]
        search, weights = _snapshot()
        setup, current = _whatif_setup(region, search, full)

        def clone(mutate):
            inst = copy.deepcopy(base)
            mutate(inst)
            return inst

        def universal(i):
            for e in i.engineers:
                e.skills = {"local", "connect", "emergency"}

        def all_car(i):
            for e in i.engineers:
                e.transport = "car"

        def faster(i):
            for r in i.requests:
                if not r.urgent: r.duration_min = int(round(r.duration_min * 0.8))

        def slower(i):
            for r in i.requests:
                if not r.urgent: r.duration_min = int(round(r.duration_min * 1.2))

        def longer_shift(i):
            for e in i.engineers:
                e.shift_end = e.shift_end + timedelta(hours=1)

        mutations = (lambda i: None, universal, all_car, faster, slower, longer_shift)

        out, ref = [], None
        for name, mutate in zip(WHATIF_CASES, mutations):
            if ref is None and current is not None:
                inst, plan, source = base, current, "план с главной"
            else:
                inst = clone(mutate)
                plan = portfolio(search).run(inst, construct(inst), list(setup["members"]), seconds=setup["seconds"],
                                             patience=setup["patience"], weights=weights,
                                             urgent_front=setup["urgent_front"]).best
                source = "посчитан здесь"
            s = metrics.summarize(plan, inst, name)
            wait = s["urgent_wait"]
            # Строки цели в её порядке: аварии, подключения, заявки, аварий сверх ориентира, бригады, км, Σ ожидания.
            row = {"urgent": s["priority_assigned"].get("emergency", 0), "connect": s["priority_assigned"].get("connection", 0),
                   "assigned": s["assigned"], "over_target": wait["over_target"], "engineers": s["used_engineers"],
                   "km": round(s["total_km"]), "wait_min": wait["sum_min"]}
            if ref is None:
                ref = row
            out.append({"case": name, "source": source, "total": len(inst.requests), **row,
                        "delta": {k: row[k] - ref[k] for k in row}})
        return {"region": region, "full": full, "seconds": setup["seconds"], "members": len(setup["members"]),
                "target_min": round(solver.URGENT_TARGET_MIN), "cases": out}


class TrafficLoad(BaseModel):
    region: str
    source: str = "profile"           # profile | file | provider
    path: str | None = None
    content: str | None = None        # файл слоёв, загруженный из браузера (текст JSON) — вместо пути
    filename: str | None = None


@api.get("/api/traffic/{region}")
def traffic_state(region: str):
    inst = _state(region)["origin"]     # файл пробок — по точкам исходной задачи (Н63)
    # refactor: Claude, 26.09.2026 — шаг 8: готовые файлы слоёв набора и пробки по часам для графика
    return {"region": region, "current": inst.matrix.time_source,
            "sources": traffic_source.describe_sources(has_key=bool(settings.traffic_key())),
            "points": len(inst.matrix.points),
            "files": traffic_source.list_files(region, len(inst.matrix.points)),
            "hours": analysis.traffic_hours(inst)}


@api.get("/api/traffic/{region}/sample")
def traffic_sample(region: str):
    """Пример файла слоёв для этого набора: формат загрузки с правильным числом точек — заменить матрицы своими и загрузить.
    refactor: Claude, 26.09.2026 — как «скачать пример» у наборов заявок (просьба Михаила)"""
    import json
    from fastapi.responses import Response
    from urllib.parse import quote
    from src.data.traffic_layers import sample_blob
    inst = _state(region)["origin"]
    body = json.dumps(sample_blob(region, inst), ensure_ascii=False, separators=(",", ":"))
    name = quote(f"{region}-пример-слоёв.json")
    return Response(body, media_type="application/json; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{name}"})


@api.post("/api/traffic")
def traffic_load(req: TrafficLoad):
    """Переключение источника времён в пути. Планы сбрасываются: дороги стали другими."""
    with COMPUTE:     # матрица задачи меняется между расчётами, не посреди (Н45)
        return _traffic_load(req)


def _traffic_load(req: TrafficLoad):
    st = _state(req.region)
    inst = st["origin"]            # новый источник — для нового плана дня, а он строится по исходной задаче (Н63)
    try:
        if req.source == "file":
            path = req.path or ""
            if req.content is not None:      # свой файл из браузера — сохраняется в data/traffic/ и подключается
                path = traffic_source.save_upload(req.region, req.filename or "слои.json", req.content,
                                                  len(inst.matrix.points))
            layers, step = traffic_source.load_from_file(path, len(inst.matrix.points))
            inst.matrix.load_layers(layers, step)
        elif req.source == "provider":
            traffic_source.fetch_from_provider(
                settings.traffic_provider(), settings.traffic_key(), inst.matrix.points, range(24))
        else:
            inst.matrix.traffic.layers = {}
    except traffic_source.TrafficSourceError as e:
        return {"ok": False, "error": str(e), "current": inst.matrix.time_source}
    st["inst"], st["plan"], st["events"] = inst, None, []
    return {"ok": True, "current": inst.matrix.time_source, "hours": analysis.traffic_hours(inst),
            "note": "Источник переключён, план нужно построить заново"}


def _asset_version() -> str:
    """Версия статики — время правки файлов. Без неё браузер держит старый app.js
    рядом с новой разметкой, и страница молча остаётся пустой."""
    stamps = [(WEB / name).stat().st_mtime for name in ("app.js", "style.css", "admin.js", "tour.js", "tour-content.js")
              if (WEB / name).exists()]
    return str(int(max(stamps))) if stamps else "0"


@api.get("/")
def index():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__V__", _asset_version()))


@api.get("/documentation", response_class=HTMLResponse)
def documentation():
    """Документация решения (DOCUMENTATION.html) рядом с прототипом. Ссылки на файлы репозитория (*.md) здесь
    не откроются — показываются текстом. refactor: Claude, 26.09.2026"""
    import re
    html = (DOCS / "DOCUMENTATION.html").read_text(encoding="utf-8")
    html = html.replace('href="ENGINE.html"', 'href="/engine"')
    return HTMLResponse(re.sub(r'<a href="[^"#:]+\.md">([^<]*)</a>', r"<b>\1</b>", html))


@api.get("/engine", response_class=HTMLResponse)
def engine_map():
    """Интерактивная схема движка поиска и графовой сети (ENGINE.html) — к разделу 17 документации."""
    return HTMLResponse((DOCS / "ENGINE.html").read_text(encoding="utf-8"))


@api.get("/admin", response_class=HTMLResponse)
def admin_page():
    html = (WEB / "admin.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__V__", _asset_version()))


api.mount("/static", StaticFiles(directory=WEB), name="static")
