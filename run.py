"""CLI: построить план по участку, сравнить с базовым вариантом, выгрузить результат.

    python run.py --region Юго-восток
    python run.py --region Восток --seconds 30
    python run.py --all --json artifacts/bench.json

Поиск — портфель из настроек (config/search.json), как в приложении.
"""
# refactor: Claude, 22.09.2026 — шаг 7: портфель популяционного поиска вместо ALNS

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import settings

from src.report import metrics
from src.instance import build
from src.data.loading import REGIONS
from src.search.portfolio import Portfolio
from src.search.report import search_report
from src.plan.solver import baseline_greedy, construct


def run_region(region: str, seconds: float, portfolio: Portfolio, quiet=False) -> dict:
    inst = build(region)
    if not quiet:
        print(f"\n{'=' * 78}\n{region}: заявок {len(inst.requests)} "
              f"(+{len(inst.nogeo)} без координат), исполнителей {len(inst.engineers)}, "
              f"матрица из {inst.matrix.source}")

    t = time.time()
    base = baseline_greedy(inst)
    base_s = metrics.summarize(base, inst, "Базовый вариант ТЗ (жадный)")
    base_time = time.time() - t

    t = time.time()
    greedy = construct(inst)
    constr_s = metrics.summarize(greedy, inst, "Конструктор (regret-2)")
    constr_time = time.time() - t

    # Как в приложении: состав и остановка — из одного снимка настроек.
    cfg = settings.read_search()
    patience = None if cfg["plan"]["stop"] == "time" else cfg["plan"]["patience"]
    result = portfolio.run(inst, greedy, settings.search_members("plan", cfg), seconds=seconds, patience=patience,
                           urgent_front=cfg.get("urgent_front", False))
    report = search_report(result, patience)
    plan = result.best
    search_s = metrics.summarize(plan, inst, "Популяционный поиск (портфель)")

    if not quiet:
        print("\n" + metrics.as_text(base_s))
        print("\n" + metrics.as_text(search_s))
        vs = metrics.compare(base_s, search_s)
        print(f"\nПротив базового варианта: заявок {vs['assigned_delta']:+d}, "
              f"исполнителей {vs['engineers_delta']:+d} ({vs['engineers_saved_pct']:+.0f}%), "
              f"пробег {vs['km_delta']:+.1f} км ({vs['km_saved_pct']:+.0f}%)")
        print(f"Доступно бригад на участке: {search_s['available_engineers']} — "
              f"план задействовал {search_s['used_engineers']}")
        print(f"Поиск: {len(report['members'])} участников, {report['candidates']} кандидатов "
              f"за {report['seconds']} c; {report['stop_reason']}: {report['hint']}")

    return {
        "region": region,
        "baseline": base_s, "construct": constr_s, "search": search_s,
        "vs_baseline": metrics.compare(base_s, search_s),
        "control": metrics.control_reference(inst),
        "timing": {"baseline_s": round(base_time, 2), "construct_s": round(constr_time, 2),
                   "search_s": report["seconds"], "epochs": report["iterations"]},
        "search_report": report,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", default="Юго-восток", choices=REGIONS)
    p.add_argument("--all", action="store_true", help="прогнать все три участка")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--json", type=Path, help="куда сохранить результаты")
    args = p.parse_args()

    regions = REGIONS if args.all else [args.region]
    portfolio = Portfolio(workers=settings.read_search()["workers"])
    try:
        results = [run_region(r, args.seconds, portfolio) for r in regions]
    finally:
        portfolio.close()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                             encoding="utf-8")
        print(f"\nРезультаты: {args.json}")


if __name__ == "__main__":
    main()
