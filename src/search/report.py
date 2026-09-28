"""Отчёт поиска для приложения: что делал портфель, почему остановился, кто нашёл ответ.

Из PortfolioResult собирается словарь для API (/api/plan → search, /api/search-stats).
Поля stop_reason, iterations, last_improvement, stall, hint сохраняют смысл прежнего
отчёта ALNS, чтобы интерфейс переключался без переделки (шаг 7): iterations — эпохи
участника-победителя, last_improvement — эпоха его последнего рекорда.

Формулировки остановки — решение Дебби 21.09 (REFACTOR-PLAN.md): «улучшений не найдено
за N эпох», а не «оптимум достигнут».
"""
# refactor: Claude, 22.09.2026 — шаг 6б
from dataclasses import asdict

from src.search.actions import OPERATORS

OPERATOR_TITLES = {'route_segment': 'кусок маршрута', 'route_pair': 'пара маршрутов',
                   'geographic': 'соседи по карте', 'time_window': 'соседи по окну',
                   'transfer': 'перенос заявки', 'dissolve_route': 'роспуск маршрута'}
MODE_TITLES = {'uniform': 'равномерный', 'learned': 'сеть'}  # в интерфейсе — по-русски (Михаил 26.09)
STOP_TITLES = {'time_limit': 'время', 'patience': 'сошлось', 'epoch_limit': 'лимит циклов'}
# Рекорд в последней пятой части расчёта — поиск ещё улучшал план.
STILL_IMPROVING = .8


class OperatorWins:
    """Получатель строк эпох (sink поиска): какие операторы давали кандидатов лучше родителя
    и новые рекорды. Копит только счётчики, строки не хранит."""

    def __init__(self):
        self.better, self.records = {}, {}
        self.progress = []          # по циклу: [цикл, секунды, заявок, бригад, км] лучшего плана — график в «Анализе»

    def __call__(self, row):
        self.progress.append([row['epoch'] + 1, round(row['elapsed'], 2), row['best_assigned'], row['best_engineers'],
                              row['best_km']])
        parent = tuple(row['parent'])
        candidates = row.get('candidates', [])
        for c in candidates:
            if tuple(c['cost']) < parent:
                name = OPERATORS[c['action']['operator']]
                self.better[name] = self.better.get(name, 0) + 1
        if row.get('record_improved') and candidates:
            winner = min(candidates, key=lambda c: tuple(c['cost']))
            name = OPERATORS[winner['action']['operator']]
            self.records[name] = self.records.get(name, 0) + 1

    def summary(self):
        return dict(better=dict(self.better), records=dict(self.records))


def stop_hint(stop_reason, epochs, best_epoch, patience=None):
    """Одна фраза для диспетчера: почему расчёт закончился и стоит ли считать дольше."""
    if stop_reason == 'patience':
        return f'Улучшений не было {patience or epochs - best_epoch} циклов поиска подряд — расчёт остановлен'
    if stop_reason == 'epoch_limit':
        return 'Исчерпан лимит циклов поиска'
    if epochs and best_epoch >= STILL_IMPROVING * epochs:
        return ('Расчёт прерван по бюджету, поиск ещё улучшал план. '
                'Увеличьте время, если нужен более плотный план')
    return (f'Последнее улучшение — на {best_epoch}-м цикле поиска из {epochs}: '
            f'за остаток бюджета лучше не нашлось')


def search_report(result, patience=None):
    """Словарь отчёта по PortfolioResult."""
    members = []
    for i, r in enumerate(result.reports):
        members.append(dict(
            number=i + 1, mode=r.member.mode, mode_title=MODE_TITLES.get(r.member.mode, r.member.mode),
            seed=r.member.seed, ok=r.ok, error=r.error, winner=i == result.winner,
            epochs=r.epochs, best_epoch=r.best_epoch, candidates=r.candidates,
            seconds=round(r.seconds, 1), stop_reason=STOP_TITLES.get(r.stop_reason, r.stop_reason),
            cost=list(r.cost) if r.cost else None,
            peak_memory_mb=round(r.peak_memory_mb) if r.peak_memory_mb is not None else None,
            cpu_seconds=round(r.cpu_seconds, 1),
            wins=r.wins, progress=r.progress,
            network=dict(hidden=r.member.hidden, experts=r.member.experts, lr=r.member.lr)
            if r.member.mode == 'learned' else None))
    lead = result.reports[result.winner] if result.winner is not None else None
    ok = [r for r in result.reports if r.ok]
    # Ответ — стартовый план (никто не улучшил или все упали): отчёт по самому долгому участнику.
    shown = lead or (max(ok, key=lambda r: r.epochs) if ok else None)
    report = dict(
        engine='population', seconds=round(result.seconds, 1), queued=0.,   # очередь меряет приложение
        prepare_seconds=round(result.prepare_seconds, 1), memory_warning=result.memory_warning,
        candidates=sum(r.candidates for r in ok), members=members,
        winner=result.winner + 1 if result.winner is not None else None,
        failed=[m for m in members if not m['ok']],
        weights=asdict(result.weights) if result.weights else None,
        improved=result.winner is not None, urgent_front_moves=result.urgent_front_moves)
    if shown is None:
        report.update(stop_reason='сбой', iterations=0, last_improvement=0, stall=0,
                      hint='Ни один поиск не завершил расчёт — показан стартовый план')
        return report
    report.update(stop_reason=STOP_TITLES.get(shown.stop_reason, shown.stop_reason), iterations=shown.epochs,
                  last_improvement=shown.best_epoch, stall=shown.epochs - shown.best_epoch,
                  hint=stop_hint(shown.stop_reason, shown.epochs, shown.best_epoch, patience))
    if lead is None:
        report['hint'] = 'Улучшить стартовый план не удалось. ' + report['hint']
    return report
