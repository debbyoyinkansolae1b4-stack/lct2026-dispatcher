"""Перепланирование после события популяционным поиском (шаг 5 REFACTOR-PLAN.md).

events.apply готовит задачу (закреплённое начало маршрутов, момент события, изменённый
состав заявок и исполнителей), строит стартовый план и проверяет результат; здесь —
только поиск между этими шагами.
"""
# refactor: Claude, 21.09.2026 — шаг 5
# refactor: Claude, 21.09.2026 — шаг 5г: закреплённое из RouteState, шум задаёт адаптер
# перенос: Claude, 26.09.2026 — пресеты режимов вместо весов энергии (PORT-PLAN.md)
import math

from src.engine.route_state import RouteState
from src.search.population import SearchConfig, run

# Шум вставки после события. Прежнее поведение — без шума (как питоновский путь);
# включение — отдельный объявленный эксперимент (FINDINGS Н26).
EVENT_REPAIR_NOISE = 0.


def population_search(seconds=6., epochs=None, mode='uniform', population=64, seed=29, clock=None):
    """Функция поиска для events.apply(search=...): популяция от стартового плана события.

    Режим — пресет машины поиска (portfolio.PRESETS: uniform — U2n, learned — сеть b).
    Граница — время; горизонт расписания 500 эпох, как в замерах (Н39: не лимит).
    """
    from src.search.portfolio import PRESETS
    # Без бюджета времени нужен лимит эпох — горизонт расписания.
    epochs = epochs if epochs is not None or seconds else 500
    config = SearchConfig(population=population, epochs=epochs, schedule_epochs=500,
                          seconds=seconds if seconds else math.inf, mode=mode, seed=seed,
                          repair_noise=EVENT_REPAIR_NOISE, **PRESETS[mode])

    def search(inst, warm, frozen):
        # Двигать нечего (всё активное уже закреплено или заявок нет) — стартовый план и есть ответ.
        pinned = RouteState(inst).pinned
        movable = [r for r in inst.requests if r.id not in pinned]
        if not movable:
            return warm
        return run(inst, warm, config, clock=clock).best
    return search
