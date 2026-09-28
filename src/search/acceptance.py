"""Принятие кандидата в текущий план: отбор по кортежу цели (лечение Н76), без весов энергии.
"""
# перенос: Claude, 26.09.2026 — отжиг по энергии удалён (PORT-PLAN.md)
# refactor: Claude, 21.09.2026 — перенос без изменения кода (шаг 3, REFACTOR-PLAN.md)
import math

# Главное цели — строки кортежа 1–5 (аварии, подключения, назначено, аварии сверх ориентира, исполнители):
# ухудшение в них не принимается никогда.
STRICT = 5


def anneal_tuple(current, candidate, epoch, epochs, costs, rng):
    """Отжиг по cost_tuple без весов энергии (патч research/patches/tuple_search.patch, Михаил 24.09).

    Не хуже по кортежу — принять. Хуже в главном (строки 1–5) — отказ. Хуже в младшей строке k (стоимость дня,
    затем ожидание) — принять с вероятностью exp(-(Δ/разброс)/τ): разброс — медиана |c[k] - current[k]| по
    кандидатам эпохи costs, совпадающим с current до k и отличающимся в k (масштаб в своих единицах — км или
    минуты этой эпохи); τ = max(.05, 1 - epoch/epochs). Нечем мерить разброс — отказ.
    """
    current, candidate = tuple(current), tuple(candidate)
    if len(current) != len(candidate):
        raise ValueError('cost_tuple разной длины')
    if candidate <= current:
        return True, dict(component=None, delta=0., spread=None, temperature=None, probability=1., accepted=True)
    k = next(i for i, (a, b) in enumerate(zip(current, candidate)) if a != b)
    delta = candidate[k] - current[k]
    if k < STRICT:
        return False, dict(component=k, delta=delta, spread=None, temperature=None, probability=0., accepted=False)
    diffs = sorted(abs(c[k] - current[k]) for c in map(tuple, costs) if c[:k] == current[:k] and c[k] != current[k])
    if not diffs:
        return False, dict(component=k, delta=delta, spread=None, temperature=None, probability=0., accepted=False)
    spread = diffs[len(diffs) // 2]
    temperature = max(.05, 1 - epoch / epochs)
    probability = math.exp(-(delta / spread) / temperature)
    accepted = rng.random() < probability
    return accepted, dict(component=k, delta=delta, spread=spread, temperature=temperature, probability=probability,
                          accepted=accepted)
