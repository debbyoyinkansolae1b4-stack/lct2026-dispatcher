"""Нормировка входа сети по самой задаче: скользящие среднее и разброс признаков.

Вместо таблицы FEATURE_STATS (72 признака, снята 20.09 с наших участков на прежнем профиле
нормативов — на другом районе молча перестаёт быть верной) среднее и разброс каждой колонки
копятся по графам, которые сеть видит в этом расчёте (нормировка наблюдений, как в RL).
Узлы графа — выборка: на первой же эпохе разброс считается по исполнителям, маршрутам
и заявкам; глобальные признаки — одна строка на эпоху, их разброс появляется со второй.

Защиты — регуляторы, а не подогнанные числа: мёртвый признак (разброс меньше DEAD_STD)
получает масштаб 1 и остаётся нулём после центрирования; значение обрезается по ±CLIP,
чтобы признак с дрейфом не заглушал остальные.
"""
# refactor: Claude, 22.09.2026 — шаг 9: нормировка по задаче вместо FEATURE_STATS (решение Михаила)
import torch

GROUPS = ('engineers', 'routes', 'requests', 'glob')
CLIP = 5.
DEAD_STD = 1e-3


class RunningNorm:
    """Среднее и разброс по колонкам каждой группы признаков, накопленные за расчёт (Уэлфорд)."""

    def __init__(self):
        self.stats = {}          # группа → (число строк, среднее, сумма квадратов отклонений), float64

    def _update(self, group, rows):
        count, mean, m2 = self.stats.get(group, (0, torch.zeros(rows.shape[1], dtype=torch.float64),
                                                torch.zeros(rows.shape[1], dtype=torch.float64)))
        k = rows.shape[0]
        batch_mean = rows.mean(0)
        batch_m2 = ((rows - batch_mean) ** 2).sum(0)
        total = count + k
        delta = batch_mean - mean
        mean = mean + delta * (k / total)
        m2 = m2 + batch_m2 + delta ** 2 * (count * k / total)
        self.stats[group] = (total, mean, m2)
        return total, mean, m2

    def __call__(self, g):
        """Обновить статистику графом g; вернуть копию графа с нормированными признаками.
        Сам g не меняется: выборка, подсказки и эталон видят сырой граф."""
        g = dict(g)
        for group in GROUPS:
            tensor = g.get(group)
            if tensor is None or not tensor.numel():
                continue
            rows = tensor.reshape(-1, tensor.shape[-1]).double()
            count, mean, m2 = self._update(group, rows)
            std = (m2 / count).sqrt() if count > 1 else torch.ones_like(mean)
            std = torch.where(std < DEAD_STD, torch.ones_like(std), std)
            g[group] = ((tensor.double() - mean) / std).clamp(-CLIP, CLIP).to(tensor.dtype)
        return g
