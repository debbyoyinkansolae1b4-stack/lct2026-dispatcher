"""Интерфейс политики выбора действия для цикла популяции.

Политика за эпоху: forward(g) — один проход по графу родителя; sample(out, g, rng) —
действие для одного кандидата с тем, что нужно обучению (ActionTrace); update(...) —
шаг обучения по исходам эпохи (у необучаемых политик — ничего).
"""
# refactor: Claude, 21.09.2026 — шаг 4 REFACTOR-PLAN.md
# refactor: Claude, 21.09.2026 — шаг 5г.3: выбор сети отдельно от действия над задачей
from dataclasses import dataclass

import torch


@dataclass
class ActionTrace:
    """Выбранное действие и его след для обучения. Тензоры сюда, а не в Action."""
    action: dict              # действие над задачей, индексы inst.requests: мутация и журнал
    logp: torch.Tensor        # log-prob обучаемых голов
    entropy: torch.Tensor     # нормированная энтропия обучаемых голов
    choice: dict              # тот же выбор в индексах узлов графа: цели обучения голов


@dataclass
class Update:
    """Что вернул шаг обучения: диагностика для журнала и хеши градиентов для эталона."""
    diagnostics: dict
    gradients: dict | None = None


class Policy:
    trains = False

    def forward(self, g):
        raise NotImplementedError

    def sample(self, out, g, rng) -> ActionTrace:
        raise NotImplementedError

    def update(self, out, traces, rewards, adv, sample_weights, golden_targets=None) -> Update:
        return Update(diagnostics={})
