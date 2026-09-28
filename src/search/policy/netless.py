"""Политика без сети — потомок сети b (U2n): та же машина поиска, выбор хода — правилом.

Сети нет — ни весов, ни прямого прохода, ни их памяти. Выходы голов — нули нужной формы:
по ним строятся маски. Все допустимые варианты равновероятны, кроме «куда бить»: доля
config.target_potential выбора маршрута и заявки — пропорционально потенциалу (наследство сети b).
"""
# refactor: Claude, 21.09.2026 — шаг 4б REFACTOR-PLAN.md (UniformPolicy)
# refactor: Claude, 21.09.2026 — шаг 5г.3: действие над задачей через instance_action
# refactor: Claude, 22.09.2026 — шаг 8½: одна политика на все режимы без сети
# перенос: Claude, 26.09.2026 — только uniform-потомок U2n (PORT-PLAN.md)
import torch

from src.search.actions import CHAINS, OPERATORS
from src.search.policy.base import ActionTrace, Policy
from src.search.policy.sampling import instance_action, sample

SCALES_HEAD = 3
REPAIRS = 6          # патч learned_full: regret-1/2/3 × {плотная, свободная}


class NetlessPolicy(Policy):
    net = opt = None
    transferred = None

    def __init__(self, config):
        self.config = config

    def forward(self, g):
        m, n = g['engineers'].shape[0], g['requests'].shape[0]
        ops = len(OPERATORS)
        return dict(operator=torch.zeros(ops), scale=torch.zeros(SCALES_HEAD), repair=torch.zeros(REPAIRS),
                    chain=torch.zeros(len(CHAINS)), engineer=torch.zeros(m, ops), route=torch.zeros(m, ops),
                    request=torch.zeros(n, ops))

    def sample(self, out, g, rng):
        action, logp, entropy = sample(out, g, rng, uniform=True, chain_max=self.config.chain_max,
                                       chain_learned=self.config.chain_learned,
                                       target_potential=self.config.target_potential)
        return ActionTrace(instance_action(action, g), logp, entropy, action)
