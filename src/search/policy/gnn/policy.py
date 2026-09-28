"""Политика на графовой сети (сеть b): выбор по распределению сети, обучение по исходам эпохи.

Порядок в конструкторе повторяет прогоны принятой базы: сид torch, создание сети,
разогревочный проход, Adam. От него зависят начальные веса и эталон шага 0.
"""
# refactor: Claude, 21.09.2026 — из цикла experiment_population_gnn.py (шаг 4)
# refactor: Claude, 21.09.2026 — шаг 5г: разогрев сети через тот же сервис вставок, что и эпохи
# refactor: Claude, 21.09.2026 — шаг 5г.3: цели по выбору в узлах графа, действие — через instance_action
# refactor: Claude, 22.09.2026 — шаг 9: нормировка входа по задаче (RunningNorm) вместо FEATURE_STATS
# patch learned_full: Claude, 25.09.2026 — ширины входа по графу, полный кредит хода
# перенос: Claude, 26.09.2026 — сеть b в продукте (PORT-PLAN.md): одноразовая, без заранее обученных весов и приборов стенда
import math

import torch

from src.engine.repair import RouteCache
from src.search.golden_trace import gradient_digest
from src.search.policy.base import ActionTrace, Policy, Update
from src.search.policy.gnn.graph import enrich_graph, graph
from src.search.policy.gnn.loss_balance import CriticBalance, clip_step_gradients
from src.search.policy.gnn.model import PlannerPopulationGNN
from src.search.policy.gnn.normalization import RunningNorm
from src.search.policy.gnn.train import train_loss
from src.search.policy.sampling import instance_action, sample


class GnnPolicy(Policy):
    """Сеть b: одноразовая, учится под задачу с нуля (решение 21.09). Режим uniform обслуживает NetlessPolicy."""

    def __init__(self, inst, start, config, trace=False, cache=None):
        self.config = config
        self.trains = config.mode == 'learned'
        self.trace = trace
        # Разогревочный проход по тому же пути, что и боевой: при приписанных признаках
        # иначе граф прежней ширины и падение на входной проекции.
        # cache — сервис маршрута поиска: разогрев видит те же вставки, что и эпохи.
        cache = cache or RouteCache(inst)
        warm = graph(inst, start, state=cache.state)
        if config.rehomable_feature or config.potential:
            warm = enrich_graph(inst, start, warm, cache, rehomable=config.rehomable_feature, potential=config.potential)
        # Патч learned_full: ширины входа — по графу (признаки цели добавлены в graph), а не константами.
        net = PlannerPopulationGNN(config.hidden, n_experts=config.experts, route_features=warm['routes'].shape[-1],
                                   request_features=warm['requests'].shape[-1], glob_features=warm['glob'].shape[-1])
        net(warm)
        # Масштаб критика в прямом проходе: отдельный обратный проход по критику не нужен.
        net.critic_scale = .1
        self.transferred = None
        self.net = net
        self.opt = torch.optim.Adam(net.parameters(), lr=config.lr)
        self.critic_balance = CriticBalance('fixed')
        # Нормировка входа по самой задаче: статистика копится по графам этого расчёта.
        self.norm = RunningNorm() if config.normalize_features else None

    def forward(self, g):
        return self.net(self.norm(g) if self.norm is not None else g)

    def sample(self, out, g, rng):
        # Кредит за ход — всем выборам (оператор, вставка, масштаб, куда бить, цепочка). В мёртвой зоне (эпоха без
        # единого отличия от родителя) ходы равномерные: сигнала для сети нет.
        action, logp, entropy = sample(out, g, rng, uniform=bool(g.get('explore_all')),
                                       chain_max=self.config.chain_max, chain_learned=self.config.chain_learned)
        return ActionTrace(instance_action(action, g), logp, entropy, action)

    def update(self, out, traces, rewards, adv, sample_weights, golden_targets=None):
        if not self.trains:
            return Update(diagnostics={})
        net = self.net
        # Цели голов — в индексах узлов графа (t.choice), а не задачи.
        loss, diagnostics = train_loss(net, out, [t.choice for t in traces], [t.logp for t in traces],
                                       [t.entropy for t in traces], rewards, adv, diagnostics=False,
                                       critic_balance=self.critic_balance, sample_weights=sample_weights,
                                       head_targets='outcomes', golden=golden_targets)
        self.opt.zero_grad()
        loss.backward()
        gradients = dict(raw=gradient_digest(net)) if self.trace else None
        # Нечисло в градиенте портит веса молча: без этой проверки прогон досчитал бы мусор.
        if not all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite gradient')
        clipping = clip_step_gradients(net, separate=True)
        if self.trace:
            gradients['clipped'] = gradient_digest(net)
        diagnostics['clipping'] = clipping
        self.opt.step()
        diagnostics.update(total_loss=float(loss.detach()),
                           gradient_norm=math.hypot(clipping['before']['network'], clipping['before']['controllers']))
        diagnostics['experts'] = net.expert_report()
        return Update(diagnostics=diagnostics, gradients=gradients)
