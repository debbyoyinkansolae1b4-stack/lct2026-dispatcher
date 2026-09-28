"""Обучение политики: награда кандидатов и функция потерь эпохи.

Порядок сложения слагаемых потерь и обращений к autograd — часть контракта:
сумма в плавающей точке зависит от порядка, а эталон шага 0 сверяет градиенты
и веса побайтово.
"""
# refactor: Claude, 21.09.2026 — из src/alt/population_gnn.py, planner_population.py и train_loss
# цикла замеров; стиль, логика без изменений
import math
from collections import Counter

import torch
import torch.nn.functional as F

from src.search.actions import OPERATORS
from src.search.policy.gnn.loss_balance import CriticBalance, shared_gradient_diagnostics

ENTITY_HEADS = ('engineer', 'route', 'request')
ENTROPY_TARGET = .75        # нормированная энтропия в [0, 1]; пол исследования явный
OUTCOME_TEMPERATURE = .25   # цель головы по исходам: softmax(средняя награда / T)


def _group_norms(net, grads):
    """Нормы градиента по верхнеуровневым модулям сети (для отчёта)."""
    groups = {}
    for (name, _), grad in zip(net.named_parameters(), grads):
        group = name.split('.')[0]
        groups[group] = groups.get(group, 0.) + (float(grad.detach().square().sum()) if grad is not None else 0.)
    return {k: v ** .5 for k, v in groups.items()}


def _outcome_loss(name, logits, actions, rewards, golden):
    """Цель головы сущности по исходам эпохи: KL к softmax средней награды выбранных сущностей.

    Считается отдельно для каждого оператора, где голова что-то выбрала.
    Возвращает (потеря, описания целей для журнала).
    """
    selected = [(a[name], a['operator'], float(r)) for a, r in zip(actions, rewards) if name in a]
    terms, descriptions = [], []
    for op in range(len(OPERATORS)):
        observations = [(i, r) for i, o, r in selected if o == op]
        if not observations:
            continue
        counts = Counter(i for i, r in observations)
        totals = Counter()
        for i, r in observations:
            totals[i] += r
        ids = sorted(counts)
        empirical = torch.tensor([totals[i] / counts[i] for i in ids])
        target = (empirical / OUTCOME_TEMPERATURE).softmax(0)
        if golden is not None:
            golden.append((name, op, ids, target))
        terms.append(F.kl_div(logits[ids, op].log_softmax(0), target, reduction='sum'))
        descriptions.append(dict(operator=op, support=len(ids), reward_spread=float(empirical.std(unbiased=False))))
    return (torch.stack(terms).mean() if terms else logits.sum() * 0), descriptions


def train_loss(net, out, actions, logps, entropies, rewards, adv, diagnostics=False, critic_balance=None,
               sample_weights=None, gate_floor=None, head_targets='hints', golden=None):
    """Потери эпохи: актор, критик, энтропия с температурой, гейты экспертов, головы сущностей.

    head_targets — чему учить головы сущностей: подсказке ('hints'), исходам эпохи
    ('outcomes') или обоим. golden — список для трассы эталона (цели голов).
    Возвращает (потеря, диагностика для журнала).
    """
    w = sample_weights if sample_weights is not None else torch.full_like(rewards, 1 / len(rewards))
    actor = -(w * torch.stack(logps) * adv.detach()).sum()
    critic_target = (w * rewards).sum().detach()
    critic = F.mse_loss(out['value'], critic_target)
    entropy = (w * torch.stack(entropies)).sum()
    alpha = net.log_alpha.clamp(-8, 2).exp()
    temperature_loss = alpha * (entropy.detach() - ENTROPY_TARGET)
    # Удержание гейта экспертов от насыщения — тем же приёмом, что и для действий.
    gate_entropy = torch.zeros(())
    gate_temperature = torch.zeros(())
    beta = torch.zeros(())
    if gate_floor is not None:
        parts = []
        for key in ('gate_action', 'gate_value'):
            p = out[key].clamp_min(1e-12)
            parts.append(-(p * p.log()).sum() / math.log(len(p)))
        gate_entropy = torch.stack(parts).mean()
        beta = net.log_beta.clamp(-8, 2).exp()
        gate_temperature = beta * (gate_entropy.detach() - gate_floor)

    aux, targets = [], {}
    for name in ENTITY_HEADS:
        logits = out[name]
        outcome, targets[name] = _outcome_loss(name, logits, actions, rewards, golden)
        hinted = (F.kl_div(logits.log_softmax(0), out['factual_targets'][name][:, None].expand_as(logits),
                           reduction='sum') / logits.shape[1]) if 'factual_targets' in out else None
        if hinted is None or head_targets == 'outcomes':
            aux.append(outcome)
        elif head_targets == 'both':
            aux.append(hinted + outcome)
        else:
            aux.append(hinted)
    lv = net.log_vars.clamp(-3, 3)
    weighted = sum(torch.exp(-lv[i]) * loss + lv[i] for i, loss in enumerate(aux))

    # Быстрый путь: масштаб критика уже приложен в прямом проходе (net.critic_scale),
    # отдельный обратный проход по критику не нужен.
    parameters = list(net.parameters())
    fast_critic = getattr(net, 'critic_scale', None) is not None
    if fast_critic:
        critic_grads = [None] * len(parameters)
        critic_groups = {}
    else:
        critic_grads = torch.autograd.grad(critic, parameters, retain_graph=True, allow_unused=True)
        critic_groups = _group_norms(net, critic_grads)
    # Отдельный проход по актору нужен только для отчётных норм и для режима adaptive:
    # в режиме fixed он стоил десятую часть эпохи ради одного числа в логе.
    balance_mode = (critic_balance or CriticBalance()).mode
    if diagnostics or balance_mode != 'fixed':
        actor_grads = torch.autograd.grad(actor, list(net.parameters()), retain_graph=True, allow_unused=True)
        actor_groups = _group_norms(net, actor_grads)
    else:
        actor_grads = [None] * len(list(net.parameters()))
        actor_groups = {}
    if fast_critic:
        correction = torch.zeros(())
        balance_info = dict(mode='forward-scaled', scale=net.critic_scale)
    else:
        correction, balance_info = (critic_balance or CriticBalance()).correction(net, actor_grads, critic_grads)
    components = dict(actor=actor, critic=.5 * critic + correction, entropy=-alpha.detach() * entropy,
                      **{f"aux_{i}": torch.exp(-lv[i]) * term for i, term in enumerate(aux)})
    conflicts = shared_gradient_diagnostics(net, components) if diagnostics else {}
    loss = (actor + .5 * critic + correction - alpha.detach() * entropy + temperature_loss + weighted
            - beta.detach() * gate_entropy + gate_temperature)
    return loss, dict(
        detailed_gradients_measured=bool(diagnostics), critic_balance=balance_info, shared_gradients=conflicts,
        actor=float(actor.detach()), critic=float(critic.detach()), entropy=float(entropy.detach()),
        alpha=float(alpha.detach()), gate_entropy=float(gate_entropy.detach()),
        beta=float(beta.detach() if gate_floor is not None else 0.),
        actor_gradient_norms=actor_groups, critic_gradient_norms=critic_groups,
        critic_prediction=float(out['value'].detach()), critic_target=float(critic_target),
        aux=[float(x.detach()) for x in aux], aux_weights=torch.exp(-lv.detach()).tolist(), targets=targets)
