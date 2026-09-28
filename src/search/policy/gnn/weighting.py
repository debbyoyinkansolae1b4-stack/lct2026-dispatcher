"""Веса кандидатов в обучении, уравновешенные по исходам, — сознательно НЕ несмещённый градиент политики.

Каждый отличающийся изменённый план получает равную массу; внутри плана каждое отличающееся действие — равную
долю, повторы делят её между собой; масса копии родителя ограничена. Веса и базовая линия эмпирические и не
участвуют в градиенте — это меняет цель обучения.
"""
# refactor: Claude, 21.09.2026 — из src/alt/population_weighting.py
from collections import defaultdict
import json
import torch


def balanced_batch(signatures,actions,rewards,noops,parent_cap=.05):
    if not 0<=parent_cap<1:raise ValueError('parent_cap must be in [0,1)')
    if not (len(signatures)==len(actions)==len(rewards)==len(noops)) or not len(rewards):
        raise ValueError('Nonempty aligned batch required')
    groups=defaultdict(lambda:defaultdict(list))
    for i,(sig,action) in enumerate(zip(signatures,actions)):
        groups[sig][json.dumps(action,sort_keys=True)].append(i)
    changed=[]
    parents=[]
    for sig,by_action in groups.items():
        indices=[i for ids in by_action.values() for i in ids]
        if any(noops[i]!=noops[indices[0]] for i in indices):raise ValueError('Inconsistent noop labels')
        if not torch.allclose(rewards[indices],rewards[indices[0]].expand(len(indices))):
            raise ValueError('Identical plans have inconsistent rewards')
        (parents if noops[indices[0]] else changed).append(sig)
    parent_mass=min(parent_cap,len(parents)/len(groups)) if changed else 1.
    w=torch.zeros_like(rewards)
    for sig,by_action in groups.items():
        mass=(parent_mass/max(len(parents),1) if sig in parents else (1-parent_mass)/len(changed))
        for ids in by_action.values():w[ids]=mass/len(by_action)/len(ids)
    w=w.detach()
    mean=(w*rewards).sum()
    std=(w*(rewards-mean).square()).sum().sqrt()
    adv=(rewards-mean)/(std+1e-6) if std>1e-6 else torch.zeros_like(rewards)
    # Parent is never credited as a good mutation merely because others are worse.
    noop=torch.tensor(noops,dtype=torch.bool)
    adv=adv.clone()
    adv[noop]=adv[noop].clamp_max(0)
    if not changed:adv.zero_()
    return w,adv,dict(unique_outcomes=len(groups),unique_changed=len(changed),
                     parent_mass=float(w[torch.tensor(noops,dtype=torch.bool)].sum()),
                     effective_samples=float(1/w.square().sum()),target=float(mean),
                     objective='outcome_balanced_surrogate_not_unbiased_reinforce')
