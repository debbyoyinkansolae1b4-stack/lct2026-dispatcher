"""Выбор действия по маскам графа: по распределению сети (b) или равномерно (U2n).

Общий для обеих политик: маски и порядок обращений к генератору одни и те же.
Каждая голова — ровно одно обращение rng.choices с весами по маске, даже когда
допустимый вариант один: от числа обращений зависит траектория поиска.
"""
# refactor: Claude, 21.09.2026 — из src/alt/population_gnn.py; стиль, логика без изменений
# refactor: Claude, 21.09.2026 — шаг 5г.3: узлы заявок — проекция, перевод индексов в instance_action
# перенос: Claude, 26.09.2026 — сеть b и U2n (PORT-PLAN.md): декодер звеньев, прицел uniform по потенциалу; без подсказок
import math

import torch

from src.search.actions import (CHAIN_EXTRA, CHAINS, DISSOLVE_ROUTE, OPERATORS, ROUTE_PAIR, ROUTE_SEGMENT,
                                TRANSFER)

EXPLORATION = .2           # доля равномерного распределения в каждой голове сети (пол исследования)


def distribution(logits, mask=None, uniform=False):
    """Вероятности вариантов головы.

    uniform — равномерно по маске. Иначе 0.8 · softmax сети + 0.2 · равномерно (пол исследования:
    сеть не может обнулить вариант). Пустая маска — ошибка, а не NaN.
    """
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError('No eligible action')
    base = mask.float() / mask.sum()
    if uniform:
        return base
    learned = logits.masked_fill(~mask, -torch.inf).softmax(0)
    return (1 - EXPLORATION) * learned + EXPLORATION * base


def operator_mask(g):
    """Какие операторы применимы к плану-родителю."""
    mask = torch.ones(len(OPERATORS), dtype=torch.bool)
    # Маршрут «занят», если в нём есть что двигать: назначения в графе — только подвижный хвост.
    busy = g['assignment'].sum(1).gt(0)
    occupied = bool(busy.any())
    mask[ROUTE_SEGMENT] = occupied
    mask[ROUTE_PAIR] = occupied and len(g['engineers']) > 1
    if 'transfer_mask' in g:
        mask[TRANSFER] = bool(g['transfer_mask'].any())
    # Распускать имеет смысл при двух условиях. Занятых маршрутов больше одного —
    # иначе заявкам некуда деться. И в родителе нет неназначенных: если место уже
    # не нашлось, освобождать ещё горсть заявок значит терять третью компоненту
    # cost_tuple ради четвёртой, а она старше. Ровно эта защита стоит в reduce_fleet
    # (`or best.unassigned: break`), и оператор обязан её повторять.
    mask[DISSOLVE_ROUTE] = int(busy.sum()) > 1 and bool(g['assigned'].all())
    if 'locked_routes' in g:
        # Маршрут с закреплённым началом распустить нельзя: исполнитель уже в работе.
        mask[DISSOLVE_ROUTE] = bool(mask[DISSOLVE_ROUTE]) and bool((busy & ~g['locked_routes']).any())
    return mask, busy


def instance_action(choice, g):
    """Выбор сети (индексы узлов графа) → действие над задачей (индексы inst.requests).

    Единственная точка перевода: заявка и якоря звеньев цепочки. Исполнитель и маршрут
    в графе не сжимаются — их индексы и так полные. Утром перевод тождественен.
    """
    ids = g.get('request_ids')
    if ids is None:
        return dict(choice)
    action = dict(choice)
    for key in choice:
        if key == 'request' or (key.startswith('link') and key.endswith('_anchor')):
            action[key] = ids[choice[key]]
    return action


def _last_target(out, action):
    """Проекция последней выбранной цели хода (заявка, иначе маршрут, иначе бригада) — запрос для якоря звена."""
    for key, kind in (('request', 'q'), ('route', 'r'), ('engineer', 'e')):
        if action.get(key) is not None:
            return out['link_proj'][kind][action[key]]
    return None


def sample(out, g, rng, uniform=False, chain_max=1, chain_learned=False, target_potential=0.):
    """Действие для одного кандидата: (словарь действия, log-prob всех выборов, энтропия).

    out — выходы сети по головам (у uniform — нули нужной формы), g — граф с масками.
    uniform — все допустимые варианты равновероятны; target_potential — у uniform доля выбора
    маршрута и заявки пропорционально потенциалу (U2n: то, что выучила сеть b), остальное равномерно.
    chain_max — сколько разрушений накладывать до одного восстановления; chain_learned — длину
    цепи выбирает голова (иначе равномерно). log-prob — сумма по всем выборам хода (полный кредит).
    """
    action = {}
    terms, entropies = [], []
    choices = []                                 # сколько вариантов было у каждого выбора (для энтропии)

    def pick(name, op=None, mask=None, key=None, bias=None):
        # key отделяет ЧТО спрашиваем у сети (name — какая голова) от того, КУДА кладём
        # ответ. Звенья цепочки спрашивают те же головы, но пишутся под своими именами.
        logits = out[name] if op is None else out[name][:, op]
        if bias is not None:
            logits = logits + bias
        p = distribution(logits, mask, uniform)
        if uniform and target_potential and name in g.get('potential_hint', {}):
            # Прицел U2n: «куда бить» — доля target_potential пропорционально потенциалу, остальное равномерно.
            # Обращений к генератору столько же.
            h = g['potential_hint'][name] * (mask if mask is not None else torch.ones_like(p, dtype=torch.bool))
            if float(h.sum()) > 0:
                p = (1 - target_potential) * p + target_potential * h / h.sum()
        i = rng.choices(range(len(p)), weights=p.detach().tolist(), k=1)[0]
        action[key or name] = i
        terms.append(p[i].clamp_min(1e-12).log())
        entropies.append(-(p * p.clamp_min(1e-12).log()).sum() / max(math.log(int((p > 0).sum())), 1.))
        choices.append(int((p > 0).sum()))
        return i

    op_mask, busy = operator_mask(g)
    op = pick('operator', mask=op_mask)
    pick('repair')
    if op not in (TRANSFER, DISSOLVE_ROUTE):
        pick('scale')
    if op in (ROUTE_SEGMENT, ROUTE_PAIR, DISSOLVE_ROUTE):
        if not bool(busy.any()):
            raise ValueError('No occupied routes')
        routes = busy & ~g['locked_routes'] if op == DISSOLVE_ROUTE and 'locked_routes' in g else busy
        route = pick('route', op, routes)
        if op == ROUTE_SEGMENT:
            anchors = g['ownership'][:, route].gt(0)
            pick('request', op, anchors)
        elif op == ROUTE_PAIR:
            others = torch.ones(g['engineers'].shape[0], dtype=torch.bool)
            others[route] = False
            pick('engineer', op, others)
    else:
        transfer = op == TRANSFER and 'transfer_mask' in g
        req = pick('request', op, g['transfer_mask'].any(0) if transfer else None)
        if op == TRANSFER:
            # Допустимость переноса ещё раз проверяет evaluate; эта маска статическая.
            mask = g['transfer_mask'][:, req] if 'transfer_mask' in g else g['compatible'][:, req].gt(0)
            if bool(mask.any()):
                pick('engineer', op, mask)
            else:
                action['no_destination'] = True
    if chain_max > 1:
        if chain_learned:
            # Длина цепи и каждое звено берутся из сети и попадают в градиент. Все звенья
            # тянутся из ОДНОГО прохода: граф и forward считаются раз на эпоху (15 мс)
            # и делятся между кандидатами, а пересборка графа после каждого звена
            # стоила бы ~3.9 с на эпоху. Ограничение намеренное и измеренное, не недосмотр.
            if chain_max > max(CHAINS):
                raise ValueError(f'chain_max={chain_max} больше головы CHAINS={CHAINS}')
            length = CHAINS[pick('chain', mask=torch.tensor([c <= chain_max for c in CHAINS]))]
            action['chain'] = length
            cross = torch.zeros(len(OPERATORS), dtype=torch.bool)
            for extra in CHAIN_EXTRA:
                cross[extra] = True
            # Декодер звеньев: звено видит предыдущее — оператор (матрица переходов), цель (близость к выбранной) и
            # масштаб от своего оператора. У uniform выходов декодера нет — звенья равномерны.
            steps = 'link_op' in out
            prev_op = action['operator']
            prev_vec = _last_target(out, action) if steps else None
            for link in range(length - 1):
                link_op = pick('operator', mask=cross, key=f'link{link}_op',
                               bias=out['link_op'][prev_op] if steps else None)
                anchor = pick('request', link_op, key=f'link{link}_anchor',
                              bias=(out['emb']['q'] @ prev_vec) if steps and prev_vec is not None else None)
                pick('scale', key=f'link{link}_scale', bias=out['link_scale'][link_op] if steps else None)
                prev_op = link_op
                if steps:
                    prev_vec = out['link_proj']['q'][anchor]
        else:
            action['chain'] = rng.randint(1, chain_max)
    logp = torch.stack(terms).sum()
    # Энтропия хода — у самой схлопнутой ручки (из тех, где было из чего выбирать), а не средняя: иначе ручки
    # с десятками вариантов держат среднее, а масштаб и цепочка схлопываются незаметно.
    real = [h for h, c in zip(entropies, choices) if c > 1]
    entropy = torch.stack(real).min() if real else torch.stack(entropies).mean()
    return action, logp, entropy
