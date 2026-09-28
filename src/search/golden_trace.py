"""Трасса эталона рефакторинга (REFACTOR-PLAN.md, шаг 0): хеши состояния на границе эпохи.

Только чтение: ни одного обращения к генераторам случайных чисел и ни одного лишнего
обратного прохода. Её вызывают и старый цикл experiment_population_gnn.py, и новый,
поэтому живёт она здесь, а не в одном из них.
"""
import dataclasses
import hashlib
from datetime import datetime

import torch


def tensor_digest(value):
    """Хеш вложенной структуры с тензорами.

    Кодирование однозначное: у каждого элемента метка типа и длина содержимого,
    у контейнера — число элементов. Без этого [1, 23], [12, 3] и [1, 2, 3] давали бы
    одни и те же байты (ошибка первой версии, найдена Codex при проверке шага 0).

    Неизвестный объект — ошибка, а не repr: repr с адресом в памяти дал бы хеш,
    разный между запусками, и эталон начал бы ложно расходиться.
    """
    h = hashlib.sha256()

    def put(tag, payload=b''):
        h.update(tag + len(payload).to_bytes(8, 'little') + payload)

    def walk(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            put(b'T', f'{t.dtype}{tuple(t.shape)}'.encode())
            put(b'B', t.numpy().tobytes())
        elif isinstance(x, dict):
            put(b'D', str(len(x)).encode())
            for key in sorted(x, key=repr):
                walk(key)
                walk(x[key])
        elif isinstance(x, (list, tuple)):
            put(b'L' if isinstance(x, list) else b'U', str(len(x)).encode())
            for item in x:
                walk(item)
        elif x is None:
            put(b'N')
        elif isinstance(x, bool):                      # раньше int: bool — его подкласс
            put(b'b', b'1' if x else b'0')
        elif isinstance(x, int):
            put(b'i', str(x).encode())
        elif isinstance(x, float):
            put(b'f', x.hex().encode())               # точное значение, без округления repr
        elif isinstance(x, str):
            put(b's', x.encode())
        elif isinstance(x, datetime):
            put(b't', x.isoformat().encode())
        elif dataclasses.is_dataclass(x):             # Stop, Unassigned: по полям, не по адресу
            put(b'C', type(x).__name__.encode())
            walk({f.name: getattr(x, f.name) for f in dataclasses.fields(x)})
        else:
            raise TypeError(f'golden trace: не знаю, как хешировать {type(x).__name__}')

    walk(value)
    return h.hexdigest()[:16]


def plan_digest(plan):
    """План целиком: исполнители, порядок и времена визитов, неназначенные с причинами."""
    routes = [(eid, [(s.request_id, s.arrive.isoformat(), s.start.isoformat(), s.end.isoformat(),
                      s.travel_min, s.travel_km) for s in route.stops])
              for eid, route in plan.routes.items()]
    unassigned = [(u.request_id, u.reason_code, u.reason) for u in plan.unassigned]
    return tensor_digest(dict(routes=routes, unassigned=unassigned, cost=list(plan.cost_tuple())))


def gradient_digest(net):
    """Уже вычисленные .grad по именам; отсутствующий градиент явно как None."""
    return tensor_digest([(name, p.grad) for name, p in net.named_parameters()])


def golden_trace(rng, net, opt, seeds, numpy_start, epoch_parts):
    """Состояние на границе эпохи: сиды, генераторы, сеть, Adam и части, собранные циклом."""
    import numpy as np
    short = lambda data: hashlib.sha256(data).hexdigest()[:16]
    network = {}
    if net is not None:                      # у политики без сети нет весов и Adam
        names = {id(p): name for name, p in net.named_parameters()}
        adam = dict(
            groups=[{k: v for k, v in group.items() if k != 'params'}
                    | dict(params=[names[id(p)] for p in group['params']]) for group in opt.param_groups],
            state={names[id(p)]: dict(state) for p, state in opt.state.items()})
        network = dict(state=tensor_digest(net.state_dict()),
                       parameter_order=tensor_digest([(name, str(p.dtype), tuple(p.shape), p.requires_grad)
                                                      for name, p in net.named_parameters()]),
                       adam=tensor_digest(adam))
    return dict(
        seeds=list(seeds),
        python_rng=short(repr(rng.getstate()).encode()),
        torch_rng=short(torch.get_rng_state().numpy().tobytes()),
        # Глобальный numpy не задаётся сидом и берёт затравку от системы, поэтому его хеш
        # между процессами разный. Эталону важно другое — что его никто не трогал.
        numpy_rng_used=repr(np.random.get_state()) != numpy_start,
        state=network.get('state'), parameter_order=network.get('parameter_order'), adam=network.get('adam'),
        **epoch_parts)
