"""Перенос весов между прогонами с явной проверкой совместимости.

Веса имеют смысл только вместе со схемой признаков и шириной голов: загруженная
не в ту архитектуру сеть не падает, а молча считает чушь. Поэтому проверяем схему
и форму каждого параметра, а не только имена.
"""
import torch


def save_weights(path, net, schema, config):
    torch.save(dict(schema=schema, state=net.state_dict(), config=config), path)


def load_weights(path, net, schema):
    """Загружает веса в сеть. Возвращает описание источника для протокола прогона."""
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if 'state' not in blob:
        raise ValueError(f'{path}: в файле нет весов (state). Прогон сохранял только результат?')
    if blob.get('schema') != schema:
        raise ValueError(f'{path}: схема признаков не та. В файле {blob.get("schema")}, ожидается {schema}')
    current = net.state_dict()
    missing = sorted(set(current) - set(blob['state']))
    extra = sorted(set(blob['state']) - set(current))
    if missing or extra:
        raise ValueError(f'{path}: другой набор параметров. Нет {missing[:3]}, лишние {extra[:3]}')
    bad = [k for k, v in blob['state'].items() if tuple(v.shape) != tuple(current[k].shape)]
    if bad:
        raise ValueError(f'{path}: другая ширина у {bad[:3]} — признаки или головы отличаются')
    net.load_state_dict(blob['state'])
    source = blob.get('config', {})
    return dict(path=str(path), trained_epochs=source.get('epochs'), trained_on=source.get('bundle'),
                across_tasks=source.get('across_tasks'), trajectories=source.get('trajectories'))
