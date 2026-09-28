"""Полная сверка двух входных наборов: где они расходятся.

Сборщик нового набора вызывает её против прежнего и падает на любом отличии,
кроме явно разрешённых полей. Сравнение рекурсивное по всем атрибутам объектов,
словарям и последовательностям, поэтому новое поле в Instance или матрице
не проскочит мимо проверки.

Вычисляемые кэши (`_window_min`, `_mode_cache`) не сравниваются: они заполняются
при первом обращении и зависят от того, что успели посчитать до сохранения.
"""

CACHES = frozenset({'_window_min', '_mode_cache'})


def differences(old, new, allowed=frozenset(), path='', out=None, seen=None):
    """Пути, по которым old и new различаются, кроме полей из allowed и кэшей."""
    out = [] if out is None else out
    seen = set() if seen is None else seen
    key = (id(old), id(new))
    if key in seen:
        return out
    if hasattr(old, '__dict__') and not isinstance(old, type):
        seen.add(key)
        if type(old) is not type(new):
            out.append(f'{path}: тип {type(old).__name__} -> {type(new).__name__}')
            return out
        a, b = vars(old), vars(new)
        for name in sorted(set(a) | set(b)):
            if name in CACHES or name in allowed:
                continue
            if name not in a or name not in b:
                out.append(f'{path}.{name}: поле есть только в одном наборе')
                continue
            differences(a[name], b[name], allowed, f'{path}.{name}', out, seen)
    elif isinstance(old, dict) and isinstance(new, dict):
        if list(old) != list(new):
            out.append(f'{path}: ключи или их порядок')
        for name in old:
            if name in new and name not in allowed:
                differences(old[name], new[name], allowed, f'{path}[{name!r}]', out, seen)
    elif isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        if type(old) is not type(new) or len(old) != len(new):
            out.append(f'{path}: длина {len(old)} -> {len(new)}')
            return out
        for i, (x, y) in enumerate(zip(old, new)):
            differences(x, y, allowed, f'{path}[{i}]', out, seen)
    elif old != new:
        out.append(f'{path}: {old!r} -> {new!r}'[:200])
    return out


def require_same(old, new, allowed=frozenset(), label='набор'):
    found = differences(old, new, frozenset(allowed))
    if found:
        shown = '\n  '.join(found[:20])
        raise SystemExit(f'{label}: расхождения сверх разрешённых {sorted(allowed)}:\n  {shown}'
                         + (f'\n  … и ещё {len(found) - 20}' if len(found) > 20 else ''))
