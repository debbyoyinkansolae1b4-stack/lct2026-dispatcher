"""Пространство действий популяционного поиска и само действие.

Действие — что сделать с планом-родителем: какой оператор разрушения, насколько
крупно, вокруг какого маршрута или заявки, с какой глубиной регрета восстанавливать
и сколько поперечных разрушений добавить цепочкой. Сущности заданы индексами
в inst.engineers и inst.requests — так их выбирают головы сети, и так они пишутся
в журнал прогона. Нумерацию менять нельзя: эталон и сохранённые журналы на неё опираются.
"""
# refactor: Claude, 21.09.2026 — константы из src/alt/population_gnn.py, действие вместо словаря
from dataclasses import dataclass, field

OPERATORS = ('route_segment', 'route_pair', 'geographic', 'time_window', 'transfer', 'dissolve_route')
# Номера операторов для кода, который работает с индексами голов сети.
ROUTE_SEGMENT, ROUTE_PAIR, GEOGRAPHIC, TIME_WINDOW, TRANSFER, DISSOLVE_ROUTE = range(len(OPERATORS))
SCALES = (.15, .4, .8)
# Добавочные звенья цепочки берём только из поперечных разрушений. Замер показал,
# почему: внутримаршрутные route_segment/route_pair дают 64-87% пустых ходов —
# соседи не изменились, и жадное восстановление кладёт заявку на прежнее место.
# geographic/time_window снимают заявки у разных исполнителей сразу и откатываются
# в 3-13% случаев, то есть именно они создают ту дисперсию, ради которой цепочка нужна.
CHAIN_EXTRA = (GEOGRAPHIC, TIME_WINDOW)
CHAINS = (1, 2, 3, 4)       # сколько разрушений накладывать до одного восстановления


@dataclass(frozen=True)
class Link:
    """Добавочное звено цепочки: поперечный оператор, заявка-якорь, масштаб."""
    operator: int
    anchor: int
    scale: int


@dataclass(frozen=True)
class Action:
    operator: int
    repair: int
    scale: int | None = None
    route: int | None = None
    request: int | None = None
    engineer: int | None = None
    no_destination: bool = False    # перенос выбран, но заявке некуда ехать
    chain: int | None = None        # длина цепочки; None — без цепочки
    # Звенья, выбранные сетью. Если chain задан, а звеньев меньше chain-1, недостающие
    # тянутся случайно при мутации — так работает цепочка без обучения.
    links: tuple[Link, ...] = field(default_factory=tuple)

    @classmethod
    def of(cls, value):
        """Действие из самого себя или из словаря журнала (ключи link{i}_op/_anchor/_scale)."""
        if isinstance(value, cls):
            return value
        data = dict(value)
        links = []
        while f'link{len(links)}_op' in data:
            i = len(links)
            links.append(Link(data.pop(f'link{i}_op'), data.pop(f'link{i}_anchor'), data.pop(f'link{i}_scale')))
        unknown = set(data) - {'operator', 'repair', 'scale', 'route', 'request', 'engineer', 'no_destination', 'chain'}
        if unknown:
            raise ValueError(f'неизвестные поля действия: {sorted(unknown)}')
        return cls(links=tuple(links), **data)

    def as_dict(self):
        """Словарь в формате журнала прогона: только заданные поля."""
        out = {'operator': self.operator, 'repair': self.repair}
        for name in ('scale', 'route', 'request', 'engineer'):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.no_destination:
            out['no_destination'] = True
        if self.chain is not None:
            out['chain'] = self.chain
        for i, link in enumerate(self.links):
            out.update({f'link{i}_op': link.operator, f'link{i}_anchor': link.anchor, f'link{i}_scale': link.scale})
        return out
