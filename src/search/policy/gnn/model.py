"""Сеть политики: базовая PopulationGNN и PlannerPopulationGNN (обмен сообщениями GATv2, как в Planner).

Порядок создания слоёв в конструкторах — часть контракта: от него зависят начальные
веса (расход генератора torch), порядок параметров и обрезка градиента по группам.
Эталон шага 0 сверяет веса и Adam побайтово.
"""
# refactor: Claude, 21.09.2026 — из src/alt/population_gnn.py и planner_population.py; стиль, логика без изменений
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv

from src.search.actions import CHAINS, OPERATORS
from src.search.policy.gnn.graph import RELATIONS

# Версия схемы признаков и голов: сохранённые веса грузятся только при совпадении (weights_io).
# v5 (22.09): вход нормируется по задаче (RunningNorm) вместо FEATURE_STATS — веса v4 учились
# на другом масштабе входа, формы те же; поднятая версия отклоняет их (Н61).
# v6 (23.09): седьмая компонента цели (аварии сверх ориентира) — разрыв до рекорда на признак шире.
SCHEMA = 'population-heterogeneous-v6'

HEADS = ('operator', 'scale', 'repair', 'chain', 'engineer_head', 'route_head', 'request_head')
MESSAGES = ('request_route', 'route_request', 'engineer_route', 'route_engineer',
            'engineer_request', 'request_engineer', 'request_request')


class PopulationGNN(nn.Module):
    """Кодировщик графа плана и головы действия: оператор, масштаб, регрет, цепочка, сущности."""
    # Ширины входа графа без приписанных после enrich_graph признаков (потенциал, переселение): с «зрением цели»
    # (перенос 26.09) — маршрут +1 (аварии сверх ориентира), заявка +4 (ожидание, флаг «> ориентира», запас, крюк),
    # общее +1 (аварии сверх ориентира всего).
    ROUTE_FEATURES = 32
    GLOB_FEATURES = 9
    REQUEST_FEATURES = 23

    def __init__(self, hidden=48, route_features=None, glob_features=None, use_gates=False, rich_pooling=False,
                 n_experts=3, request_features=None):
        super().__init__()
        self.hidden = hidden
        self.use_gates = use_gates
        self.glob_features = glob_features or self.GLOB_FEATURES
        # Ширина входа маршрута параметр, а не константа: признаки, приписанные после
        # encode() (например rehomable_share из enrich_graph), расширяют её.
        self.route_features = route_features or self.ROUTE_FEATURES
        # Ширина входа заявки тоже параметр: крюк добавляется двадцатым признаком.
        self.request_features = request_features or self.REQUEST_FEATURES
        self.engineer_in = nn.Linear(14, hidden)
        self.route_in = nn.Linear(self.route_features, hidden)
        self.request_in = nn.Linear(self.request_features, hidden)
        # По одному обучаемому числу на признак. 2*sigmoid(0)=1, то есть на старте
        # гейт тождественен его отсутствию; дальше может придавить к нулю или усилить вдвое.
        self.gates = nn.ParameterDict({k: nn.Parameter(torch.zeros(n)) for k, n in
                                       (('engineers', 14), ('routes', self.route_features),
                                        ('requests', self.request_features), ('glob', self.glob_features))})
        self.messages = nn.ModuleList([nn.ModuleDict({k: nn.Linear(hidden, hidden) for k in MESSAGES})
                                       for _ in range(2)])
        self.norms = nn.ModuleList([nn.ModuleDict({k: nn.LayerNorm(hidden) for k in ('e', 'r', 'q')})
                                    for _ in range(2)])
        # При rich_pooling узлы сворачиваются не только средним, но ещё разбросом
        # и максимумом: два плана с одинаковой средней загрузкой маршрутов иначе
        # неотличимы, хотя в одном все ровные, а в другом один перегружен.
        self.rich_pooling = rich_pooling
        width = hidden * 3 * (3 if rich_pooling else 1) + self.glob_features
        # Число экспертов — параметр: критерий приёмки MMoE — «не вредит», и надо уметь
        # сравнить три эксперта с одним.
        self.n_experts = n_experts
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width, hidden), nn.SiLU()) for _ in range(n_experts)])
        self.gate_action = nn.Linear(width, n_experts)
        self.gate_value = nn.Linear(width, n_experts)
        self.operator = nn.Linear(hidden, len(OPERATORS))
        self.scale = nn.Linear(hidden, 3)
        self.repair = nn.Linear(hidden, 6)      # патч learned_full: regret-1/2/3 × {плотная, свободная}
        self.chain = nn.Linear(hidden, len(CHAINS))
        self.engineer_head = nn.Linear(hidden, len(OPERATORS))
        self.route_head = nn.Linear(hidden, len(OPERATORS))
        self.request_head = nn.Linear(hidden, len(OPERATORS))
        self.value = nn.Linear(hidden, 1)
        # Патч learned_full (Михаил 25.09): цепочка — декодер по шагам без пересборки графа. Оператор звена зависит от
        # оператора предыдущего (матрица переходов), якорь звена — от предыдущей выбранной цели (проекция векторов
        # узлов), масштаб звена — от его оператора. Всё с нуля: на старте поведение прежнее.
        self.link_op = nn.Parameter(torch.zeros(len(OPERATORS), len(OPERATORS)))
        self.link_scale = nn.Parameter(torch.zeros(len(OPERATORS), 3))
        self.link_query = nn.Linear(hidden, hidden)
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.log_alpha = nn.Parameter(torch.tensor(-2.))
        # Отдельная температура для гейта экспертов: softmax по экспертам насыщался
        # на эпохе 5-9 из 150 в 29 прогонах из 30, две трети ёмкости умирали.
        self.log_beta = nn.Parameter(torch.tensor(-2.))
        # Масштаб градиента критика в общих слоях. None — через отдельный обратный
        # проход; число — тот же эффект в прямом проходе.
        self.critic_scale = None
        # Малые ненулевые веса: почти равномерно, но градиент кодировщика идёт с первого шага.
        for name in HEADS:
            head = getattr(self, name)
            nn.init.normal_(head.weight, std=.01)
            nn.init.zeros_(head.bias)

    def check_widths(self, g):
        """Ширина входа против сети — с понятным сообщением, а не падением внутри nn.Linear."""
        if g['routes'].shape[-1] != self.route_features:
            raise ValueError(f"routes: признаков {g['routes'].shape[-1]}, сеть ждёт {self.route_features}")
        if g['glob'].shape[-1] != self.glob_features:
            raise ValueError(f"glob: признаков {g['glob'].shape[-1]}, сеть ждёт {self.glob_features}")

    def pool(self, x):
        """Свёртка узлов в вектор фиксированной длины: среднее, при rich_pooling ещё разброс и максимум."""
        if not self.rich_pooling:
            return x.mean(0)
        spread = x.std(0) if x.shape[0] > 1 else torch.zeros_like(x[0])
        return torch.cat([x.mean(0), spread, x.amax(0)])

    def gated(self, g, key):
        x = g[key]
        return x * (2 * torch.sigmoid(self.gates[key])) if self.use_gates else x

    def expert_report(self):
        """Насколько выходы экспертов различны: косинусное расстояние 0 — эксперты одинаковы."""
        x = getattr(self, 'last_experts', None)
        if x is None:
            return {}
        flat = x.reshape(x.shape[0], -1)
        norms = flat.norm(dim=1)
        pairs = []
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                denom = (norms[i] * norms[j]).clamp_min(1e-12)
                pairs.append(float(1 - (flat[i] @ flat[j]) / denom))
        return dict(cosine_distance=pairs, mean_cosine_distance=sum(pairs) / max(len(pairs), 1),
                    norms=[round(float(v), 4) for v in norms])

    def gate_report(self):
        """Значения гейта признаков как есть: сколько сеть оставила от каждого признака."""
        return {k: [round(float(x), 4) for x in (2 * torch.sigmoid(v.detach()))] for k, v in self.gates.items()}

    def heads(self, e, r, q, act, val, gate_action, gate_value):
        return dict(operator=self.operator(act), scale=self.scale(act), repair=self.repair(act),
                    chain=self.chain(act), engineer=self.engineer_head(e + act), route=self.route_head(r + act),
                    request=self.request_head(q + act), value=self.value(val).squeeze(),
                    gate_action=gate_action, gate_value=gate_value)

    def forward(self, g):
        self.check_widths(g)
        e = self.engineer_in(self.gated(g, 'engineers'))
        r = self.route_in(self.gated(g, 'routes'))
        q = self.request_in(self.gated(g, 'requests'))
        for m, n in zip(self.messages, self.norms):
            en = n['e'](e + m['route_engineer'](r) + m['request_engineer'](g['compatible'] @ q))
            rn = n['r'](r + m['engineer_route'](e) + m['request_route'](g['assignment'] @ q))
            qn = n['q'](q + m['route_request'](g['ownership'] @ r) + m['engineer_request'](g['reverse_compatible'] @ e)
                        + m['request_request'](g['neighbors'] @ q))
            e, r, q = en, rn, qn
        state = torch.cat([self.pool(e), self.pool(r), self.pool(q), self.gated(g, 'glob')])
        experts = torch.stack([x(state) for x in self.experts])
        self.last_experts = experts.detach()
        act = (self.gate_action(state).softmax(0)[:, None] * experts).sum(0)
        # Критик достаёт до общих слоёв двумя путями: через выходы экспертов и через
        # вход гейта значения (тот же state). Масштабировать надо оба, иначе
        # эквивалентности нет — это поймал tests/test_critic_scaling_parity.py.
        k = self.critic_scale
        scaled = experts if k is None else k * experts + (1 - k) * experts.detach()
        state_v = state if k is None else k * state + (1 - k) * state.detach()
        val = (self.gate_value(state_v).softmax(0)[:, None] * scaled).sum(0)
        return self.heads(e, r, q, act, val, self.gate_action(state).softmax(0), self.gate_value(state).softmax(0))


class PlannerPopulationGNN(PopulationGNN):
    """Та же сеть с обменом сообщениями GATv2 по типизированным связям (RELATIONS), как в Planner."""

    def __init__(self, hidden=256, route_features=None, glob_features=None, use_gates=False, balanced_gates=False,
                 rich_pooling=False, n_experts=3, request_features=None):
        super().__init__(hidden, route_features, glob_features, use_gates, rich_pooling, n_experts, request_features)
        self.messages = nn.ModuleList([HeteroConv({rel: GATv2Conv((hidden, hidden), hidden, heads=4, concat=False,
                                                                  add_self_loops=False, edge_dim=4)
                                                   for rel in RELATIONS}, aggr='sum') for _ in range(2)])
        # Головы с нулей: старт ровно равномерный, остаточные связи сохраняют изолированные узлы.
        for name in HEADS:
            head = getattr(self, name)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.link_query.weight)
        nn.init.zeros_(self.link_query.bias)
        if balanced_gates:
            # Гейты MMoE были единственными головами без зануления: при ширине входа 776
            # и обычной инициализации softmax уходит в насыщение за несколько шагов.
            for gate in (self.gate_action, self.gate_value):
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
        self.log_alpha.data.zero_()

    def forward(self, g):
        self.check_widths(g)
        x = {'e': self.engineer_in(self.gated(g, 'engineers')), 'r': self.route_in(self.gated(g, 'routes')),
             'q': self.request_in(self.gated(g, 'requests'))}
        eye = torch.eye(len(x['e']))
        # Запасные рёбра из плотных матриц — если enrich_graph не строил свои.
        matrices = [eye, eye, g['assignment'], g['ownership'], g['compatible'], g['reverse_compatible'],
                    g['neighbors'], torch.zeros_like(g['neighbors']), torch.zeros_like(g['neighbors'])]
        edges, attrs = {}, {}
        for rel, matrix in zip(RELATIONS, matrices):
            dst, src = matrix.nonzero(as_tuple=True)
            edges[rel] = torch.stack([src, dst])
            attrs[rel] = matrix[dst, src, None].expand(-1, 4)
        edges = g.get('edge_index', edges)
        attrs = g.get('edge_attr', attrs)
        for conv, norm in zip(self.messages, self.norms):
            y = conv(x, edges, edge_attr_dict=attrs)
            x = {k: norm[k](x[k] + y[k]).relu() for k in x}
        e, r, q = x['e'], x['r'], x['q']
        state = torch.cat([self.pool(e), self.pool(r), self.pool(q), self.gated(g, 'glob')])
        experts = torch.stack([expert(state) for expert in self.experts])
        self.last_experts = experts.detach()
        k = self.critic_scale
        state_v = state if k is None else k * state + (1 - k) * state.detach()
        ga = self.gate_action(state).softmax(0)
        gv = self.gate_value(state_v).softmax(0)
        act = (ga[:, None] * experts).sum(0)
        # Множитель критика — на выходах экспертов, а не на val: gate_value получает полный
        # градиент, всё выше экспертов — ослабленный.
        val = (gv[:, None] * (experts if k is None else k * experts + (1 - k) * experts.detach())).sum(0)
        out = self.heads(e, r, q, act, val, ga, gv)
        # Для декодера звеньев: векторы узлов и их проекции (запрос «что рядом с выбранным»).
        out['emb'] = dict(q=q, r=r, e=e)
        out['link_proj'] = dict(q=self.link_query(q), r=self.link_query(r), e=self.link_query(e))
        out['link_op'], out['link_scale'] = self.link_op, self.link_scale
        return out
