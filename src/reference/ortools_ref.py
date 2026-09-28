"""OR-Tools в общих с нашим поиском рамках: те же ограничения, приоритеты — лексикографически.

Пробки и история дня (события) не поддерживаются — задача с ними отклоняется явно, а не молча упрощается.
Время — целые микросекунды (точность datetime), цена расстояния — миллиметры.
Маршруты OR-Tools пересобираются моделью продукта и проверяются тем же независимым валидатором.
"""
import time
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from src.plan.fastroute import day_zero
from src.model import Unassigned
from src.plan.scheduling import static_reason
from src.plan.solver import _rebuild
from src.plan.validate import validate

SCALE = 1_000_000  # integer millimetres per kilometre
TIME_SCALE = 60_000_000  # microseconds per minute


def objective_weights(inst):
    n = len(inst.requests)
    distance_bound = n * max((round(x*SCALE) for row in inst.matrix.km for x in row), default=0)
    vehicle = distance_bound + 1
    drop = len(inst.engineers)*vehicle + distance_bound + 1
    penalties = [drop*(n+1)**r.priority for r in inst.requests]
    if sum(penalties) + len(inst.engineers)*vehicle + distance_bound >= 2**62:
        raise ValueError('Lexicographic objective exceeds safe integer range')
    return vehicle, penalties


def solve(inst, seconds=20., verbose=False, first_solution='PARALLEL_CHEAPEST_INSERTION', metaheuristic='GUIDED_LOCAL_SEARCH'):
    # refactor: Claude, 26.09.2026 — стратегия задаётся (портфель OR-Tools на нескольких ядрах); по умолчанию — прежняя
    began = time.monotonic()
    if inst.matrix.traffic.enabled or inst.matrix.traffic.layers:
        raise ValueError('OR-Tools reference requires shared static travel times')
    if inst.earliest_start is not None or inst.committed or inst.completed:
        raise ValueError('OR-Tools reference supports initial static planning only')
    if seconds <= 0 or not inst.engineers:
        raise ValueError('Positive budget and nonempty engineer pool required')
    reqs, engineers = inst.requests, inst.engineers
    vehicle_cost, penalties = objective_weights(inst)
    end = len(reqs)+1
    manager = pywrapcp.RoutingIndexManager(end+1,len(engineers),[0]*len(engineers),[end]*len(engineers))
    routing = pywrapcp.RoutingModel(manager)
    points = [inst.index['depot']]+[inst.index[r.id] for r in reqs]+[None]
    services = [0]+[round(r.duration_min*TIME_SCALE) for r in reqs]+[0]
    day = day_zero(engineers[0])
    stamp = lambda dt: round((dt-day).total_seconds()*1_000_000)
    horizon = max(stamp(e.shift_end) for e in engineers)
    callbacks=[]
    def distance(a,b):
        i,j=manager.IndexToNode(a),manager.IndexToNode(b)
        return 0 if end in (i,j) else round(inst.matrix.distance(points[i],points[j])*SCALE)
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(distance))
    for eng in engineers:
        table=inst.matrix._plain_matrix(eng.transport)
        def transit(a,b,table=table):
            i,j=manager.IndexToNode(a),manager.IndexToNode(b)
            return services[i]+(0 if end in (i,j) else round(table[points[i]][points[j]]*TIME_SCALE))
        callbacks.append(routing.RegisterTransitCallback(transit))
    routing.AddDimensionWithVehicleTransits(callbacks,horizon,horizon,False,'Time')
    dim=routing.GetDimensionOrDie('Time')
    for v,e in enumerate(engineers):
        dim.CumulVar(routing.Start(v)).SetValue(stamp(e.shift_start))
        dim.CumulVar(routing.End(v)).SetMax(stamp(e.shift_end))
    for node,r in enumerate(reqs,1):
        idx=manager.NodeToIndex(node)
        lo,hi=max(0,stamp(r.window_start)),min(horizon,stamp(r.window_end))
        allowed=[v for v,e in enumerate(engineers) if e.available and static_reason(r,e) is None]
        routing.AddDisjunction([idx],penalties[node-1])
        if not allowed or lo>hi:
            routing.ActiveVar(idx).SetValue(0)
        else:
            routing.VehicleVar(idx).SetValues([-1]+allowed)
            dim.CumulVar(idx).SetRange(lo,hi)
    items=sorted({k for r in reqs for k in r.equipment})
    for number,item in enumerate(items):
        demands=[0]+[r.equipment.get(item,0) for r in reqs]+[0]
        def demand(idx,demands=demands):return demands[manager.IndexToNode(idx)]
        idx=routing.RegisterUnaryTransitCallback(demand)
        routing.AddDimensionWithVehicleCapacity(idx,0,[e.equipment.get(item,0) for e in engineers],True,f'Stock{number}')
    routing.SetFixedCostOfAllVehicles(vehicle_cost)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy=getattr(routing_enums_pb2.FirstSolutionStrategy,first_solution)
    params.local_search_metaheuristic=getattr(routing_enums_pb2.LocalSearchMetaheuristic,metaheuristic)
    remaining=seconds-(time.monotonic()-began)
    if remaining<=0:raise RuntimeError('Budget exhausted building OR-Tools model')
    params.time_limit.FromMilliseconds(max(1,int(remaining*1000)))
    params.log_search=verbose
    solution=routing.SolveWithParameters(params)
    if solution is None:return {'ok':False,'error':'OR-Tools found no solution','seconds':time.monotonic()-began}
    orders={e.id:[] for e in engineers}
    for v,e in enumerate(engineers):
        idx=solution.Value(routing.NextVar(routing.Start(v)))
        while not routing.IsEnd(idx):
            orders[e.id].append(reqs[manager.IndexToNode(idx)-1])
            idx=solution.Value(routing.NextVar(idx))
    assigned={r.id for order in orders.values() for r in order}
    plan=_rebuild(inst,orders,[Unassigned(r.id,'не назначена эталонным решателем','reference') for r in reqs if r.id not in assigned])
    check=validate(plan,inst)
    if not check['ok']:raise RuntimeError(f'Invalid OR-Tools plan: {check}')
    return dict(ok=True,assigned=plan.assigned_count,engineers=plan.used_engineers,km=plan.total_km,
                priority_counts=list(plan.priority_counts),plan=plan,validation=check,
                seconds=time.monotonic()-began,solver_objective=solution.ObjectiveValue())


# refactor: Claude, 26.09.2026 — портфель OR-Tools: решатель маршрутов однопоточный (в параметрах поиска нет числа
# потоков), поэтому на нескольких ядрах — разные стратегии в отдельных процессах, берётся лучший план. Порядок —
# по силе на наших участках (research/ortools_portfolio.py, RESULTS «Портфель OR-Tools на 4 ядрах»).
STRATEGIES = (('PARALLEL_CHEAPEST_INSERTION', 'GUIDED_LOCAL_SEARCH'),
              ('PATH_CHEAPEST_ARC', 'GUIDED_LOCAL_SEARCH'),
              ('PARALLEL_CHEAPEST_INSERTION', 'SIMULATED_ANNEALING'),
              ('LOCAL_CHEAPEST_INSERTION', 'TABU_SEARCH'),
              ('LOCAL_CHEAPEST_INSERTION', 'GUIDED_LOCAL_SEARCH'),
              ('PATH_CHEAPEST_ARC', 'TABU_SEARCH'),
              ('SAVINGS', 'GUIDED_LOCAL_SEARCH'),
              ('PATH_CHEAPEST_ARC', 'SIMULATED_ANNEALING'))


def _solve_one(inst, seconds, strategy, weights):
    if weights is not None:
        weights.apply()            # цель в этом процессе — как у сверки (стоимость пересобранного плана)
    return solve(inst, seconds, first_solution=strategy[0], metaheuristic=strategy[1])


def solve_portfolio(inst, seconds, processes, weights=None):
    """processes разных стратегий параллельно (не больше, чем стратегий), лучший по [аварии, подключения, заявки,
    бригады, км]. Возвращает то же, что solve, плюс число процессов и итог каждой стратегии."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    chosen = STRATEGIES[:max(1, min(processes, len(STRATEGIES)))]
    began = time.monotonic()
    with ProcessPoolExecutor(len(chosen), mp_context=multiprocessing.get_context('spawn')) as pool:
        outs = list(pool.map(_solve_one, [inst] * len(chosen), [seconds] * len(chosen), chosen, [weights] * len(chosen)))
    ok = [o for o in outs if o['ok']]
    if not ok:
        return {'ok': False, 'error': outs[0]['error'], 'seconds': time.monotonic() - began, 'processes': len(chosen)}
    key = lambda o: (-o['priority_counts'][0], -o['priority_counts'][1], -o['assigned'], o['engineers'], o['km'])
    best = min(ok, key=key)
    return {**best, 'seconds': time.monotonic() - began, 'processes': len(chosen),
            'strategies': [{'strategy': '+'.join(st), 'ok': o['ok'], 'engineers': o.get('engineers'),
                            'km': round(o['km'], 1) if o['ok'] else None} for st, o in zip(chosen, outs)]}
