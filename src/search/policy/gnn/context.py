"""Контекст для графа сети: признаки плана (features.py) плюс пространственный и временной запас маршрутов и заявок."""
import torch
# refactor: Claude, 22.09.2026 — шаг 8а: кодировщик из features.py; ContextOperatorNet (сеть ALNS) удалён
from src.search.policy.gnn.features import encode

FEATURE_SCHEMA = 'spatial-slack-raw-v1'
ROUTE_CONTEXT = ('window_slack_min', 'window_slack_mean', 'shift_slack', 'waiting',
                 'latitude_offset', 'longitude_offset', 'spatial_spread', 'max_leg')
REQUEST_CONTEXT = ('latitude_offset', 'longitude_offset')


def encode_context(inst, plan, iteration=0, iterations=200, no_improve=0):
    routes, requests, glob = encode(inst, plan, iteration, iterations, no_improve)
    lat0, lon0 = inst.matrix.points[inst.index['depot']]
    rows=[]
    for eng in inst.engineers:
        route=plan.routes[eng.id]
        reqs=[inst.by_id[s.request_id] for s in route.stops]
        slack=[max(0.,(r.window_end-s.start).total_seconds()/60) for r,s in zip(reqs,route.stops)]
        coords=[(r.lat-lat0,r.lon-lon0) for r in reqs if r.geocoded]
        x=sum(a for a,b in coords)/max(len(coords),1)
        y=sum(b for a,b in coords)/max(len(coords),1)
        spread=sum((a-x)**2+(b-y)**2 for a,b in coords)/max(len(coords),1)
        end=route.stops[-1].end if route.stops else eng.shift_start
        # Сырые минуты и километры: масштаб задаёт нормировка, а не подобранные делители.
        rows.append([min(slack,default=0),sum(slack)/max(len(slack),1),
                     max(0.,(eng.shift_end-end).total_seconds()/60),
                     sum((s.start-s.arrive).total_seconds()/60 for s in route.stops),
                     x,y,spread**.5,max((s.travel_km for s in route.stops),default=0)])
    extra=[]
    for u in plan.unassigned:
        r=inst.by_id.get(u.request_id)
        if r is not None:
            extra.append([r.lat-lat0,r.lon-lon0] if r.geocoded else [0.,0.])
    return (torch.cat((routes,torch.tensor(rows,dtype=torch.float32).reshape(-1,len(ROUTE_CONTEXT))),1),
            torch.cat((requests,torch.tensor(extra,dtype=torch.float32).reshape(-1,len(REQUEST_CONTEXT))),1),glob)
