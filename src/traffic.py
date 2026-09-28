"""Загруженность дорог: во сколько раз дорога дольше обычного в это время.

Вынесено из travel.py отдельно: расстояния и скорости режимов — про географию,
а пробки — про время суток, и смешивать их в одном классе значит делать оба
непонятными.

Два уровня, второй важнее первого:

  профиль по часам — один коэффициент на весь город. Дёшево, лучше чем ничего,
    но не различает въезд в центр и выезд из него;
  слои матрицы — отдельная матрица времён на каждый интервал, у каждой пары точек
    свой профиль. Это то, что заполняется настоящим источником: платным матричным
    сервисом с departure_time, историей поездок или прогнозом.

Планировщик ни о том, ни о другом не знает: он спрашивает время в пути на момент
выезда, а откуда оно взялось — дело этого модуля.
"""


class TrafficModel:
    def __init__(self, config: dict | None = None, weekday: int = 0):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled"))
        self.hourly = cfg.get("hourly", [1.0] * 24)
        self.weekday_factor = cfg.get("weekday", {})
        self.sensitivity = cfg.get("mode_sensitivity", {})
        self.weekday = weekday
        self.layers: dict[int, list[list[float]]] = {}
        self.layer_step = 60

    # --- уровень 1: коэффициент по часу и дню недели ---

    def factor(self, depart_min: float | None, mode: str) -> float:
        if not self.enabled or depart_min is None:
            return 1.0
        sensitivity = self.sensitivity.get(mode, 1.0)
        if sensitivity <= 0:
            return 1.0
        hour = int(depart_min // 60) % 24
        base = self.hourly[hour] * self.weekday_factor.get(str(self.weekday), 1.0)
        return 1.0 + (base - 1.0) * sensitivity

    # --- уровень 2: готовые времена из внешнего источника ---

    def load_layers(self, layers: dict[int, list[list[float]]], step_min: int = 60,
                    expected_size: int | None = None) -> None:
        for start, matrix in layers.items():
            if expected_size and (len(matrix) != expected_size or len(matrix[0]) != expected_size):
                raise ValueError(f"слой {start}: размер {len(matrix)} против {expected_size} точек")
        self.layers = dict(layers)
        self.layer_step = step_min

    def layer_minutes(self, i: int, j: int, depart_min: float | None) -> float | None:
        """Время из слоя, если источник подключён и слой на этот интервал есть."""
        if not self.layers or depart_min is None:
            return None
        bucket = (int(depart_min // self.layer_step) * self.layer_step) % (24 * 60)
        layer = self.layers.get(bucket)
        if layer is None or i >= len(layer) or j >= len(layer):
            return None           # точка добавлена событием после загрузки слоёв — её время считается по профилю
        return layer[i][j]

    def describe(self) -> str:
        if self.layers:
            return f"слои по {self.layer_step} мин ({len(self.layers)} интервалов)"
        return "профиль по часам" if self.enabled else "без учёта загруженности"
