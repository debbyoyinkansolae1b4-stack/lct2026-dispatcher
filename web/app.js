// Число со словом в нужном падеже: 1 заявка, 2 заявки, 5 заявок. refactor: Claude, 26.09.2026
function plural(n, one, few, many) {
  const a = Math.abs(n) % 100, b = a % 10;
  return a > 10 && a < 20 ? many : b === 1 ? one : b >= 2 && b <= 4 ? few : many;
}
const count = (n, one, few, many) => `${n} ${plural(n, one, few, many)}`;
const signed = (n, one, few, many) => `${n > 0 ? "+" : ""}${count(n, one, few, many)}`;

const COLORS = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#008080","#9a6324",
                "#800000","#808000","#000075","#f032e6","#469990","#bfa100","#666666"];
const $ = (id) => document.getElementById(id);

let map, layers = [], state = null, pickedPoint = null, pickMarker = null;
let routeLines = {}, routeMarkers = {}, routeArrows = {}, activeRoute = null, dataset = null;

function initMap() {
  map = L.map("map", { zoomControl: true }).setView([55.72, 37.64], 11);
  // Свой префикс вместо стандартного: в нём ссылка на Leaflet и флаг, которым тут не место.
  // Указание на OpenStreetMap оставляем — этого требует лицензия данных ODbL.
  map.attributionControl.setPrefix("");
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 19, attribution: "Данные © OpenStreetMap" }).addTo(map);
  map.on("click", (e) => {
    if ($("eventKind").value !== "urgent") return;
    pickedPoint = e.latlng;
    if (pickMarker) map.removeLayer(pickMarker);
    pickMarker = L.marker(e.latlng, { title: "новая срочная заявка" }).addTo(map)
      .bindPopup("Новая срочная заявка здесь").openPopup();
    $("eventHint").textContent = `Точка выбрана: ${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}`;
  });
}

function clearMap() {
  layers.forEach((l) => map.removeLayer(l));
  layers = [];
  routeLines = {}; routeMarkers = {}; routeArrows = {}; activeRoute = null;
}

// Азимут между двумя точками в градусах — на него поворачивается шеврон
function bearing(a, b) {
  const rad = Math.PI / 180;
  const y = Math.sin((b[1] - a[1]) * rad) * Math.cos(b[0] * rad);
  const x = Math.cos(a[0] * rad) * Math.sin(b[0] * rad)
          - Math.sin(a[0] * rad) * Math.cos(b[0] * rad) * Math.cos((b[1] - a[1]) * rad);
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

// Стрелки вдоль маршрута: номера показывают порядок, но направление между точками
// глазами не читается, особенно когда линия идёт по дорогам и петляет
function addDirectionArrows(points, color, engineerId) {
  if (points.length < 2) return [];
  const spans = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const d = map.distance(points[i - 1], points[i]);
    spans.push(d);
    total += d;
  }
  if (!total) return [];

  const count = Math.min(8, Math.max(3, Math.round(points.length / 12)));
  const step = total / (count + 1);
  const arrows = [];
  let target = step, walked = 0;

  for (let i = 1; i < points.length && arrows.length < count; i++) {
    walked += spans[i - 1];
    while (walked >= target && arrows.length < count) {
      const angle = bearing(points[i - 1], points[i]);
      const icon = L.divIcon({
        className: "", iconSize: [14, 14], iconAnchor: [7, 7],
        html: `<div class="route-arrow" style="color:${color};transform:rotate(${angle - 90}deg)">➤</div>`,
      });
      const marker = L.marker(points[i], { icon, interactive: false,
                                           keyboard: false, zIndexOffset: -200 }).addTo(map);
      arrows.push(marker);
      layers.push(marker);
      target += step;
    }
  }
  routeArrows[engineerId] = arrows;
  return arrows;
}

// Значки карты и легенда к ним — из одного описания, чтобы легенда не разошлась с картой.
// refactor: Claude, 26.09.2026 — легенда видна и до плана; неназначенная отличается от аварии (просьба Михаила)
const MARK = {
  depot: { radius: 9, color: "#1c1d22", weight: 3, fillColor: "#fff", fillOpacity: 1 },
  urgent: { radius: 7, color: "#b45309", weight: 2, fillColor: "#fde68a", fillOpacity: .9 },
  request: { radius: 5, color: "#8a8f98", weight: 2, fillColor: "#d7dae0", fillOpacity: .9 },
  unassigned: { radius: 7, color: "#ff0053", weight: 2, dashArray: "3 3", fillColor: "#fff", fillOpacity: .95 },
};

function swatch(m) {
  const r = m.radius, size = 2 * r + 2 * m.weight;
  return `<svg width="${size}" height="${size}" aria-hidden="true"><circle cx="${size / 2}" cy="${size / 2}" r="${r}"
    fill="${m.fillColor}" stroke="${m.color}" stroke-width="${m.weight}" ${m.dashArray ? `stroke-dasharray="${m.dashArray}"` : ""}/></svg>`;
}

function showLegend(planned) {
  const row = (icon, text) => `<div class="lg-row"><span class="lg-icon">${icon}</span><span>${text}</span></div>`;
  const pin = (urgent) => `<span class="stop-pin${urgent ? " urgent" : ""}" style="background:${COLORS[0]};width:18px;height:18px;font-size:10px">1</span>`;
  $("legend").classList.remove("hidden");
  $("legend").innerHTML = `<b>Как читать карту</b>` + row(swatch(MARK.depot), "офис участка — старт всех бригад") + (planned
    ? row(pin(false), "остановка: цвет — бригада, число — порядок объезда")
      + row(pin(true), "авария в маршруте (жёлтое кольцо)")
      + row(swatch(MARK.unassigned), "заявка не назначена — причина по клику")
      + row(`<svg width="22" height="8" aria-hidden="true"><line x1="1" y1="4" x2="21" y2="4" stroke="${COLORS[0]}" stroke-width="3"/></svg>`,
            "путь бригады, стрелки — направление")
      + `<div class="hint">Клик по фамилии слева или по линии — подсветить маршрут.</div>`
    : row(swatch(MARK.urgent), "авария") + row(swatch(MARK.request), "плановая заявка (подключение, локальная работа)")
      + `<div class="hint">Наведите на точку — номер, тип и окно. «Построить план» распределит заявки по бригадам.</div>`);
}

function drawDepot(d) {
  layers.push(L.circleMarker([d.lat, d.lon], { ...MARK.depot }).addTo(map).bindPopup(`Офис участка<br>${d.address}`));
}

function drawDataset(d) {
  clearMap();
  drawDepot(d.depot);
  const bounds = [[d.depot.lat, d.depot.lon]];
  d.requests.forEach((r) => {
    if (!r.lat) return;
    bounds.push([r.lat, r.lon]);
    layers.push(L.circleMarker([r.lat, r.lon], { ...(r.urgent ? MARK.urgent : MARK.request) }).addTo(map).bindTooltip(`${r.id} · ${r.district}<br>${r.hd_type}<br>окно ${r.window}`,
      { direction: "top" }));
  });
  if (bounds.length > 1) map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
}

function renderDataset(d) {
  const c = d.counts;
  $("dataset").innerHTML = `
    <h3>Заявки на день ${d.day}</h3>
    <div>${count(c.requests, "заявка", "заявки", "заявок")} · ${count(c.engineers, "исполнитель", "исполнителя", "исполнителей")} доступно · ${count(c.districts, "район", "района", "районов")}</div>
    <div class="hint">Работы на ${c.work_hours} ч, из них срочных заявок ${c.urgent}.
      Офис участка: ${d.depot.address}</div>
    <div class="dataset-chips">${d.by_skill.map((s) =>
      `<span class="chip">${s.skill}: ${s.count}</span>`).join("")}</div>
    <div class="dataset-chips">${d.windows.map((w) =>
      `<span class="chip">${w.start} — ${w.count}</span>`).join("")}</div>
    <div class="hint">Расстояния: ${d.travel_source === "osrm" ? "по дорогам (OSRM)" : "оценка по координатам"},
      время в пути: ${d.time_source}.</div>`;
}

async function loadDataset() {
  const region = $("region").value;
  setStatus("Загружаем данные дня…");
  try {
    const resp = await fetch(`/api/dataset/${encodeURIComponent(region)}`);
    if (!resp.ok) throw new Error(`сервер ответил ${resp.status}`);
    dataset = await resp.json();
  } catch (e) {
    setStatus(`Не удалось загрузить данные участка: ${e.message}. Проверьте, запущен ли сервер.`);
    return;
  }
  renderDataset(dataset);
  drawDataset(dataset);
  ["metrics", "validation", "compare", "bounds", "advice", "routes", "unassigned", "check"].forEach((id) => { $(id).innerHTML = ""; });
  $("diff").classList.add("hidden");
  showLegend(false);
  setStatus(`${dataset.region}: загружено заявок — ${dataset.counts.requests}. ` +
            `Нажмите «Построить план», чтобы распределить их по исполнителям.`);
}

function highlightRoute(engineerId) {
  activeRoute = activeRoute === engineerId ? null : engineerId;
  Object.entries(routeLines).forEach(([id, line]) => {
    const on = activeRoute === null || activeRoute === id;
    line.setStyle({ opacity: on ? 0.85 : 0.12, weight: activeRoute === id ? 5 : 3 });
  });
  Object.entries(routeMarkers).forEach(([id, markers]) => {
    const on = activeRoute === null || activeRoute === id;
    markers.forEach((m) => m.setOpacity(on ? 1 : 0.25));
  });
  Object.entries(routeArrows).forEach(([id, arrows]) => {
    const on = activeRoute === null || activeRoute === id;
    arrows.forEach((m) => m.setOpacity(on ? 1 : 0.15));
  });
  // Бегущий пунктир на выделенном маршруте: направление видно без вглядывания
  Object.entries(routeLines).forEach(([id, l]) => {
    const el = l.getElement && l.getElement();
    if (el) el.classList.toggle("route-active", activeRoute === id);
  });
  document.querySelectorAll(".route").forEach((el) => {
    el.classList.toggle("active", el.dataset.engineer === activeRoute);
  });
  if (activeRoute && routeMarkers[activeRoute]?.length) {
    map.fitBounds(L.featureGroup(routeMarkers[activeRoute]).getBounds(), { padding: [60, 60] });
  }
}

function draw(data) {
  clearMap();
  const d = data.depot;
  layers.push(L.circleMarker([d.lat, d.lon], { ...MARK.depot }).addTo(map).bindPopup(`Офис участка<br>${d.address}`));

  const bounds = [[d.lat, d.lon]];
  data.routes.forEach((route, i) => {
    const color = COLORS[i % COLORS.length];
    const pts = [[d.lat, d.lon]];
    routeMarkers[route.engineer_id] = [];
    route.stops.forEach((s, k) => {
      pts.push([s.lat, s.lon]);
      bounds.push([s.lat, s.lon]);
      // Номер — это порядок объезда: диспетчеру важно видеть последовательность,
      // а не только то, что точки одного цвета
      const icon = L.divIcon({
        className: "", iconSize: [22, 22], iconAnchor: [11, 11],
        html: `<div class="stop-pin${s.urgent ? " urgent" : ""}" style="background:${color}">${k + 1}</div>`,
      });
      const m = L.marker([s.lat, s.lon], { icon, title: `${k + 1}. ${route.engineer}` }).addTo(map);
      m.bindTooltip(`<b>${k + 1}. ${route.engineer}</b><br>${s.start}–${s.end} · ${s.district}<br>
        ${s.hd_type}<br>окно ${s.window}${s.urgent ? "<br><b>срочная</b>" : ""}`,
        { direction: "top" });
      m.on("click", () => showExplain(s.request_id));
      layers.push(m);
      routeMarkers[route.engineer_id].push(m);
    });
    // Линия проезда по дорогам, если её отдал маршрутизатор; иначе прямые между точками
    const path = route.line || pts;
    const line = L.polyline(path, { color, weight: 3, opacity: .85 }).addTo(map);
    line.on("click", () => highlightRoute(route.engineer_id));
    routeLines[route.engineer_id] = line;
    layers.push(line);
    addDirectionArrows(path, color, route.engineer_id);
  });

  data.unassigned.forEach((u) => {
    if (!u.lat) return;
    bounds.push([u.lat, u.lon]);
    layers.push(L.circleMarker([u.lat, u.lon], { ...MARK.unassigned }).addTo(map)
      .bindPopup(`<b>Не назначена</b><br>${u.request_id} · ${u.hd_type}<br>${u.reason}`));
  });

  if (bounds.length > 1) map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
}

function renderMetrics(data) {
  const s = data.summary, c = data.comparison;
  $("metrics").innerHTML = `
    <div class="card"><div class="v">${s.assigned}/${s.requests_total}</div><div class="k">заявок назначено</div></div>
    <div class="card"><div class="v">${s.used_engineers}</div><div class="k">исполнителей</div></div>
    <div class="card"><div class="v">${s.total_km.toFixed(0)} км</div><div class="k">пробег</div></div>
    ${waitCard(s.urgent_wait)}
    ${trafficCard(s.traffic)}`;
  renderUrgentWait(s.urgent_wait);

  const cls = (v, good) => `<span class="${good ? "delta-good" : "delta-bad"}">${v}</span>`;
  $("compare").innerHTML = `
    <h3>Сравнение</h3>
    <div>Базовый вариант ТЗ: назначено ${data.baseline.assigned} из ${s.requests_total}, исполнителей —
      ${data.baseline.used_engineers}, пробег ${data.baseline.total_km.toFixed(0)} км</div>
    <div style="margin-top:6px">Наш план:
      ${cls(signed(c.assigned_delta, "заявка", "заявки", "заявок"), c.assigned_delta >= 0)},
      ${cls(signed(c.engineers_delta, "исполнитель", "исполнителя", "исполнителей"), c.engineers_delta <= 0)},
      ${cls(c.km_delta.toFixed(0) + " км", c.km_delta <= 0)}</div>
    <div class="hint">Доступно бригад на участке: ${s.available_engineers} — план задействовал ${s.used_engineers}.
      ${data.meta.engine === "manual" ? "План изменён вручную, без поиска;"
        : `Расчёт ${data.meta.seconds ?? "—"} с, циклов поиска: ${data.meta.iterations ?? "—"};`}
      расстояния: ${data.travel_source === "osrm" ? "по дорогам (OSRM)" : "оценка по координатам"}.</div>`;

  $("advice").innerHTML = data.advice.length
    ? "<h3>Чего не хватает</h3>" + data.advice.map((a) => `<div class="hint">• ${a.text}</div>`).join("")
    : "<h3>Чего не хватает</h3><div class='hint'>Все заявки распределены.</div>";
}

// Пробки: сколько времени в пути добавила загруженность против свободной дороги — для профиля, файла со слоями
// и внешнего сервиса одинаково (src/metrics.traffic_effect). refactor: Claude, 26.09.2026
const MODE_SHORT = { car: "машины", public: "общ. транспорт", bike: "велосипед", foot: "пешком" };
function trafficCard(t) {
  if (!t || !t.travel_min) return "";
  const hours = (m) => (m / 60).toFixed(1).replace(".", ",") + " ч";
  const sign = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v);
  const modes = Object.entries(t.by_mode).filter(([k]) => k !== "foot")
    .map(([k, v]) => `${MODE_SHORT[k] || k} ${sign(v.extra_pct)}%`).join(", ");
  return `<div class="card"><div class="v">${sign(t.extra_pct)}%</div><div class="k">пробки: время в пути</div>
    <div class="hint">${hours(t.travel_min)} в пути, без пробок ${hours(t.free_min)}${modes ? ` · ${modes}` : ""}
      · ${t.source}</div></div>`;
}

// Ожидание аварий: карточка — максимум ожидания; список — по каждой аварии (прогноз опозданий, ТЗ 2.5).
function waitCard(w) {
  if (!w || !w.total) return `<div class="card"><div class="v">—</div><div class="k">аварий нет</div></div>`;
  return `<div class="card"><div class="v">${w.max_min} мин</div><div class="k">ожидание аварий, макс.</div>
    <div class="hint">назначено ${w.assigned}/${w.total} · всего ${w.sum_min} мин ·
      дольше ${w.target_min / 60} ч: ${w.over_target}</div></div>`;
}

function renderUrgentWait(w) {
  if (!w || !w.total) { $("urgentWait").innerHTML = ""; return; }
  const row = (i) => i.start
    ? `<li><span>${i.address}</span><span>${i.base} → ${i.start} <b>+${i.wait_min} мин</b>, ${i.engineer}${i.status === "выполнена" ? " · выполнена" : ""}</span></li>`
    : `<li><span>${i.address}</span><span class="delta-bad">не назначена</span></li>`;
  $("urgentWait").innerHTML = `
    <div class="valid-head" onclick="document.getElementById('waitList').classList.toggle('hidden')">
      <b>Ожидание аварий</b> <span class="hint">— от поступления или начала смен до начала работ</span></div>
    <ul id="waitList" class="valid-list hidden">${w.items.map(row).join("")}</ul>`;
}

function renderValidation(data) {
  const v = data.validation;
  if (!v) { $("validation").innerHTML = ""; return; }
  $("validation").innerHTML = `
    <div class="valid-head" onclick="document.getElementById('validList').classList.toggle('hidden')">
      <span class="valid-badge ${v.ok ? "valid-ok" : "valid-bad"}">${v.ok ? "✓" : "!"}</span>
      <span><b>${v.ok ? "Все обязательные ограничения соблюдены" : "Есть нарушения"}</b><br>
        <span class="hint">${v.summary}. Проверка независимая: план перечитан заново.</span></span>
    </div>
    <ul id="validList" class="valid-list hidden">
      ${v.checks.map((c) => `<li><span>${c.title}</span>
        <span class="${c.ok ? "delta-good" : "delta-bad"}">
          ${c.ok ? `${c.checked} проверок` : `нарушений ${c.violations.length}`}</span></li>`).join("")}
      ${v.unassigned_explained ? `<li><span>Не назначено с объяснением</span>
        <span>${v.unassigned_explained}</span></li>` : ""}
    </ul>`;
}

function renderBounds(data) {
  const s = data.search || {};
  // Предупреждение памяти — и после события, где границ нет (Н55).
  const memory = s.memory_warning
    ? `<div class="hint delta-bad" style="margin-top:6px">Память: ${s.memory_warning}</div>` : "";
  if (!data.bounds) { $("bounds").innerHTML = memory; return; }
  const b = data.bounds;
  const row = (label, r, unit, reached) => `
    <div class="kv"><span>${label}</span>
      <span>${r.plan}${unit} ${reached
        ? "<span class='delta-good'>— граница достигнута</span>"
        : `против ${r.bound}${unit} <span class="hint">(+${r.gap_pct}%)</span>`}</span></div>`;
  // Аварии, начатые в пределах ориентира: больше границы не бывает ни в одном плане (шаг 6 PORT-PLAN, 26.09).
  const urgentRow = (u) => u ? `
    <div class="kv"><span>Аварии, начатые за ${u.target_min / 60} ч</span>
      <span>${u.plan} из ${u.total} ${u.plan >= u.bound
        ? "<span class='delta-good'>— граница достигнута</span>"
        : `против ${u.bound} <span class="hint">(границы)</span>`}</span></div>` : "";
  $("bounds").innerHTML = `<h3>Насколько это близко к пределу</h3>
    ${row("Выполнено заявок", b.assigned, "", b.assigned.plan >= b.assigned.bound)}
    ${urgentRow(b.urgent)}
    ${row("Исполнителей", b.engineers, "", false)}
    ${row("Пробег", b.distance, " км", false)}
    <div class="hint">Границы получены ослаблением задачи: по заявкам — вместимость смен
      и окон, ${b.urgent ? "по авариям — к каждой сразу едет ближайшая бригада с допуском по самой быстрой дороге дня, " : ""}по людям — объём работ, по пробегу — вес остовного дерева. Ни один алгоритм
      не может их перейти, но и достичь их обычно нельзя: настоящий план обязан соблюдать
      окна и ездить по дорогам.</div>
    ${s.stop_reason ? `<div class="hint" style="margin-top:6px"><b>Поиск: ${s.stop_reason}</b>
      циклов поиска: ${s.iterations}, последнее улучшение — на ${s.last_improvement}-м.
      ${s.hint}</div>` : ""}
    ${memory}`;
}

function renderRoutes(data) {
  $("routes").innerHTML = "<h3>Маршруты</h3>" + data.routes.map((r, i) => `
    <div class="route" data-engineer="${r.engineer_id}">
      <div class="head" onclick="toggleRoute(this, '${r.engineer_id}')">
        <span class="name"><span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span>${r.engineer}</span>
        <span class="meta">${count(r.stops.length, "заявка", "заявки", "заявок")} · ${r.km} км · ${r.transport}</span>
      </div>
      <ol class="hidden">${r.stops.map((s, k) => `
        <li onclick="event.stopPropagation(); showExplain('${s.request_id}')">
          <b>${k + 1}.</b> ${s.start}–${s.end} · ${s.district} · ${s.hd_type}${s.urgent ? " ⚡" : ""}${s.status === "Завершено" ? " · завершено (подтверждено)" : ""}
        </li>`).join("")}</ol>
    </div>`).join("")
    + (data.idle.length ? `<div class="hint">Не выходят на линию: ${data.idle.join(", ")}. При событии их вызывают только отдельно — галочкой «вызвать бригаду с выходного».</div>` : "");

  $("unassigned").innerHTML = "<h3>Не назначены</h3>" + (data.unassigned.length
    ? data.unassigned.map((u) => `<div class="unass"><b>${u.request_id}</b> · ${u.district} ·
        ${u.hd_type}<br>${u.reason}</div>`).join("")
    : "<div class='hint'>Нет — все заявки распределены.</div>");
}

function renderTargets(data) {
  const kind = $("eventKind").value;
  const sel = $("eventTarget");
  if (kind === "cancel" || kind === "assign" || kind === "complete") {
    sel.innerHTML = data.routes.flatMap((r) => r.stops.filter((s) => s.status !== "Завершено").map((s) =>
      `<option value="${s.request_id}">${s.request_id} · ${s.district} · ${s.start}</option>`)).join("");
    $("eventHint").textContent = "Заявка отменится, план пересоберётся с учётом уже начатых работ.";
  } else if (kind === "engineer_off") {
    sel.innerHTML = data.routes.map((r) =>
      `<option value="${r.engineer_id}">${r.engineer} (${count(r.stops.length, "заявка", "заявки", "заявок")})</option>`).join("");
    $("eventHint").textContent = "Начатые до этого времени заявки за ним останутся, остальные уйдут другим.";
  } else {
    sel.innerHTML = "<option value=''>— адрес или точка на карте —</option>";
    $("eventHint").textContent = "Введите адрес или кликните точку на карте, затем «Перестроить».";
  }
  if (kind === "complete") $("eventHint").textContent = "Подтверждение диспетчером после планового окончания. Заявка останется в истории и не будет распределяться повторно.";
  if (kind === "assign") {
    data.unassigned.forEach((u) => sel.add(new Option(`${u.request_id} · не назначена`, u.request_id)));
    $("assignEngineer").replaceChildren(...data.available_engineers.map((e) => new Option(e.name, e.id)));
    $("eventHint").textContent = "Выберите заявку и исполнителя. Начатый выезд защищён; нарушение ограничений отклоняется. Следующий свободный расчёт может изменить назначение.";
  }
  $("assignRow").style.display = kind === "assign" ? "" : "none";
  $("addressRow").style.display = kind === "urgent" ? "" : "none";
  $("reserveRow").style.display = kind === "urgent" || kind === "engineer_off" ? "" : "none";
}

async function findAddress() {
  const address = $("eventAddress").value.trim();
  if (!address) return;
  $("eventHint").textContent = "Ищем адрес…";
  const r = await (await fetch("/api/geocode", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ address }) })).json();
  if (!r.ok) { $("eventHint").textContent = r.error; return; }

  pickedPoint = { lat: r.lat, lng: r.lon };
  if (pickMarker) map.removeLayer(pickMarker);
  pickMarker = L.marker([r.lat, r.lon]).addTo(map)
    .bindPopup(`Новая срочная заявка<br>${r.matched || address}`).openPopup();
  map.setView([r.lat, r.lon], 15);
  $("eventHint").innerHTML = `Найдено: ${r.matched || address}` +
    (r.warning ? `<br><span class="delta-bad">${r.warning}</span>` : "");
}

function renderDiff(diff, text) {
  const el = $("diff");
  el.classList.remove("hidden");
  const rows = [];
  if (diff.added.length) rows.push(`добавлено: ${diff.added.map((a) => a.address
    ? `${a.urgent ? "авария — " : ""}${a.address} → ${a.engineer}, начало ${a.start}` : a.request_id).join("; ")}`);
  if (diff.removed.length) rows.push(`снято: ${diff.removed.map((a) => a.request_id).join(", ")}`);
  if (diff.moved.length) rows.push(`переназначено заявок: ${diff.moved.length}`);
  if (diff.order_changed.length) rows.push(`порядок объезда изменён у исполнителей: ${diff.order_changed.length}`);
  el.innerHTML = `<h3>Что изменилось</h3><div>${text}</div>
    <div style="margin-top:6px">${rows.join("<br>") || "перестановок не потребовалось"}</div>
    <div class="hint">Заявок: ${diff.assigned_before} → ${diff.assigned_after} ·
      исполнителей: ${diff.engineers_before} → ${diff.engineers_after} ·
      пробег: ${diff.km_before} → ${diff.km_after} км</div>`;
}

function toggleRoute(head, engineerId) {
  head.parentNode.querySelector("ol").classList.toggle("hidden");
  highlightRoute(engineerId);
}

async function showExplain(requestId) {
  const check = state && state.mode === "ortools" ? "?check=1" : "";     // в сверке объясняем сверочный план
  const r = await fetch(`/api/explain/${encodeURIComponent($("region").value)}/${requestId}${check}`);
  const e = await r.json();
  const el = $("detail");
  el.classList.remove("hidden");
  el.innerHTML = `<button class="close" onclick="document.getElementById('detail').classList.add('hidden')">×</button>
    <h4>Заявка ${e.request_id}</h4>
    <div>${e.headline}</div>
    <ul>${e.constraints.map((c) => `<li>${c}</li>`).join("")}</ul>
    <div class="hint">Кто ещё мог бы поехать и как изменился бы общий пробег плана:</div>
    <ul>${e.alternatives.map((a) => `<li>${a.engineer} — ${a.current ? "назначена сейчас"
      : `${a.delta_km > 0 ? "+" : ""}${a.delta_km} км${a.extra_crew ? ", и понадобится ещё одна бригада" : ""}`}</li>`).join("")}</ul>
    <div class="hint">Не подошли по ограничениям исполнителей: ${e.blocked_count}.</div>`;
}

function setStatus(text) { $("status").textContent = text; }

// Сверка с OR-Tools — особый режим с упрощёнными рамками: галочка, полоса над картой, своя надпись на кнопке;
// план сверки — для сравнения, события к нему не применяются. refactor: Claude, 26.09.2026 — шаг 7 PORT-PLAN
const ORTOOLS_FRAME_SHORT = "Без пробок; цель — аварии → подключения → заявки → бригады → км (без «аварий сверх 2 ч» "
  + "и ожидания — их OR-Tools не видит); события недоступны";

function setOrtoolsMode(on) {
  document.body.classList.toggle("ortools-mode", on);
  $("plan").textContent = on ? "Сверить с OR-Tools" : "Построить план";
  $("modeBanner").classList.toggle("hidden", !on);
  const sec = Number($("seconds").value);
  $("modeBanner").innerHTML = `<b>Режим сверки с OR-Tools — упрощённые рамки.</b> ${ORTOOLS_FRAME_SHORT}.
    Считаются три плана: базовый вариант ТЗ, OR-Tools и наш поиск — по ${sec} с (около ${2 * sec + 5} с).
    План на карте — для сравнения; рабочий план и события дня сохраняются. Снимите галочку, чтобы вернуться к ним.`;
  $("apply").disabled = on;
  if (!on && state && state.mode === "ortools") {       // сверку убираем — возвращаем рабочий план со всеми событиями
    fetch(`/api/plan/${encodeURIComponent($("region").value)}`).then(async (r) => {
      if (r.ok) { showPlan(await r.json()); setStatus("Сверка закрыта — на экране снова рабочий план."); }
      else { state = null; await loadDataset(); setStatus("Сверка закрыта. Рабочего плана ещё нет — нажмите «Построить план»."); }
    });
  }
}

async function buildPlan() {
  const btn = $("plan");
  const ortools = $("ortools").checked;
  btn.disabled = true;
  setStatus(ortools ? "Считаем три плана в упрощённых рамках: базовый ТЗ, наш поиск, затем OR-Tools…" : "Считаем план…");
  const body = {
    region: $("region").value, seconds: Number($("seconds").value),
    mode: $("mode").value, ortools,
  };
  const r = await fetch("/api/plan", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  btn.disabled = false;
  if (!r.ok) { setStatus(`Не удалось: ${(await r.json()).detail}`); return; }
  showPlan(await r.json());
}

// План на экране: рабочий или сверочный (у сверки — своя таблица наверху панели).
function showPlan(data) {
  state = data;
  $("diff").classList.add("hidden");
  renderMetrics(state); renderValidation(state); renderBounds(state); renderRoutes(state); renderTargets(state); draw(state);
  showLegend(true);
  $("check").innerHTML = "";
  if (state.ortools_check) renderOrtoolsCheck(state.ortools_check);
  setStatus(`${state.mode === "ortools" ? "Сверка с OR-Tools · " : ""}${state.region}: назначено ${state.summary.assigned} из ${state.summary.requests_total},
    исполнителей — ${state.summary.used_engineers}, пробег ${state.summary.total_km.toFixed(0)} км`);
}

function renderOrtoolsCheck(c) {
  const time = (r) => r.seconds === null ? "мгновенно" : `${r.seconds} с · ${r.cores} ${r.cores === 1 ? "процесс" : r.cores < 5 ? "процесса" : "процессов"}`;
  const valid = c.rows.filter((r) => !r.error).every((r) => r.valid);
  $("check").innerHTML = `
    <h3>Сверка с OR-Tools</h3>
    <table class="check"><thead><tr><th>План</th><th>Аварии</th><th>Подкл.</th><th>Заявки</th><th>Бригады</th><th>км</th></tr></thead><tbody>
      ${c.rows.map((r) => r.error ? `<tr><td>${r.name}</td><td colspan="5" class="delta-bad">${r.error}</td></tr>`
        : `<tr class="${r.best ? "best" : ""}"><td>${r.name}${r.best ? " ✓" : ""}<span class="sub">${time(r)}${r.valid ? "" : " · нарушения!"}</span></td>
            <td>${r.urgent}</td><td>${r.connect}</td><td>${r.assigned}/${r.total}</td><td>${r.engineers}</td><td>${r.km}</td></tr>`).join("")}
    </tbody></table>
    <div class="hint">${valid ? "Все планы прошли одну и ту же проверку ограничений. " : ""}Лучший (✓) — по строкам цели слева
      направо. Бюджет OR-Tools и нашего поиска один — ${c.seconds} с ожидания, процессов тоже поровну: решатель OR-Tools
      однопоточный, поэтому в каждом процессе своя стратегия, берётся лучший план. Рамки: ${c.frame}. Проверить может каждый: кнопка считает все три плана заново, при вас.</div>`;
}

async function applyEvent() {
  if (!state) return;
  const kind = $("eventKind").value;
  const body = { region: $("region").value, kind, at: $("eventAt").value };
  if (kind === "urgent" || kind === "engineer_off") body.reserve = $("eventReserve").checked;   // эскалация: вызов с выходного
  if (kind === "assign") { body.request_id = $("eventTarget").value; body.engineer_id = $("assignEngineer").value; }
  if (kind === "cancel" || kind === "complete") body.request_id = $("eventTarget").value;
  if (kind === "engineer_off") body.engineer_id = $("eventTarget").value;
  if (kind === "urgent") {
    if (!pickedPoint) { setStatus("Сначала кликните точку на карте"); return; }
    body.lat = pickedPoint.lat; body.lon = pickedPoint.lng;
    body.address = $("eventAddress").value.trim() || null;
    body.district = "срочная";
  }
  setStatus("Перестраиваем план…");
  $("apply").disabled = true;
  $("plan").disabled = true;
  $("region").disabled = true;
  try {
    const r = await fetch("/api/event", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const next = await r.json();
    if (!r.ok) throw new Error(typeof next.detail === "string" ? next.detail : "Не удалось применить событие");
    state = next;
    renderMetrics(state); renderValidation(state); renderBounds(state);
    renderRoutes(state); renderTargets(state); draw(state);
    renderDiff(state.diff, state.event_text);
    setStatus(kind === "assign" ? "Ручное назначение сохранено" : "План перестроен");
  } catch (err) {
    setStatus(err.message);
  } finally {
    $("apply").disabled = false;
    $("plan").disabled = false;
    $("region").disabled = false;
  }
}

(async function start() {
  initMap();
  const r = await fetch("/api/regions");
  const { regions, titles } = await r.json();
  // Наборы: встроенные участки и добавленные на вкладке «Данные» админки (значение — идентификатор набора).
  $("region").innerHTML = regions.map((x) => `<option value="${x}">${(titles || {})[x] || x}</option>`).join("");
  // Ссылка «открыть на главной» из вкладки «Данные»: /?region=<набор>.
  const wanted = new URLSearchParams(location.search).get("region");
  if (wanted && regions.includes(wanted)) $("region").value = wanted;
  // Режим остановки по умолчанию — из настроек поиска (админка). refactor: Claude, 22.09.2026 — шаг 7
  const cfg = await (await fetch("/api/settings")).json();
  $("mode").value = cfg.search.plan.stop === "time" ? "time" : "converge";
  $("plan").onclick = buildPlan;
  $("ortools").onchange = () => setOrtoolsMode($("ortools").checked);
  $("seconds").addEventListener("change", () => $("ortools").checked && setOrtoolsMode(true));
  $("apply").onclick = applyEvent;
  $("eventKind").onchange = () => state && renderTargets(state);
  $("findAddress").onclick = findAddress;
  $("eventAddress").addEventListener("keydown", (e) => { if (e.key === "Enter") findAddress(); });
  $("region").onchange = loadDataset;
  await loadDataset();      // диспетчер видит заявки дня до всякого планирования
})();
