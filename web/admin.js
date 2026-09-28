const $ = (id) => document.getElementById(id);
const TRANSPORT = { car: "Автомобиль", public: "Общественный транспорт",
                    foot: "Пешеход", bike: "Велосипед" };
let data = null;

function renderNormatives(n) {
  $("normNote").textContent = n.note || "";
  const reserve = n.travel_reserve_minutes;
  const input = (cls, part, value) =>
    `<input data-cls="${cls}" data-part="${part}" type="number" min="0" max="480" step="5" value="${value}">`;
  $("normTable").innerHTML = Object.entries(n.classes).map(([key, c]) => `
    <tr><td>${c.title}</td>
        <td class="n">${input(key, "technical_minutes", c.technical_minutes)}</td>
        <td class="n">${input(key, "documents_minutes", c.documents_minutes)}</td>
        <td class="n">${c.on_site_minutes}</td>
        <td class="n">${c.base_minutes}</td>
    </tr>`).join("");
  $("normReserve").textContent = `Норматив включает ${reserve} мин резерва на дорогу. ` +
    `В плане резерв заменяется рассчитанной дорогой: заявка занимает время на адресе плюс ` +
    `фактический путь, превышение норматива допустимо.`;
}

function renderAssumptions(a) {
  $("skillsMode").value = a.skills_mode || "history";
  $("skillsNote").textContent = a.skills_mode_comment || "";
  $("shiftStart").value = a.shift_start;
  $("shiftEnd").value = a.shift_end;
  $("mix").innerHTML = "<div class='hint'>Доли типов транспорта у исполнителей</div>" +
    Object.entries(a.transport_mix).map(([mode, share]) => `
      <div class="kv"><span>${TRANSPORT[mode] || mode}</span>
        <span><input data-mode="${mode}" type="number" min="0" max="1" step="0.05" value="${share}"
              style="width:70px;text-align:right"></span></div>`).join("");

  $("misc").innerHTML = `
    <div class="kv"><span>Срочными считаются</span><span>${a.urgent_bk_types.join(", ")}</span></div>
    <div class="kv"><span>Требуют автомобиля</span><span>${a.require_car_for_hd.join(", ") || "—"}</span></div>
    ${Object.entries(a.speed_kmh).map(([mode, kmh]) => `
      <div class="kv"><span>Скорость: ${TRANSPORT[mode] || mode}</span>
        <span>${kmh === null ? "по данным маршрутизатора" : kmh + " км/ч"}</span></div>`).join("")}
    ${Object.entries(a.fixed_overhead_min).map(([mode, min]) => `
      <div class="kv"><span>Надбавка на парковку/ожидание: ${TRANSPORT[mode] || mode}</span>
        <span>${min} мин</span></div>`).join("")}`;
  $("miscNote").textContent = a.note || "";
}

async function loadStats() {
  // refactor: Claude, 22.09.2026 — шаг 7: отчёт портфеля вместо статистики ALNS
  const region = $("statRegion").value;
  const s = await (await fetch(`/api/search-stats/${encodeURIComponent(region)}`)).json();
  if (!s.available) {
    $("stats").innerHTML = "<div class='hint'>Для этого участка план ещё не считали в текущей сессии.</div>";
    return;
  }
  const bars = (obj) => {
    const items = Object.entries(obj);
    const max = Math.max(1, ...items.map(([, v]) => v));
    return items.length ? items.map(([k, v]) => `
      <div style="margin:5px 0"><div class="kv" style="border:0;padding:0"><span>${k}</span><span>${v}</span></div>
        <div class="bar"><i style="width:${(v / max * 100).toFixed(0)}%"></i></div></div>`).join("")
      : "<div class='hint'>Нет.</div>";
  };
  $("stats").innerHTML = `
    <div class="kv"><span>Расчёт</span><span>${s.seconds} с, проверено вариантов плана: ${s.candidates}</span></div>
    <div class="kv"><span>Остановка</span><span>${s.stop_reason}: ${s.hint}</span></div>
    ${s.members.map((m) => `<div class="kv"><span>№${m.number} ${m.mode_title}, сид ${m.seed}${m.winner ? " — лучший" : ""}</span>
      <span>${m.ok ? `циклов поиска: ${m.epochs}, лучший план — на ${m.best_epoch}-м` : "сбой: " + m.error}</span></div>`).join("")}
    <div id="progressChart"></div>
    <div class="hint" style="margin-top:8px">Какие изменения плана дали новые лучшие планы</div>${bars(s.records)}
    <div class="hint" style="margin-top:8px">Какие изменения давали вариант лучше текущего плана</div>${bars(s.better)}`;
  renderProgress(s.members.filter((m) => m.ok && m.progress && m.progress.length));
}

// Как улучшалось решение: у каждого поиска — лучший план на каждый момент расчёта. Две панели с общей осью времени:
// бригады (ступеньки) и пробег; цвет — участник, в постоянном порядке. Наведение — значения всех участников.
// refactor: Claude, 26.09.2026 — просьба Михаила: графики в «Анализе»
const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"];
function renderProgress(members) {
  const box = $("progressChart");
  if (!members.length) { box.innerHTML = ""; return; }
  const W = 700, H = 120, L = 46, R = 12, T = 10, B = 22;
  const tMax = Math.max(...members.flatMap((m) => m.progress.map((p) => p[1])));
  const x = (t) => L + (t / tMax) * (W - L - R);
  const panel = (title, idx, fmt, step) => {
    const vals = members.flatMap((m) => m.progress.map((p) => p[idx]));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * .08, y = (v) => T + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * (H - T - B);
    const ticks = step ? [...new Set([Math.round(lo), Math.round(hi)])] : [lo, (lo + hi) / 2, hi];
    const lines = members.map((m, i) => {
      const pts = m.progress.map((p, k) => {
        const px = x(p[1]).toFixed(1), py = y(p[idx]).toFixed(1);
        if (!step || k === 0) return `${k ? "L" : "M"}${px},${py}`;
        return `H${px}V${py}`;
      }).join("");
      return `<path d="${pts}" fill="none" stroke="${SERIES[(m.number - 1) % SERIES.length]}" stroke-width="2"
        stroke-linejoin="round" ${m.winner ? "" : 'stroke-opacity=".75"'}/>`;
    }).join("");
    return `<div class="pc-title">${title}</div>
      <svg viewBox="0 0 ${W} ${H}" class="pc" data-idx="${idx}">
        ${ticks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="pc-grid"/>
          <text x="${L - 6}" y="${y(v) + 4}" class="pc-axis" text-anchor="end">${fmt(v)}</text>`).join("")}
        ${lines}<line class="pc-cross" x1="0" x2="0" y1="${T}" y2="${H - B}" visibility="hidden"/>
        <text x="${W - R}" y="${H - 6}" class="pc-axis" text-anchor="end">${tMax.toFixed(0)} с</text>
        <text x="${L}" y="${H - 6}" class="pc-axis">0 с</text>
      </svg>`;
  };
  box.innerHTML = `
    <div class="hint" style="margin-top:10px">Как улучшалось решение: лучший план каждого поиска по ходу расчёта</div>
    <div class="pc-legend">${members.map((m) => `<span><i style="background:${SERIES[(m.number - 1) % SERIES.length]}"></i>
      №${m.number} ${m.mode_title}${m.winner ? " — лучший" : ""}</span>`).join("")}</div>
    ${panel("Бригад в лучшем плане", 3, (v) => Math.round(v), true)}
    ${panel("Пробег лучшего плана, км", 4, (v) => Math.round(v), false)}
    <div id="pcTip" class="pc-tip hidden"></div>`;
  const tip = $("pcTip");
  box.querySelectorAll("svg.pc").forEach((svg) => {
    svg.onmousemove = (e) => {
      const r = svg.getBoundingClientRect(), t = ((e.clientX - r.left) / r.width * W - L) / (W - L - R) * tMax;
      if (t < 0 || t > tMax) return;
      box.querySelectorAll(".pc-cross").forEach((c) => { c.setAttribute("x1", x(t)); c.setAttribute("x2", x(t)); c.setAttribute("visibility", "visible"); });
      const at = (m) => m.progress.filter((p) => p[1] <= t).pop() || m.progress[0];
      tip.innerHTML = `<b>${t.toFixed(1)} с</b>` + members.map((m) => { const p = at(m);
        return `<div><i style="background:${SERIES[(m.number - 1) % SERIES.length]}"></i>№${m.number}: цикл ${p[0]}, бригад ${p[3]}, ${Math.round(p[4])} км, заявок ${p[2]}</div>`; }).join("");
      tip.classList.remove("hidden");
      const br = box.getBoundingClientRect();
      tip.style.left = `${Math.min(e.clientX - br.left + 14, br.width - 260)}px`; tip.style.top = `${e.clientY - br.top + 10}px`;
    };
    svg.onmouseleave = () => { tip.classList.add("hidden"); box.querySelectorAll(".pc-cross").forEach((c) => c.setAttribute("visibility", "hidden")); };
  });
}

async function save(kind) {
  const body = {};
  if (kind === "normatives") {
    const classes = {};
    document.querySelectorAll("#normTable input[data-cls]").forEach((i) => {
      (classes[i.dataset.cls] ||= {})[i.dataset.part] = Number(i.value);
    });
    body.normatives = { classes };
  } else {
    const mix = {};
    document.querySelectorAll("#mix input[data-mode]").forEach((i) => {
      mix[i.dataset.mode] = Number(i.value);
    });
    body.assumptions = { shift_start: $("shiftStart").value, shift_end: $("shiftEnd").value,
                         transport_mix: mix, skills_mode: $("skillsMode").value };
  }
  const r = await (await fetch("/api/settings", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  const el = kind === "normatives" ? $("normSaved") : $("asmSaved");
  // Время на адресе и полный норматив считает сервер — перерисовываем по его ответу.
  if (kind === "normatives") renderNormatives((await (await fetch("/api/settings")).json()).normatives);
  el.textContent = r.rejected.length
    ? `не принято: ${r.rejected.join("; ")}`
    : `сохранено, планы пересчитаются (${r.applied.length} полей)`;
  el.className = r.rejected.length ? "" : "saved";
  setTimeout(() => { el.textContent = ""; }, 6000);
}

// «Что если»: быстро — один равномерный поиск по 6 с на вариант; как на главной — тот же состав и бюджет, «как сейчас» —
// план с главной, если он построен в тех же настройках. refactor: Claude, 26.09.2026 — шаг 6 PORT-PLAN
async function whatifNote() {
  const e = await json(`/api/whatif/${encodeURIComponent($("whatifRegion").value)}/estimate`);
  const min = (sec) => sec < 90 ? `${Math.round(sec)} с` : `${(sec / 60).toFixed(1)} мин`;
  const upTo = e.converge ? "до " : "";
  $("whatifNote").textContent = $("whatifFull").checked
    ? `Как на главной: ${e.full_runs} вариантов, ${upTo}${e.full_case_seconds} с на каждый, участников поиска — ${e.members}; всего `
      + `${upTo}${min(e.full_seconds)}, примерно в ${Math.max(1, Math.round(e.full_seconds / e.fast_seconds))} раза дольше `
      + `быстрого. «Как сейчас» — ${e.current_from_plan ? "план с главной" : "считается здесь (плана с главной в этих настройках нет)"}. `
      + "Пока идёт расчёт, построение плана на главной ждёт."
    : `Быстрый расчёт: один равномерный поиск, ${e.fast_seconds / e.cases} с на вариант — около ${min(e.fast_seconds)}. `
      + "«Как сейчас» тоже считается здесь, поэтому может отличаться от плана на главной; варианты сравнимы между собой, "
      + "небольшая разница может быть случайной.";
}

async function runWhatif() {
  const region = $("whatifRegion").value, full = $("whatifFull").checked;
  $("whatif").innerHTML = "<div class='hint'>Считаем варианты…</div>";
  const r = await json(`/api/whatif/${encodeURIComponent(region)}?full=${full}`);
  // Строки цели в её порядке; better — в какую сторону изменение к лучшему.
  const cols = [["urgent", "Аварии", 1], ["connect", "Подключения", 1], ["assigned", "Заявки", 1],
                ["over_target", `Аварий сверх ${r.target_min / 60} ч`, -1], ["engineers", "Бригады", -1],
                ["km", "км", -1], ["wait_min", "Ожидание аварий, мин", -1]];
  const cell = (c, key, better) => {
    const d = c.delta[key];
    const mark = d === 0 || c.case === r.cases[0].case ? "" :
      ` <span class="${d * better > 0 ? "delta-good" : "delta-bad"}">${d > 0 ? "+" : ""}${d}</span>`;
    return `<td class="n">${key === "assigned" ? `${c[key]}/${c.total}` : c[key]}${mark}</td>`;
  };
  $("whatif").innerHTML = `
    <table class="norm"><thead><tr><th>Вариант</th>${cols.map(([, t]) => `<th class="n">${t}</th>`).join("")}</tr></thead>
      <tbody>${r.cases.map((c) => `<tr><td style="white-space:nowrap">${c.case}${c.source === "план с главной"
        ? ' <span class="hint">(план с главной)</span>' : ""}</td>${cols.map(([k, , b]) => cell(c, k, b)).join("")}</tr>`).join("")}
      </tbody></table>
    <div class="hint">${r.full ? "Как на главной" : "Быстрый расчёт"}: участников поиска — ${r.members}, ${r.seconds} с на вариант.
      Разница — против «как сейчас».</div>`;
}

async function loadTrafficState() {
  const region = $("trafficRegion").value;
  const s = await json(`/api/traffic/${encodeURIComponent(region)}`);
  $("trafficSource").innerHTML = s.sources.map((x) => `<option value="${x.id}">${x.title}</option>`).join("");
  // Готовые файлы набора (data/traffic/, scripts/build_traffic_layers.py) и «другой файл» — путь вручную.
  $("trafficFile").innerHTML = s.files.map((f) => `<option value="${f.path}">${f.name} — ${f.layers} слоёв по ${f.step_min} мин</option>`)
    .join("") + `<option value="">загрузить свой файл…</option>`;
  $("trafficFile").dataset.notes = JSON.stringify(Object.fromEntries(s.files.map((f) => [f.path, f.note])));
  $("trafficState").innerHTML = `Сейчас: <b>${s.current}</b>, точек в участке ${s.points}`;
  if (s.current.startsWith("слои")) $("trafficSource").value = "file";     // список показывает действующий источник
  showSourceNote(s.sources);
  $("trafficSource").onchange = () => showSourceNote(s.sources);
  $("trafficFile").onchange = () => showSourceNote(s.sources);
  renderTrafficChart(s.hours);
}

function showSourceNote(sources) {
  const chosen = sources.find((x) => x.id === $("trafficSource").value);
  const file = $("trafficSource").value === "file";
  const custom = file && !$("trafficFile").value;
  const fileNote = file && !custom ? (JSON.parse($("trafficFile").dataset.notes || "{}")[$("trafficFile").value] || "") : "";
  $("trafficNote").textContent = [chosen ? chosen.note : "", fileNote && `Файл: ${fileNote}.`].filter(Boolean).join(" ");
  $("trafficFile").style.display = file ? "" : "none";
  $("trafficUpload").style.display = custom ? "" : "none";
  $("trafficSample").style.display = custom ? "" : "none";
  $("trafficSample").href = `/api/traffic/${encodeURIComponent($("trafficRegion").value)}/sample`;
}

async function applyTraffic() {
  const file = $("trafficSource").value === "file";
  const body = { region: $("trafficRegion").value, source: $("trafficSource").value,
                 path: file ? ($("trafficFile").value || null) : null };
  // Свой файл — загружается из браузера (сохранится в data/traffic/ и появится в списке набора).
  if (file && !$("trafficFile").value) {
    const upload = $("trafficUpload").files[0];
    if (!upload) { $("trafficState").innerHTML = `<span class="delta-bad">Выберите файл со слоями</span>`; return; }
    body.content = await upload.text();
    body.filename = upload.name;
  }
  const r = await (await fetch("/api/traffic", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  $("trafficState").innerHTML = r.ok
    ? `Сейчас: <b>${r.current}</b>. <span class="saved">${r.note}</span>`
    : `Сейчас: <b>${r.current}</b>. <span class="delta-bad">${r.error}</span>`;
  if (r.hours) renderTrafficChart(r.hours);
  if (r.ok && body.content) await loadTrafficState();      // загруженный файл — теперь в списке набора
}

// Пробки по часам для текущего источника: профиль — один коэффициент на участок; слои (файл, внешний сервис) —
// среднее по парам точек и разброс 10–90%. Высота — во сколько раз дорога на машине дольше свободной.
// refactor: Claude, 26.09.2026
function renderTrafficChart(t) {
  if (!t) { $("trafficChart").innerHTML = ""; return; }
  if (!t.enabled) { $("trafficChart").innerHTML = "<div class='hint'>Загруженность не учитывается: время в пути — свободная дорога.</div>"; return; }
  const num = (v) => String(v).replace(".", ",");
  const top = Math.max(...t.hours.map((h) => h.p90 ?? h.car)) * 1.08;
  const pct = (v) => (v / top * 100).toFixed(1);
  const [from, to] = t.shift;
  const layered = t.hours.some((h) => h.from_layer);
  const bars = t.hours.map((h) => {
    const tip = h.from_layer
      ? `${h.hour}:00 — машина ×${num(h.car)} в среднем, у 80% пар ×${num(h.p10)}–${num(h.p90)}`
      : `${h.hour}:00 — машина ×${num(h.car)}, общественный транспорт ×${num(h.public)}`;
    const whisker = h.from_layer ? `<b style="bottom:${pct(h.p10)}%;height:${(pct(h.p90) - pct(h.p10)).toFixed(1)}%"></b>` : "";
    return `<div class="h${h.hour + 1 > from && h.hour < to ? " shift" : ""}" title="${tip}"><i style="height:${pct(h.car)}%"></i>${whisker}</div>`;
  }).join("");
  const sens = Object.entries(t.sensitivity).map(([k, v]) => `${TRANSPORT[k] || k} ${num(v)}`).join(", ");
  $("trafficChart").innerHTML = `
    <div class="tchart">${bars}<div class="one" style="bottom:${pct(1)}%" title="×1 — свободная дорога"></div></div>
    <div class="taxis">${t.hours.map((h) => `<span>${h.hour % 3 === 0 ? h.hour : ""}</span>`).join("")}</div>
    <div class="hint">Во сколько раз дорога на машине дольше свободной, по часу выезда (${t.source}); пунктир — ×1,
      фон — часы смен ${num(from)}–${num(to)} ч.${layered ? " Усик — разброс между парами точек: у 80% пар коэффициент внутри него." : ""}
      Наведите на столбик — точные числа.</div>
    <div class="hint">${layered ? "Слои задают время на машине; остальной транспорт страдает от той же загруженности в меру "
      + "чувствительности" : "Чувствительность транспорта к пробкам"}: ${sens}.</div>`;
}

// Вкладка «Цель» — прежний вид (список цен, как до 542e5b2) + веса для поиска на той же странице. Поле, на которое при
// нынешних настройках ничего не влияет, недоступно, причина — строкой под ним. refactor: Claude, 23.09.2026 — по просьбе Михаила
// «Цель»: порядок строк слева, настройки справа (перенос 26.09: поиск сравнивает планы по порядку строк — весов нет).
const GOAL_ROWS = [
  { title: "Назначенные аварии", dir: "↑ больше" },
  { title: "Назначенные подключения", dir: "↑ больше" },
  { title: "Все назначенные заявки", dir: "↑ больше" },
  { title: "late", dir: "↓ меньше" },
  { title: "Использованные исполнители", dir: "↓ меньше" },
  { title: "Стоимость маршрута", dir: "↓ меньше", cost: true },
  { title: "Суммарное ожидание аварий", dir: "↓ меньше" },
];
let weightValues = {};

const km = (x) => (x < 10 ? x.toFixed(1).replace(".", ",") : x.toFixed(0)) + " км";

function tailExamples(v) {
  if (v.urgent_delay_mode === "square") {
    const w = Number(v.urgent_delay_square) || 0;
    return `Одна авария ждёт 1 ч — ${km(w * 3600)}, 3 ч — ${km(w * 32400)}, 8 ч — ${km(w * 230400)}`;
  }
  const w = Number(v.urgent_delay_minute) || 0;
  return `Час ожидания любой аварии — ${km(w * 60)}, 8 ч — ${km(w * 480)}`;
}

function renderWeights(w) {
  weightValues = { ...w.values };
  const v = w.values;
  $("goalLate").checked = !!v.urgent_target_first;
  $("goalLimit").value = v.urgent_target_minutes ?? 120;
  $("goalMode").value = v.urgent_delay_mode || "linear";
  $("goalPrice").value = $("goalMode").value === "square" ? (v.urgent_delay_square ?? 0.00033) : (v.urgent_delay_minute ?? 0.02);
  $("goalMoves").checked = v.event_move_km > 0;
  $("goalMovePrice").value = v.event_move_km > 0 ? v.event_move_km : 5;
  ["goalLate", "goalLimit", "goalPrice", "goalMoves", "goalMovePrice"].forEach((id) => { $(id).oninput = $(id).onchange = updateGoal; });
  // Смена способа — своя цена: у каждого своя единица, значение другого способа сохраняется.
  $("goalMode").onchange = () => {
    $("goalPrice").value = $("goalMode").value === "square" ? (weightValues.urgent_delay_square ?? 0.00033)
                                                           : (weightValues.urgent_delay_minute ?? 0.02);
    updateGoal();
  };
  updateGoal();
}

function updateGoal() {
  const v = currentWeights(), late = v.urgent_target_first;
  $("goalLimit").disabled = !late;
  $("goalMovePrice").disabled = !$("goalMoves").checked;
  $("goalUnit").textContent = v.urgent_delay_mode === "square" ? "км за минуту² ожидания каждой аварии" : "км за минуту ожидания";
  $("goalExamples").textContent = tailExamples(v);
  $("goalRows").innerHTML = GOAL_ROWS.map((r, i) => {
    const off = i === 3 && !late;
    const title = i === 3 ? `Аварии с ожиданием дольше ${v.urgent_target_minutes} мин` : r.title;
    const formula = r.cost ? `<span class="gform">Пробег + цена ожидания аварий${v.event_move_km > 0 ? " + цена переназначений" : ""},
      в км: ${tailExamples(v).charAt(0).toLowerCase() + tailExamples(v).slice(1)}</span>` : "";
    return `<div class="grow ${r.cost ? "gcost" : ""} ${off ? "off" : ""}"><span class="gnum">${i + 1}</span>
      <span>${r.cost ? `<b>${title}</b>` : title}</span><span class="gdir">${off ? "выключено — строка не учитывается" : r.dir}</span>${formula}</div>`;
  }).join("");
}

function currentWeights() {
  const mode = $("goalMode").value, price = Number($("goalPrice").value);
  const out = {
    urgent_target_first: $("goalLate").checked,
    urgent_target_minutes: Number($("goalLimit").value),
    urgent_delay_mode: mode,
    urgent_delay_square: mode === "square" ? price : weightValues.urgent_delay_square,
    urgent_delay_minute: mode === "linear" ? price : weightValues.urgent_delay_minute,
    event_move_km: $("goalMoves").checked ? Number($("goalMovePrice").value) : 0,
  };
  return out;
}

async function postWeights(body, savedId) {
  const r = await fetch("/api/weights", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify(body) });
  const out = await r.json();
  if (!r.ok) { $(savedId).textContent = out.detail; $(savedId).className = "delta-bad"; return; }
  $(savedId).textContent = "сохранено, планы пересчитаются";
  $(savedId).className = "saved";
  markChanged((await (await fetch("/api/settings")).json()).changed);
  setTimeout(() => { $(savedId).textContent = ""; }, 6000);
  return true;
}

const saveWeights = async () => {
  const values = currentWeights();
  if (await postWeights(values, "weightsSaved")) weightValues = { ...weightValues, ...values };
};

// Поиск: портфель (шаг 6в). refactor: Claude, 22.09.2026
const MODES = { learned: "сеть", uniform: "равномерный" };
let search = null;

// Настройки поиска у каждого участника (потока) свои; пустое поле — значение по умолчанию. refactor: Claude, 23.09.2026
// Горизонт расписания (температура отбора) из интерфейса убран 26.09: при 20 с эффекта нет (RESULTS «Горизонт расписания»).
const MEMBER_KNOBS = [
  { f: "population", title: "вариантов плана за цикл", def: 64, step: 1,
    hint: "сколько вариантов текущего плана пробует каждый цикл поиска; больше — шире выбор, но цикл дольше" },
  { f: "chain_max", title: "звеньев цепочки", def: 3, step: 1,
    hint: "сколько разрушений накладывать до одного восстановления (1–4): длиннее — смелее ходы" },
  { f: "repair_noise", title: "шум вставки", def: "", step: 0.01, placeholder: "0.15 / 0",
    hint: "случайность при восстановлении плана; пусто — по умолчанию (план 0.15, событие 0)" },
];
const NET_KNOBS = [
  { f: "hidden", title: "ширина сети", type: "number", def: 256, step: 1, modes: ["learned"],
    hint: "число признаков в скрытых слоях; шире — точнее, но цикл поиска медленнее и нужно больше памяти" },
  { f: "experts", title: "экспертов", type: "number", def: 3, step: 1, modes: ["learned"],
    hint: "голов-экспертов, между которыми выбирает гейт" },
  { f: "lr", title: "шаг обучения", type: "number", def: 0.001, step: 0.0001, modes: ["learned"],
    hint: "шаг Adam: больше — быстрее учится и шумнее" },
  { f: "chain_learned", title: "длину цепочки выбирает сеть", type: "check", def: true, modes: ["learned"],
    hint: "выкл. — длина цепочки разрушений равномерно от 1 до «звеньев цепочки»" },
  { f: "rehomable_feature", title: "признак «заявку можно переселить»", type: "check", def: true, modes: ["learned"],
    hint: "вход сети: есть ли у заявки другая бригада, куда её можно вставить" },
  { f: "normalize_features", title: "нормировка входа по задаче", type: "check", def: true, modes: ["learned"],
    hint: "признаки сети нормируются по самой задаче (скользящее среднее и разброс)" },
  { f: "target_potential", title: "прицел по потенциалу", type: "number", def: 0.5, step: 0.05, modes: ["uniform"],
    hint: "доля случаев, когда место изменения плана выбирается по потенциалу — выгоде переноса заявки к другой бригаде; остальное поровну. эту подсказку равномерный режим взял у сети; 0 — всё поровну" },
  { f: "open_cost", title: "цена открытия бригады при вставке, км", type: "number", def: 300, step: 10, modes: ["uniform"],
    hint: "вставка открывает новую бригаду, только если это экономит больше км. Какой план лучше, не меняет — только то, какие планы строятся" },
];
const openKnobs = new Set();

function memberRow(kind, m, i) {
  const net = m.mode === "learned";
  const num = (field, v, step) => `<td class="n"><input data-kind="${kind}" data-i="${i}" data-f="${field}"
    type="number" step="${step}" value="${v ?? ""}" ${net || field === "seed" ? "" : "disabled"}></td>`;
  const key = `${kind}-${i}`;
  const knobs = MEMBER_KNOBS.map((k) => `<div class="kv"><span>${k.title}<br><span class="hint">${k.hint}</span></span>
    <span><input data-kind="${kind}" data-i="${i}" data-f="${k.f}" data-optional="1" type="number" step="${k.step}"
      value="${m[k.f] ?? ""}" placeholder="${k.placeholder ?? k.def}" style="width:80px;text-align:right"></span></div>`).join("");
  const netKnobs = NET_KNOBS.filter((k) => k.modes.includes(m.mode)).map((k) => {
    const v = m[k.f] ?? k.def;
    const input = k.type === "check"
      ? `<input data-kind="${kind}" data-i="${i}" data-f="${k.f}" type="checkbox" ${v ? "checked" : ""}>`
      : k.type === "number"
      ? `<input data-kind="${kind}" data-i="${i}" data-f="${k.f}" type="number" step="${k.step}" value="${v}"
           style="width:80px;text-align:right">`
      : `<select data-kind="${kind}" data-i="${i}" data-f="${k.f}">${Object.entries(k.options).map(([o, t]) =>
          `<option value="${o}" ${o === v ? "selected" : ""}>${t}</option>`).join("")}</select>`;
    return `<div class="kv"><span>${k.title}<br><span class="hint">${k.hint}</span></span><span>${input}</span></div>`;
  }).join("");
  return `<tr><td>Поиск ${i + 1}</td>
    <td><select data-kind="${kind}" data-i="${i}" data-f="mode">${Object.entries(MODES).map(([k, t]) =>
      `<option value="${k}" ${k === m.mode ? "selected" : ""}>${t}</option>`).join("")}</select></td>
    ${num("seed", m.seed, 1)}
    <td><button data-knobs="${key}" title="Настройки этого поиска">настройки</button></td>
    <td><button data-kind="${kind}" data-drop="${i}" title="Убрать поиск">×</button></td></tr>
    <tr class="${openKnobs.has(key) ? "" : "hidden"}" data-knobs-row="${key}"><td colspan="5">${knobs}${netKnobs
      ? `<div class="hint" style="margin-top:8px"><b>${net ? "Сеть" : "Равномерный"}</b></div>${netKnobs}` : ""}</td></tr>`;
}

function renderSearch() {
  $("urgentFront").checked = !!search.urgent_front;
  $("searchWorkers").value = search.workers;
  $("searchCores").textContent = `ядер на машине: ${search.cores}`;
  $("searchStop").value = search.plan.stop;
  $("searchPatience").value = search.plan.patience;
  $("eventSeconds").value = search.event.seconds;
  for (const kind of ["plan", "event"]) {
    $(kind + "Members").innerHTML = search[kind].members.map((m, i) => memberRow(kind, m, i)).join("");
  }
  $("searchNote").textContent = search.note || "";
  document.querySelectorAll("[data-f]").forEach((el) => {
    el.onchange = () => {
      const m = search[el.dataset.kind].members[Number(el.dataset.i)];
      if (el.dataset.optional && el.value === "") delete m[el.dataset.f];      // пусто — по умолчанию
      else if (el.type === "checkbox") m[el.dataset.f] = el.checked;
      else if (el.tagName === "SELECT") m[el.dataset.f] = el.value;
      else m[el.dataset.f] = Number(el.value);
      if (el.dataset.f === "mode") renderSearch();
    };
  });
  document.querySelectorAll("[data-knobs]").forEach((el) => {
    el.onclick = () => {
      const key = el.dataset.knobs;
      if (openKnobs.has(key)) openKnobs.delete(key); else openKnobs.add(key);
      document.querySelector(`[data-knobs-row="${key}"]`).classList.toggle("hidden");
    };
  });
  document.querySelectorAll("[data-drop]").forEach((el) => {
    el.onclick = () => { search[el.dataset.kind].members.splice(Number(el.dataset.drop), 1); renderSearch(); };
  });
}

// Общие поля сразу пишутся в search: перерисовка таблицы участников их не сбрасывает (Н44).
function bindSearchFields() {
  $("searchWorkers").onchange = () => { search.workers = Number($("searchWorkers").value); };
  $("urgentFront").onchange = () => { search.urgent_front = $("urgentFront").checked; };
  $("searchStop").onchange = () => { search.plan.stop = $("searchStop").value; };
  $("searchPatience").onchange = () => { search.plan.patience = Number($("searchPatience").value); };
  $("eventSeconds").onchange = () => { search.event.seconds = Number($("eventSeconds").value); };
}

async function saveSearch() {
  const body = {
    urgent_front: $("urgentFront").checked,
    workers: Number($("searchWorkers").value),
    plan: { ...search.plan, stop: $("searchStop").value, patience: Number($("searchPatience").value) },
    event: { ...search.event, seconds: Number($("eventSeconds").value) },
  };
  const r = await fetch("/api/search-settings", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const out = await r.json();
  if (!r.ok) {
    $("searchSaved").textContent = out.detail;
    $("searchSaved").className = "";
    return;
  }
  search = out;
  renderSearch();
  updateGoal();
  markChanged((await (await fetch("/api/settings")).json()).changed);
  $("searchSaved").textContent = "сохранено, действует со следующего расчёта";
  $("searchSaved").className = "saved";
  setTimeout(() => { $("searchSaved").textContent = ""; }, 6000);
}

// Вкладки: Данные / Цель / Нормативы и допущения / Поиск / Анализ. Выбранная — в адресе (#goal),
// чтобы ссылка и перезагрузка открывали ту же вкладку. refactor: Claude, 23.09.2026
function showTab(name) {
  const tabs = [...document.querySelectorAll("#tabs [data-tab]")].map((b) => b.dataset.tab);
  const tab = tabs.includes(name) ? name : tabs[0];
  document.querySelectorAll("#tabs [data-tab]").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll("[data-panel]").forEach((p) => p.classList.toggle("active", p.dataset.panel === tab));
  resetBar(tab);
}

// Сброс к заводским: одна строка у вкладок — состояние текущей вкладки, «Сбросить вкладку», «сбросить всё».
// refactor: Claude, 26.09.2026 — вместо двух кнопок друг под другом (замечание Михаила)
let changedTabs = {};
const RESETTABLE = ["goal", "norms", "search"];
function resetBar(tab) {
  tab = tab || document.querySelector("#tabs .active")?.dataset.tab;
  const own = RESETTABLE.includes(tab);
  $("resetNote").textContent = own ? (changedTabs[tab] ? "вкладка изменена" : "заводские настройки") : "";
  $("resetTab").style.display = own ? "" : "none";
  $("resetTab").disabled = !changedTabs[tab];
  $("resetTab").onclick = () => resetSettings(tab);
  $("resetAll").disabled = !Object.values(changedTabs).some(Boolean);
}

// Наборы данных (вкладка «Данные»): список, вариант с другими бригадами, загрузка файла.
const BRIGADE_FIELDS = [
  { key: "count", title: "бригад всего", min: 1 },
  { key: "emergency", title: "с аварийным допуском", min: 0 },
  { key: "connect", title: "с подключениями", min: 0 },
  { key: "local", title: "с ремонтом (локальные)", min: 0 },
  { key: "seed", title: "сид", min: 0 },
];
let regionTitles = {};

function brigadeForm(id, values) {
  $(id).innerHTML = BRIGADE_FIELDS.map((f) => `<div class="kv"><span>${f.title}</span>
    <span><input data-b="${f.key}" type="number" min="${f.min}" step="1" value="${values[f.key]}"
      style="width:70px;text-align:right"></span></div>`).join("");
}

function brigadeValues(id) {
  const out = {};
  document.querySelectorAll(`#${id} input[data-b]`).forEach((i) => { out[i.dataset.b] = Number(i.value); });
  return out;
}

function brigadeText(b) {
  if (b.source !== "generated") return "по контрольному дню";
  return `${b.count}: аварийных ${b.emergency}, подключения ${b.connect}, ремонт ${b.local}; сид ${b.seed}`;
}

function flash(id, text, ok) {
  $(id).textContent = text;
  $(id).className = ok ? "saved" : "delta-bad";
  if (ok) setTimeout(() => { $(id).textContent = ""; }, 6000);
}

async function loadDatasets() {
  const { datasets } = await json("/api/datasets");
  const kind = (d) => d.builtin ? "встроенный" : d.base ? `вариант набора «${regionTitles[d.base] || d.base}»` : "загружен из файла";
  $("datasetList").innerHTML = datasets.map((d) => `<tr>
    <td><a href="/?region=${encodeURIComponent(d.id)}" title="Открыть на главной">${d.title}</a>
      <div class="hint" style="margin:0">${kind(d)}</div></td>
    <td>${brigadeText(d.brigades)}</td>
    <td class="n">${d.created || ""}</td>
    <td class="n">${d.builtin ? "" : `<button data-remove="${d.id}" title="Удалить набор">×</button>`}</td></tr>`).join("");
  document.querySelectorAll("[data-remove]").forEach((el) => {
    el.onclick = async () => {
      if (!confirm("Удалить набор? Загруженный файл тоже будет удалён.")) return;
      const r = await fetch(`/api/datasets/${encodeURIComponent(el.dataset.remove)}`, { method: "DELETE" });
      const out = await r.json();
      flash("datasetSaved", r.ok ? "набор удалён" : out.detail, r.ok);
      await refreshRegions();
    };
  });
  $("genBase").innerHTML = datasets.map((d) => `<option value="${d.id}">${d.title}</option>`).join("");
}

// Параметры варианта по умолчанию — бригады выбранного набора: меняют одно число, а не всё с нуля.
async function prefillBrigades() {
  const base = $("genBase").value;
  const d = await json(`/api/dataset/${encodeURIComponent(base)}`);
  if ($("genBase").value !== base) return;            // пока грузили, выбрали другой набор
  const has = (title) => d.engineers.filter((e) => e.skills.includes(title)).length;
  brigadeForm("genForm", { count: d.counts.engineers, emergency: has("Аварийные работы"),
    connect: has("Подключения и дозаказы"), local: has("Локальные работы"), seed: 1 });
  $("genForm").dataset.base = base;
}

// «Новый набор»: заявки — из набора или из файла (свой или пример), бригады — по параметрам; одна кнопка.
// refactor: Claude, 26.09.2026 — два блока с одинаковыми полями бригад сведены в один (просьба Михаила)
let sampleFile = null;                       // выбранный пример: {name, bytes}
const FILE_BRIGADES = { count: 12, emergency: 4, connect: 12, local: 12, seed: 1 };
const fromFile = () => document.querySelector("input[name=ndSource]:checked").value === "file";

async function switchSource() {
  $("ndFile").classList.toggle("hidden", !fromFile());
  $("genBase").disabled = fromFile();
  if (fromFile()) brigadeForm("genForm", FILE_BRIGADES);      // у файла контрольного дня нет — типовой состав
  else await prefillBrigades();
}

async function useSample() {
  const r = await fetch("/api/datasets/sample");
  sampleFile = { name: "пример_заявок.csv", bytes: new Uint8Array(await r.arrayBuffer()) };
  $("upFile").value = "";
  $("sampleNote").textContent = "Выбран пример: пример_заявок.csv — 37 заявок дня Востока с координатами, офис в файле.";
}

async function createDataset() {
  if (!fromFile()) {
    const body = { base: $("genBase").value, ...brigadeValues("genForm"), title: $("genTitle").value || null };
    const r = await fetch("/api/datasets/generate", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const out = await r.json();
    if (!r.ok) return flash("genSaved", out.detail, false);
    flash("genSaved", `создан: ${out.dataset.title}`, true);
    $("genTitle").value = "";
    return refreshRegions();
  }
  const own = $("upFile").files[0];
  const file = own ? { name: own.name, bytes: new Uint8Array(await own.arrayBuffer()) } : sampleFile;
  if (!file) return flash("genSaved", "выберите файл или нажмите «Взять пример»", false);
  let binary = "";
  for (let i = 0; i < file.bytes.length; i += 0x8000) binary += String.fromCharCode(...file.bytes.subarray(i, i + 0x8000));
  $("genSaved").textContent = "разбираем файл; адреса без координат ищем геокодером…";
  $("genSaved").className = "hint";
  const body = { filename: file.name, content: btoa(binary), office: $("upOffice").value || null,
                 brigades: brigadeValues("genForm"), title: $("genTitle").value || null };
  const r = await fetch("/api/datasets/upload", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const out = await r.json();
  if (!r.ok) { $("upReport").innerHTML = ""; return flash("genSaved", out.detail, false); }
  flash("genSaved", `загружен: ${out.dataset.title}`, true);
  const rep = out.report;
  $("upReport").innerHTML = `
    <div class="kv"><span>Строк заявок</span><span>${rep.rows}</span></div>
    <div class="kv"><span>Принято</span><span>${rep.accepted}</span></div>
    <div class="kv"><span>Координаты: из файла / справочник / геокодер / не найдено</span>
      <span>${rep.from_file} / ${rep.from_cache} / ${rep.online} / ${rep.not_found}</span></div>
    ${rep.skipped.length ? `<div class="hint">Пропущено ${rep.skipped.length}:</div>` +
      rep.skipped.slice(0, 30).map((x) => `<div class="hint" style="margin:2px 0">строка ${x.row}, заявка ${x.id}: ${x.reason}</div>`).join("")
      + (rep.skipped.length > 30 ? `<div class="hint">…и ещё ${rep.skipped.length - 30}</div>` : "") : ""}`;
  $("genTitle").value = "";
  await refreshRegions();
}

async function loadSummary() {
  const region = $("sumRegion").value;
  const s = await (await fetch(`/api/analysis/${encodeURIComponent(region)}`)).json();
  const u = s.urgent, h = (u.target_min / 60).toFixed(0);
  $("summary").innerHTML = `
    <table class="norm"><thead><tr><th>Работы</th><th class="n">Заявок</th><th class="n">Часов работы</th>
      <th class="n">Бригад с навыком</th><th class="n">Часов их смен</th></tr></thead>
      <tbody>${s.skills.map((k) => `<tr><td>${k.skill}</td><td class="n">${k.requests}</td><td class="n">${k.work_hours}</td>
        <td class="n">${k.engineers}</td><td class="n">${k.shift_hours}</td></tr>`).join("")}</tbody></table>
    <div class="hint">Бригада с несколькими навыками считается в каждой строке: часы смен — верхняя оценка.</div>
    ${u.urgent && u.crews ? `
    <div class="kv"><span>Аварий / бригад с аварийным допуском</span><span>${u.urgent} / ${u.crews}</span></div>
    <div class="kv"><span>Если к каждой аварии сразу едет ближайшая бригада<br><span class="hint">нижняя граница по
      кратчайшему пути и наименьшему за день времени в пути: быстрее ни в одном плане</span></span><span>в ${h} ч: ${u.alone_within} из ${u.urgent}, дольше всех — ${u.alone_max} мин</span></div>
    <div class="kv"><span>Если бригады с допуском с утра делают только аварии<br><span class="hint">оценка: с концом смены,
      но без запасов, транспорта и их плановых заявок — не проверенный план</span></span>
      <span>в ${h} ч: ${u.only_within} из ${u.urgent}${u.only_max !== null ? `, дольше всех — ${u.only_max} мин` : ""}${
        u.only_unassigned ? `, не помещаются в смены: ${u.only_unassigned}` : ""}</span></div>`
    : `<div class="hint">Аварий или бригад с аварийным допуском в наборе нет.</div>`}`;
}

// Списки наборов в админке (Данные, пробки, «что если», последний расчёт) — с подписями, как на главной.
async function refreshRegions() {
  const { regions, titles } = await json("/api/regions");
  regionTitles = titles || {};
  const options = regions.map((r) => `<option value="${r}">${regionTitles[r] || r}</option>`).join("");
  for (const id of ["statRegion", "whatifRegion", "trafficRegion", "sumRegion"]) {
    const keep = $(id).value;
    $(id).innerHTML = options;
    if (regions.includes(keep)) $(id).value = keep;
  }
  const keepBase = $("genBase").value;
  await loadDatasets();
  if (keepBase && regions.includes(keepBase)) $("genBase").value = keepBase;
}

// Каждый блок загружается отдельно: сбой одного (например, сервер старее страницы) не оставляет пустыми остальные
// и виден строкой вверху, а не молча пустыми списками. refactor: Claude, 23.09.2026
async function guarded(title, fn) {
  try {
    await fn();
  } catch (e) {
    const box = $("adminErrors");
    box.classList.remove("hidden");
    box.innerHTML += `<div>Не загрузилось: ${title} (${e.message}). Если сервер запущен до обновления кода — перезапустите его.</div>`;
  }
}

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} — ответ ${r.status}`);
  return r.json();
}

// Заводские умолчания: метка у изменённых вкладок и сброс вкладки или всего. refactor: Claude, 23.09.2026
function markChanged(changed) {
  changedTabs = changed || {};
  for (const [tab, differs] of Object.entries(changedTabs)) {
    document.querySelector(`#tabs [data-tab=${tab}]`)?.classList.toggle("changed", differs);
  }
  resetBar();
}

async function resetSettings(scope) {
  const what = scope === "all" ? "все вкладки" : "эту вкладку";
  if (!confirm(`Вернуть заводские настройки (${what})? Сохранённые изменения будут потеряны, планы пересчитаются.`)) return;
  const r = await fetch("/api/settings/reset", { method: "POST", headers: { "Content-Type": "application/json" },
                                                 body: JSON.stringify({ scope }) });
  if (!r.ok) { alert((await r.json()).detail); return; }
  location.reload();
}

(async function start() {
  showTab(location.hash.slice(1));
  document.querySelectorAll("#tabs [data-tab]").forEach((b) => {
    b.onclick = () => { history.replaceState(null, "", "#" + b.dataset.tab); showTab(b.dataset.tab); };
  });
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
  await guarded("настройки", async () => {
    data = await json("/api/settings");
    renderNormatives(data.normatives);
    renderAssumptions(data.assumptions);
    search = data.search;
    renderWeights(data.weights);
    renderSearch();
    bindSearchFields();
    markChanged(data.changed);
  });
  $("resetAll").onclick = () => resetSettings("all");
  $("saveSearch").onclick = saveSearch;
  const add = (kind) => () => {
    const seeds = search[kind].members.map((m) => m.seed);
    search[kind].members.push({ mode: "uniform", seed: Math.max(0, ...seeds) + 1 });
    renderSearch();
  };
  $("addPlanMember").onclick = add("plan");
  $("addEventMember").onclick = add("event");
  $("saveNorm").onclick = () => save("normatives");
  $("saveWeights").onclick = saveWeights;
  $("saveAsm").onclick = () => save("assumptions");
  $("loadStats").onclick = loadStats;
  $("runWhatif").onclick = runWhatif;
  $("whatifFull").onchange = () => guarded("оценка времени «что если»", whatifNote);
  $("whatifRegion").addEventListener("change", () => guarded("оценка времени «что если»", whatifNote));
  $("loadSummary").onclick = loadSummary;
  $("trafficRegion").onchange = loadTrafficState;
  $("applyTraffic").onclick = applyTraffic;
  $("genBase").onchange = prefillBrigades;
  $("genCreate").onclick = createDataset;
  $("useSample").onclick = useSample;
  $("upFile").onchange = () => { sampleFile = null; $("sampleNote").textContent = ""; };
  document.querySelectorAll("input[name=ndSource]").forEach((el) => { el.onchange = switchSource; });
  await guarded("наборы данных", refreshRegions);
  await guarded("пробки", loadTrafficState);
  await guarded("бригады выбранного набора", prefillBrigades);
  await guarded("оценка времени «что если»", whatifNote);
})();
