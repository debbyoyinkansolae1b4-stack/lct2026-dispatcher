// Интерактивный тур по приложению: кнопка «?» в шапке, темы по всему функционалу главной и /admin (шаг 13 PORT-PLAN).
// Тексты тем — web/tour-content.js. Шаг подсвечивает элемент и показывает подсказку; если шагу нужно действие
// пользователя (построить план, кликнуть точку), тур ждёт его и идёт дальше сам. Без внешних библиотек.
// refactor: Claude, 26.09.2026
(function () {
  const TOPICS = window.TOUR_TOPICS || [];
  const PAGE = location.pathname.startsWith("/admin") ? "admin" : "main";
  let topic = null, index = 0, timer = null, spot = null, pop = null, current = null;

  const $q = (sel) => (sel ? document.querySelector(sel) : null);
  const visible = (el) => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== "hidden";
  const ready = (sel) => { const el = $q(sel); return visible(el) ? el : null; };

  function button() {
    const box = document.querySelector("header .controls");
    if (!box || document.getElementById("tourButton")) return;
    const b = document.createElement("button");
    b.id = "tourButton";
    b.title = "Как пользоваться: интерактивный тур по приложению";
    b.setAttribute("aria-label", b.title);
    b.textContent = "?";
    b.onclick = menu;
    box.appendChild(b);
  }

  function menu() {
    stop();
    const wrap = document.createElement("div");
    wrap.className = "tour-menu";
    const group = (page, title) => `<h3>${title}</h3>` + TOPICS.filter((t) => t.page === page).map((t) =>
      `<button data-topic="${t.id}"><b>${t.title}</b><span>${t.summary}</span></button>`).join("");
    wrap.innerHTML = `<div class="tour-card"><button class="tour-x" title="Закрыть">×</button>
      <h2>Как пользоваться</h2><p class="hint">Выберите тему: тур подсветит, куда смотреть и где кликнуть. Шаги, которым
      нужно ваше действие, ждут его. Выйти можно в любой момент. Полное описание решения —
      <a href="/documentation" target="_blank">документация</a>.</p>
      ${group("main", "Главная — план диспетчера")}${group("admin", "Администрирование")}</div>`;
    document.body.appendChild(wrap);
    wrap.querySelector(".tour-x").onclick = () => wrap.remove();
    wrap.onclick = (e) => { if (e.target === wrap) wrap.remove(); };
    wrap.querySelectorAll("[data-topic]").forEach((b) => {
      b.onclick = () => { wrap.remove(); start(b.dataset.topic, 0); };
    });
  }

  function start(id, step) {
    const t = TOPICS.find((x) => x.id === id);
    if (!t) return;
    if (t.page !== PAGE) {                  // тема другой страницы — переход с продолжением тура
      location.href = (t.page === "admin" ? "/admin" : "/") + `?tour=${id}&step=${step}` + (t.tab ? `#${t.tab}` : "");
      return;
    }
    topic = t;
    show(step);
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    [spot, pop].forEach((x) => x && x.remove());
    spot = pop = current = null;
    topic = null;
    window.removeEventListener("scroll", place, true);
    window.removeEventListener("resize", place);
  }

  function openTab(tab) {
    const b = tab && document.querySelector(`#tabs [data-tab=${tab}]`);
    if (b && !b.classList.contains("active")) b.click();
  }

  function show(i) {
    clearInterval(timer);
    index = Math.max(0, Math.min(i, topic.steps.length - 1));
    const st = topic.steps[index];
    openTab(st.tab || topic.tab);
    if (st.need && !ready(st.need)) {         // шагу нужно состояние (план построен и т. п.) — просим действие и ждём
      render($q(st.needEl) || $q(st.el), st.title, st.needText, false);
      timer = setInterval(() => { if (ready(st.need)) show(index); }, 400);
      return;
    }
    const el = ready(st.el) || $q(st.el);
    if (!el) {                                // элемента нет на экране — объясняем и ждём, пока появится
      render(null, st.title, st.missing || "Этот элемент появится после предыдущего действия.", false);
      timer = setInterval(() => { if (ready(st.el)) show(index); }, 400);
      return;
    }
    el.scrollIntoView({ block: "center", behavior: "smooth" });
    render(el, st.title, st.text, !st.advance);
    if (st.advance) {                         // шаг ждёт действия пользователя — дальше сам
      timer = setInterval(() => {
        if (ready(st.advance)) { clearInterval(timer); setTimeout(() => next(), 350); }
      }, 400);
    }
  }

  function next() { if (index + 1 < topic.steps.length) show(index + 1); else finish(); }

  function finish() {
    const done = topic;
    stop();
    const nextTopic = TOPICS[TOPICS.indexOf(done) + 1] || null;
    pop = document.createElement("div");
    pop.className = "tour-pop tour-center";
    pop.innerHTML = `<b>Тема «${done.title}» пройдена</b>
      <div class="tour-actions"><button data-a="menu">Другие темы</button>
      ${nextTopic ? `<button class="primary" data-a="next">Дальше: «${nextTopic.title}»</button>` : ""}
      <button data-a="close">Закрыть</button></div>`;
    document.body.appendChild(pop);
    pop.querySelector("[data-a=menu]").onclick = menu;
    pop.querySelector("[data-a=close]").onclick = stop;
    if (nextTopic) pop.querySelector("[data-a=next]").onclick = () => { stop(); start(nextTopic.id, 0); };
  }

  function render(el, title, text, canNext) {
    if (!spot) { spot = document.createElement("div"); spot.className = "tour-spot"; document.body.appendChild(spot); }
    if (!pop) { pop = document.createElement("div"); pop.className = "tour-pop"; document.body.appendChild(pop); }
    current = el;
    const n = topic.steps.length;
    pop.classList.toggle("tour-center", !el);
    pop.innerHTML = `<div class="tour-head"><span>${topic.title} · ${index + 1} / ${n}</span>
        <button class="tour-x" title="Закончить тур">×</button></div>
      <b>${title}</b><div class="tour-text">${text}</div>
      <div class="tour-actions">
        <button data-a="prev" ${index === 0 ? "disabled" : ""}>Назад</button>
        ${canNext ? `<button class="primary" data-a="next">${index + 1 < n ? "Далее" : "Готово"}</button>`
                  : `<span class="hint">ждём вашего действия…</span><button data-a="skip">Пропустить</button>`}
      </div>`;
    pop.querySelector(".tour-x").onclick = stop;
    pop.querySelector("[data-a=prev]").onclick = () => show(index - 1);
    const nx = pop.querySelector("[data-a=next]") || pop.querySelector("[data-a=skip]");
    nx.onclick = next;
    spot.style.display = el ? "block" : "none";
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    place();
    setTimeout(place, 400);                   // после плавной прокрутки
  }

  // Подсветка и подсказка следуют за элементом при прокрутке и смене размеров окна.
  function place() {
    if (!pop) return;
    if (!current || !visible(current)) { spot && (spot.style.display = "none"); return; }
    const r = current.getBoundingClientRect(), pad = 6, gap = 12;
    Object.assign(spot.style, { display: "block", left: `${r.left - pad}px`, top: `${r.top - pad}px`,
                                width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px` });
    const pw = pop.offsetWidth, ph = pop.offsetHeight, W = innerWidth, H = innerHeight;
    let top = r.bottom + gap, left = r.left;
    if (top + ph > H - 8) top = r.top - ph - gap;                 // снизу не помещается — сверху
    if (top < 8) {                                                // и сверху нет — сбоку или поверх края элемента
      top = Math.min(Math.max(8, r.top), H - ph - 8);
      left = r.right + gap + pw < W ? r.right + gap : Math.max(8, r.left - pw - gap);
    }
    left = Math.min(Math.max(8, left), W - pw - 8);
    Object.assign(pop.style, { top: `${top}px`, left: `${left}px` });
  }

  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && topic) stop(); });
  window.addEventListener("load", () => {
    button();
    const q = new URLSearchParams(location.search);
    if (q.get("tour")) {
      history.replaceState(null, "", location.pathname + location.hash);
      // ?tour=menu — список тем (ссылка для экспертов), иначе — тема с шага step
      setTimeout(() => (q.get("tour") === "menu" ? menu() : start(q.get("tour"), Number(q.get("step") || 0))), 600);
    }
  });
})();
