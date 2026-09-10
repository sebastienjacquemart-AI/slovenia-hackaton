// Submission review dashboard — vanilla JS, no build step.
// Run `python3 -m http.server` from the repo root and open /dashboard/.

const SVG_NS = "http://www.w3.org/2000/svg";
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const EXPECTED_HEADER = ["id", "sales_count_p25", "sales_count_p50", "sales_count_p75", "sales_count_p95"];

const state = {
  stores: new Map(),        // store_id -> {city, department, store_type, store_cluster}
  products: new Map(),      // product_id -> {product_family, product_class, is_perishable}
  events: [],                // [{date, event_type, scope, location, is_transferred}]
  testById: new Map(),      // id -> {date, store_id, product_id, promotion}
  seriesIndex: new Map(),   // "store|product" -> [{id, date, promotion}, ...] chronological
  baselineById: new Map(),  // id -> {p25,p50,p75,p95}
  submissionById: null,     // id -> {p25,p50,p75,p95}, set once a file is loaded
};

// ---------- CSV parsing ----------

function splitCSVLine(line) {
  if (line.indexOf('"') === -1) return line.split(",");
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++; } else { inQuotes = false; }
      } else {
        cur += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      out.push(cur);
      cur = "";
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out;
}

function forEachRow(text, cb) {
  const lines = text.split("\n");
  for (let i = 1; i < lines.length; i++) {
    let line = lines[i];
    if (!line) continue;
    if (line.endsWith("\r")) line = line.slice(0, -1);
    cb(splitCSVLine(line));
  }
}

async function loadText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.text();
}

// ---------- Loading context data ----------

async function loadContext() {
  const [storeTxt, itemTxt, eventsTxt] = await Promise.all([
    loadText("../gustavo_context/store_info.csv"),
    loadText("../gustavo_context/item_catalogue.csv"),
    loadText("../gustavo_context/events_and_holidays.csv"),
  ]);

  forEachRow(storeTxt, (f) => {
    state.stores.set(f[0], { city: f[1], department: f[2], store_type: f[3], store_cluster: f[4] });
  });
  forEachRow(itemTxt, (f) => {
    state.products.set(f[0], { product_family: f[1], product_class: f[2], is_perishable: f[3] });
  });
  forEachRow(eventsTxt, (f) => {
    state.events.push({ date: f[0], event_type: f[1], scope: f[2], location: f[3], is_transferred: f[4] });
  });

  const testTxt = await loadText("../data/test.csv");
  forEachRow(testTxt, (f) => {
    const [id, date, storeId, productId, promotion] = f;
    state.testById.set(id, { date, store_id: storeId, product_id: productId, promotion });
    const key = storeId + "|" + productId;
    let arr = state.seriesIndex.get(key);
    if (!arr) { arr = []; state.seriesIndex.set(key, arr); }
    arr.push({ id, date, promotion });
  });

  const baselineTxt = await loadText("../predictions_baseline.csv");
  forEachRow(baselineTxt, (f) => {
    state.baselineById.set(f[0], { p25: +f[1], p50: +f[2], p75: +f[3], p95: +f[4] });
  });
}

// ---------- Submission validation ----------

function validateAndIndexSubmission(text) {
  const lines = text.split("\n");
  const headerLine = (lines[0] || "").replace(/\r$/, "");
  const header = splitCSVLine(headerLine).map((h) => h.trim());
  const schemaOk = header.length === EXPECTED_HEADER.length &&
    EXPECTED_HEADER.every((h, i) => h === header[i]);

  const map = new Map();
  let monotonicViolations = 0;

  for (let i = 1; i < lines.length; i++) {
    let line = lines[i];
    if (!line) continue;
    if (line.endsWith("\r")) line = line.slice(0, -1);
    const f = splitCSVLine(line);
    if (f.length < 5) continue;
    const p25 = +f[1], p50 = +f[2], p75 = +f[3], p95 = +f[4];
    map.set(f[0], { p25, p50, p75, p95 });
    if (!(p25 <= p50 && p50 <= p75 && p75 <= p95)) monotonicViolations++;
  }

  let missing = 0, extra = 0;
  for (const id of state.testById.keys()) if (!map.has(id)) missing++;
  for (const id of map.keys()) if (!state.testById.has(id)) extra++;

  return { schemaOk, header, map, monotonicViolations, missing, extra, total: map.size };
}

function renderValidationStrip(v) {
  const card = document.getElementById("validationCard");
  const list = document.getElementById("validationList");
  list.textContent = "";
  card.style.display = "block";

  function addItem(status, text) {
    const div = document.createElement("div");
    div.className = "status-item status-" + status;
    const icon = document.createElement("span");
    icon.className = "status-icon";
    icon.textContent = status === "good" ? "✓" : status === "warning" ? "⚠" : "✕";
    const label = document.createElement("span");
    label.textContent = text;
    div.appendChild(icon);
    div.appendChild(label);
    list.appendChild(div);
  }

  addItem(v.schemaOk ? "good" : "critical",
    v.schemaOk ? "Schema OK — columns match sample_submission.csv" :
      `Schema mismatch — got [${v.header.join(", ")}]`);

  const idsOk = v.missing === 0 && v.extra === 0;
  addItem(idsOk ? "good" : "warning",
    `IDs: ${v.total - v.extra}/${state.testById.size} matched` +
    (v.missing ? `, ${v.missing} missing` : "") +
    (v.extra ? `, ${v.extra} unexpected` : ""));

  addItem(v.monotonicViolations === 0 ? "good" : "critical",
    v.monotonicViolations === 0 ?
      "Quantile order OK — P25 ≤ P50 ≤ P75 ≤ P95 on every row" :
      `${v.monotonicViolations} row(s) violate P25 ≤ P50 ≤ P75 ≤ P95`);
}

// ---------- Dropdowns & info panels ----------

function storeLabel(id) {
  const s = state.stores.get(id);
  return `${id} · ${s.city} (${s.store_type}, cluster ${s.store_cluster})`;
}
function productLabel(id) {
  const p = state.products.get(id);
  return `${id} · ${p.product_family}${p.is_perishable === "1" ? " · perishable" : ""}`;
}

function populateDropdowns() {
  const storeSelect = document.getElementById("storeSelect");
  const productSelect = document.getElementById("productSelect");

  storeSelect.textContent = "";
  [...state.stores.keys()].sort((a, b) => +a - +b).forEach((id) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = storeLabel(id);
    storeSelect.appendChild(opt);
  });

  productSelect.textContent = "";
  [...state.products.keys()].sort((a, b) => +a - +b).forEach((id) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = productLabel(id);
    productSelect.appendChild(opt);
  });

  storeSelect.disabled = false;
  productSelect.disabled = false;
  storeSelect.addEventListener("change", onSelectionChange);
  productSelect.addEventListener("change", onSelectionChange);
}

function renderInfoPanels(storeId, productId) {
  const s = state.stores.get(storeId);
  const p = state.products.get(productId);
  document.getElementById("storeInfo").innerHTML =
    `<strong>${s.city}</strong>, ${s.department}<br>Format ${s.store_type} · cluster ${s.store_cluster}`;
  document.getElementById("productInfo").innerHTML =
    `<strong>${p.product_family}</strong> (class ${p.product_class})<br>` +
    (p.is_perishable === "1" ? "Perishable" : "Non-perishable");
}

function onSelectionChange() {
  const storeId = document.getElementById("storeSelect").value;
  const productId = document.getElementById("productSelect").value;
  renderInfoPanels(storeId, productId);
  renderChart(storeId, productId);
}

// ---------- Holiday matching ----------

function eventAppliesToStore(event, store) {
  if (event.is_transferred === "True") return false;
  if (event.scope === "National") return true;
  if (event.scope === "Regional") return event.location === store.department;
  if (event.scope === "Local") return event.location === store.city;
  return false;
}

// ---------- Chart ----------

const MARGIN = { top: 24, right: 20, bottom: 34, left: 50 };
const WIDTH = 820, HEIGHT = 340;
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;

function niceMax(v) {
  if (v <= 0) return 10;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const norm = v / mag;
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return step * mag;
}

function formatDate(dateStr) {
  const [y, m, d] = dateStr.split("-");
  return MONTHS[+m - 1] + " " + (+d);
}

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

function renderChart(storeId, productId) {
  const svg = document.getElementById("chart");
  svg.textContent = "";

  const points = state.seriesIndex.get(storeId + "|" + productId) || [];
  if (points.length === 0) {
    const empty = svgEl("text", { x: WIDTH / 2, y: HEIGHT / 2, "text-anchor": "middle", fill: "var(--text-muted)" });
    empty.textContent = "No data for this store × product combination.";
    svg.appendChild(empty);
    return;
  }

  const rows = points.map((pt) => ({
    ...pt,
    sub: state.submissionById ? state.submissionById.get(pt.id) : null,
    base: state.baselineById.get(pt.id),
  }));

  let yMax = 0;
  rows.forEach((r) => {
    if (r.sub) yMax = Math.max(yMax, r.sub.p95);
    if (r.base) yMax = Math.max(yMax, r.base.p95);
  });
  yMax = niceMax(yMax * 1.1);

  const x = (i) => MARGIN.left + (rows.length === 1 ? 0 : (i / (rows.length - 1)) * PLOT_W);
  const y = (v) => MARGIN.top + PLOT_H - (v / yMax) * PLOT_H;

  // gridlines + y ticks
  const tickCount = 4;
  for (let t = 0; t <= tickCount; t++) {
    const v = (yMax / tickCount) * t;
    const gy = y(v);
    svg.appendChild(svgEl("line", { x1: MARGIN.left, x2: WIDTH - MARGIN.right, y1: gy, y2: gy, stroke: "var(--grid)", "stroke-width": 1 }));
    const label = svgEl("text", { x: MARGIN.left - 8, y: gy + 4, "text-anchor": "end", fill: "var(--text-muted)", "font-size": 11 });
    label.textContent = Math.round(v).toLocaleString();
    svg.appendChild(label);
  }

  // x axis baseline
  svg.appendChild(svgEl("line", { x1: MARGIN.left, x2: WIDTH - MARGIN.right, y1: MARGIN.top + PLOT_H, y2: MARGIN.top + PLOT_H, stroke: "var(--axis)", "stroke-width": 1 }));

  // x tick labels, every 4th date
  rows.forEach((r, i) => {
    if (i % 4 !== 0 && i !== rows.length - 1) return;
    const label = svgEl("text", { x: x(i), y: MARGIN.top + PLOT_H + 18, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11 });
    label.textContent = formatDate(r.date);
    svg.appendChild(label);
  });

  // holiday markers
  const store = state.stores.get(storeId);
  const rangeStart = rows[0].date, rangeEnd = rows[rows.length - 1].date;
  const holidaysByDate = new Map();
  state.events.forEach((ev) => {
    if (ev.date < rangeStart || ev.date > rangeEnd) return;
    if (!eventAppliesToStore(ev, store)) return;
    holidaysByDate.set(ev.date, ev);
  });
  rows.forEach((r, i) => {
    const ev = holidaysByDate.get(r.date);
    if (!ev) return;
    const gx = x(i);
    svg.appendChild(svgEl("line", {
      x1: gx, x2: gx, y1: MARGIN.top, y2: MARGIN.top + PLOT_H,
      stroke: "var(--text-muted)", "stroke-width": 1, "stroke-dasharray": "3,3",
    }));
    const label = svgEl("text", { x: gx, y: MARGIN.top - 8, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 10 });
    label.textContent = ev.event_type;
    svg.appendChild(label);
  });

  // submission band + line
  if (state.submissionById) {
    const withSub = rows.map((r, i) => ({ i, r })).filter(({ r }) => r.sub);
    if (withSub.length > 0) {
      const outerPath = bandPath(withSub, x, y, (r) => r.sub.p25, (r) => r.sub.p95);
      svg.appendChild(svgEl("path", { d: outerPath, fill: "var(--series-1)", "fill-opacity": 0.10, stroke: "none" }));
      const innerPath = bandPath(withSub, x, y, (r) => r.sub.p25, (r) => r.sub.p75);
      svg.appendChild(svgEl("path", { d: innerPath, fill: "var(--series-1)", "fill-opacity": 0.18, stroke: "none" }));
      const linePath = linePathFor(withSub, x, y, (r) => r.sub.p50);
      svg.appendChild(svgEl("path", { d: linePath, fill: "none", stroke: "var(--series-1)", "stroke-width": 2 }));
    }
  }

  // baseline dashed line
  const withBase = rows.map((r, i) => ({ i, r })).filter(({ r }) => r.base);
  if (withBase.length > 0) {
    const basePath = linePathFor(withBase, x, y, (r) => r.base.p50);
    svg.appendChild(svgEl("path", { d: basePath, fill: "none", stroke: "var(--series-2)", "stroke-width": 2, "stroke-dasharray": "6,4" }));
  }

  // promotion markers
  rows.forEach((r, i) => {
    if (r.promotion !== "True") return;
    const gx = x(i), gy = MARGIN.top + PLOT_H;
    svg.appendChild(svgEl("circle", { cx: gx, cy: gy - 8, r: 4, fill: "var(--series-3)" }));
  });

  // crosshair + hover hit area
  const hit = svgEl("rect", { x: MARGIN.left, y: MARGIN.top, width: PLOT_W, height: PLOT_H, fill: "transparent" });
  svg.appendChild(hit);
  const crosshair = svgEl("line", { y1: MARGIN.top, y2: MARGIN.top + PLOT_H, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  svg.appendChild(crosshair);

  const tooltip = document.getElementById("tooltip");
  hit.addEventListener("pointermove", (e) => {
    const rect = svg.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * WIDTH;
    let idx = Math.round(((mx - MARGIN.left) / PLOT_W) * (rows.length - 1));
    idx = Math.max(0, Math.min(rows.length - 1, idx));
    const r = rows[idx];
    crosshair.setAttribute("x1", x(idx));
    crosshair.setAttribute("x2", x(idx));
    crosshair.setAttribute("visibility", "visible");
    showTooltip(tooltip, r, holidaysByDate.get(r.date), e.clientX - rect.left, e.clientY - rect.top);
  });
  hit.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    tooltip.style.display = "none";
  });

  renderLegend();
}

function bandPath(items, x, y, lowFn, highFn) {
  const forward = items.map(({ i, r }) => `${x(i)},${y(lowFn(r))}`);
  const backward = items.slice().reverse().map(({ i, r }) => `${x(i)},${y(highFn(r))}`);
  return `M ${forward.join(" L ")} L ${backward.join(" L ")} Z`;
}

function linePathFor(items, x, y, valFn) {
  return items.map(({ i, r }, idx) => `${idx === 0 ? "M" : "L"} ${x(i)},${y(valFn(r))}`).join(" ");
}

function showTooltip(tooltip, r, holiday, left, top) {
  tooltip.textContent = "";
  const dateRow = document.createElement("div");
  dateRow.className = "tt-date";
  dateRow.textContent = r.date + (r.promotion === "True" ? " · promotion" : "") + (holiday ? ` · ${holiday.event_type}` : "");
  tooltip.appendChild(dateRow);

  function addRow(color, label, value) {
    const row = document.createElement("div");
    row.className = "tt-row";
    const left = document.createElement("span");
    left.className = "tt-label";
    if (color) {
      const key = document.createElement("span");
      key.className = "tt-key";
      key.style.background = color;
      left.appendChild(key);
    }
    left.appendChild(document.createTextNode(label));
    const right = document.createElement("span");
    right.className = "tt-value";
    right.textContent = value;
    row.appendChild(left);
    row.appendChild(right);
    tooltip.appendChild(row);
  }

  if (r.sub) {
    addRow("var(--series-1)", "Submission P25–P95", `${Math.round(r.sub.p25)}–${Math.round(r.sub.p95)}`);
    addRow(null, "Submission P50", Math.round(r.sub.p50));
  }
  if (r.base) {
    addRow("var(--series-2)", "Baseline P50", Math.round(r.base.p50));
  }

  tooltip.style.left = (left + 12) + "px";
  tooltip.style.top = (top + 12) + "px";
  tooltip.style.display = "block";
}

function renderLegend() {
  const legend = document.getElementById("legend");
  legend.textContent = "";
  const items = [
    { swatch: "line", color: "var(--series-1)", label: "Submission (P50, P25–P95 band)" },
    { swatch: "dashed", color: "var(--series-2)", label: "Baseline P50" },
    { swatch: "dot", color: "var(--series-3)", label: "Promotion day" },
  ];
  items.forEach((it) => {
    const div = document.createElement("div");
    div.className = "legend-item";
    const svg = svgEl("svg", { width: 16, height: 10, class: "legend-swatch" });
    if (it.swatch === "dot") {
      svg.appendChild(svgEl("circle", { cx: 8, cy: 5, r: 4, fill: it.color }));
    } else {
      svg.appendChild(svgEl("line", {
        x1: 0, x2: 16, y1: 5, y2: 5, stroke: it.color, "stroke-width": 2,
        "stroke-dasharray": it.swatch === "dashed" ? "6,4" : "none",
      }));
    }
    div.appendChild(svg);
    const label = document.createElement("span");
    label.textContent = it.label;
    div.appendChild(label);
    legend.appendChild(div);
  });
}

// ---------- Wiring ----------

if (typeof document !== "undefined") {
  document.getElementById("fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const status = document.getElementById("loadStatus");
    status.textContent = "Reading " + file.name + "…";
    const reader = new FileReader();
    reader.onload = () => {
      const result = validateAndIndexSubmission(reader.result);
      state.submissionById = result.map;
      renderValidationStrip(result);
      status.textContent = "Loaded " + file.name;
      const storeId = document.getElementById("storeSelect").value;
      const productId = document.getElementById("productSelect").value;
      if (storeId && productId) renderChart(storeId, productId);
    };
    reader.readAsText(file);
  });

  (async function init() {
    const status = document.getElementById("loadStatus");
    status.textContent = "Loading context data…";
    await loadContext();
    status.textContent = "";
    populateDropdowns();
    const storeId = document.getElementById("storeSelect").value;
    const productId = document.getElementById("productSelect").value;
    renderInfoPanels(storeId, productId);
    renderChart(storeId, productId);
  })();
}

// Exposed for the self-check in test_app.mjs — no effect in the browser.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { splitCSVLine, eventAppliesToStore, niceMax, formatDate, validateAndIndexSubmission, state };
}
