// Submission review dashboard — vanilla JS, no build step.
// Run `python3 -m http.server` from the repo root and open /dashboard/.

const SVG_NS = "http://www.w3.org/2000/svg";
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const EXPECTED_HEADER = ["id", "sales_count_p25", "sales_count_p50", "sales_count_p75", "sales_count_p95"];

// ponytail: coarse staple/discretionary split by product_family text (per
// Gustavo/Olivier's risk-tier notes — no ready-made staple flag exists in
// the data). Good enough to triage, not to automate stocking decisions on.
const STAPLE_FAMILIES = new Set([
  "BREAD/BAKERY", "DAIRY", "EGGS", "CLEANING", "HOME CARE",
  "PERSONAL CARE", "GROCERY I", "POULTRY", "PRODUCE",
]);

const FLAG_TYPES = {
  monotonicity: { label: "Quantile order violated (P25≤P50≤P75≤P95)", severity: "critical" },
  perishable_spike: { label: "Perishable: P95 ≫ P50 (stockout risk)", severity: "warning" },
  promo_no_lift: { label: "Promoted day, no forecast lift", severity: "warning" },
  holiday_flat: { label: "Holiday/pre-holiday treated as normal", severity: "warning" },
  essential_low: { label: "Essential item, P50 = 0", severity: "warning" },
  zero_forecast: { label: "Median forecast is 0", severity: "info" },
  wide_spread: { label: "Unusually wide P25–P95 spread", severity: "info" },
  baseline_deviation: { label: "Large deviation from baseline", severity: "info" },
};
const SEVERITY_RANK = { critical: 0, warning: 1, info: 2 };

// What a reviewer should actually do about each flag — Gustavo's "practical
// action view" ask, folded into the exceptions table rather than a second page.
const ACTION_BY_FLAG = {
  monotonicity: "Data integrity — fix/reject before use",
  perishable_spike: "Replenishment — perishable, verify upper-bound stock",
  promo_no_lift: "Promotion readiness — review the planned promo",
  holiday_flat: "Calendar readiness — check holiday/payday staffing & stock",
  essential_low: "Replenishment — essential item, confirm before it hits zero",
  zero_forecast: "Manager review — confirm zero is expected, not a data gap",
  wide_spread: "Manager review — high uncertainty, use judgement not the raw number",
  baseline_deviation: "Manager review — sanity-check the change vs. baseline",
};

const HISTORY_LOOKBACK_DAYS = 30;

const state = {
  stores: new Map(),        // store_id -> {city, department, store_type, store_cluster}
  products: new Map(),      // product_id -> {product_family, product_class, is_perishable}
  events: [],                // [{date, event_type, scope, location, is_transferred}]
  testById: new Map(),      // id -> {date, store_id, product_id, promotion}
  seriesIndex: new Map(),   // "store|product" -> [{id, date, promotion}, ...] chronological
  baselineById: new Map(),  // id -> {p25,p50,p75,p95}
  historyIndex: new Map(),  // "store|product" -> [{date, sales_count, promotion}, ...], last HISTORY_LOOKBACK_DAYS only
  submissionById: null,     // id -> {p25,p50,p75,p95}, set once a file is loaded
  shapIndex: new Map(),     // id -> Map("0.25"|"0.5"|"0.75"|"0.95" -> {base, prediction, contribs: {bucket: value}})
};

// Matches the bucket names written by scripts/export_shap_for_dashboard.py —
// SHAP contributions are additive, so these sum exactly to prediction - base.
const SHAP_BUCKETS = [
  { key: "sales_trend", label: "Recent sales trend" },
  { key: "promotion", label: "Promotion" },
  { key: "calendar", label: "Calendar (date/weekday)" },
  { key: "holiday_event", label: "Holiday & events" },
  { key: "product", label: "Product" },
  { key: "store", label: "Store" },
  { key: "oil_price", label: "Oil price" },
  { key: "traffic", label: "Store traffic" },
];
const SHAP_QUANTILES = ["0.25", "0.5", "0.75", "0.95"];

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
  let testMinDate = null;
  forEachRow(testTxt, (f) => {
    const [id, date, storeId, productId, promotion] = f;
    state.testById.set(id, { date, store_id: storeId, product_id: productId, promotion });
    const key = storeId + "|" + productId;
    let arr = state.seriesIndex.get(key);
    if (!arr) { arr = []; state.seriesIndex.set(key, arr); }
    arr.push({ id, date, promotion });
    if (testMinDate === null || date < testMinDate) testMinDate = date;
  });

  const [baselineTxt, historyTxt] = await Promise.all([
    loadText("../predictions_baseline.csv"),
    loadText("../data/sales_history.csv"),
  ]);
  forEachRow(baselineTxt, (f) => {
    state.baselineById.set(f[0], { p25: +f[1], p50: +f[2], p75: +f[3], p95: +f[4] });
  });

  const historyCutoff = testMinDate ? addDays(testMinDate, -HISTORY_LOOKBACK_DAYS) : null;
  forEachRow(historyTxt, (f) => {
    const [, date, storeId, productId, salesCount, promotion] = f;
    if (historyCutoff && date < historyCutoff) return;
    const key = storeId + "|" + productId;
    let arr = state.historyIndex.get(key);
    if (!arr) { arr = []; state.historyIndex.set(key, arr); }
    arr.push({ date, sales_count: +salesCount, promotion });
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

// ---------- SHAP explanation parsing ----------
// Expects the CSV written by scripts/export_shap_for_dashboard.py:
// id,quantile,base_value,prediction,contrib_<bucket>,... (one row per id×quantile)

function parseShapCsv(text) {
  const lines = text.split("\n");
  const header = splitCSVLine((lines[0] || "").replace(/\r$/, ""));
  const contribCols = header
    .map((h, i) => ({ h, i }))
    .filter(({ h }) => h.startsWith("contrib_"));

  const index = new Map();
  for (let i = 1; i < lines.length; i++) {
    let line = lines[i];
    if (!line) continue;
    if (line.endsWith("\r")) line = line.slice(0, -1);
    const f = splitCSVLine(line);
    if (f.length < header.length) continue;
    const id = f[0];
    const contribs = {};
    contribCols.forEach(({ h, i: colIdx }) => { contribs[h.slice("contrib_".length)] = +f[colIdx]; });
    let byQuantile = index.get(id);
    if (!byQuantile) { byQuantile = new Map(); index.set(id, byQuantile); }
    byQuantile.set(f[1], { base: +f[2], prediction: +f[3], contribs });
  }
  return index;
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

// ---------- Exception scanning (Gustavo's review checklist, run across all rows) ----------

function buildHolidayAdjacentSetsByStore() {
  const map = new Map(); // storeId -> Set(dates that are a holiday, or the day before one)
  for (const [storeId, store] of state.stores) {
    const holidayDates = new Set();
    state.events.forEach((ev) => {
      if (eventAppliesToStore(ev, store)) holidayDates.add(ev.date);
    });
    const adjacent = new Set(holidayDates);
    holidayDates.forEach((d) => adjacent.add(addDays(d, -1)));
    map.set(storeId, adjacent);
  }
  return map;
}

function buildSeriesStats(holidaySetsByStore) {
  const stats = new Map(); // "store|product" -> {nonPromoAvg, nonHolidayAvg} (P50, submission)
  for (const [key, points] of state.seriesIndex) {
    const storeId = key.split("|")[0];
    const holidaySet = holidaySetsByStore.get(storeId);
    let nonPromoSum = 0, nonPromoCount = 0, nonHolidaySum = 0, nonHolidayCount = 0;
    points.forEach((pt) => {
      const sub = state.submissionById.get(pt.id);
      if (!sub) return;
      if (pt.promotion !== "True") { nonPromoSum += sub.p50; nonPromoCount++; }
      if (!holidaySet.has(pt.date)) { nonHolidaySum += sub.p50; nonHolidayCount++; }
    });
    stats.set(key, {
      nonPromoAvg: nonPromoCount ? nonPromoSum / nonPromoCount : null,
      nonHolidayAvg: nonHolidayCount ? nonHolidaySum / nonHolidayCount : null,
    });
  }
  return stats;
}

function scanExceptions() {
  const flags = [];
  if (!state.submissionById) return flags;

  const holidaySetsByStore = buildHolidayAdjacentSetsByStore();
  const seriesStats = buildSeriesStats(holidaySetsByStore);

  for (const [id, sub] of state.submissionById) {
    const t = state.testById.get(id);
    if (!t) continue; // unexpected id — already surfaced by the validation strip
    const product = state.products.get(t.product_id);
    const store = state.stores.get(t.store_id);
    if (!product || !store) continue;

    function addFlag(type, detail) {
      flags.push({ id, storeId: t.store_id, productId: t.product_id, date: t.date, promotion: t.promotion, type, detail });
    }

    if (!(sub.p25 <= sub.p50 && sub.p50 <= sub.p75 && sub.p75 <= sub.p95)) {
      addFlag("monotonicity", `P25 ${sub.p25} / P50 ${sub.p50} / P75 ${sub.p75} / P95 ${sub.p95} out of order`);
    }

    if (product.is_perishable === "1") {
      const spiked = sub.p50 === 0 ? sub.p95 >= 3 : sub.p95 / sub.p50 >= 2;
      if (spiked) addFlag("perishable_spike", `P50 ${sub.p50} but P95 ${sub.p95}`);
    }

    const stats = seriesStats.get(t.store_id + "|" + t.product_id);
    if (t.promotion === "True" && stats && stats.nonPromoAvg !== null && stats.nonPromoAvg > 0.5 &&
        sub.p50 <= stats.nonPromoAvg * 1.05) {
      addFlag("promo_no_lift", `P50 ${sub.p50} vs. this series' typical non-promo P50 ${stats.nonPromoAvg.toFixed(1)}`);
    }

    const holidaySet = holidaySetsByStore.get(t.store_id);
    if (holidaySet.has(t.date) && stats && stats.nonHolidayAvg !== null && stats.nonHolidayAvg > 0.5) {
      const diffRatio = Math.abs(sub.p50 - stats.nonHolidayAvg) / stats.nonHolidayAvg;
      if (diffRatio < 0.1) addFlag("holiday_flat", `P50 ${sub.p50} ≈ this series' normal-day average ${stats.nonHolidayAvg.toFixed(1)}`);
    }

    if (sub.p50 === 0 && sub.p95 > 0) {
      if (STAPLE_FAMILIES.has(product.product_family)) {
        addFlag("essential_low", `Median forecast is 0 (P95 ${sub.p95}), family ${product.product_family}`);
      } else {
        addFlag("zero_forecast", `Median forecast is 0 (P95 ${sub.p95}), family ${product.product_family}`);
      }
    }

    if (sub.p95 - sub.p25 >= 2.5 * Math.max(sub.p50, 1)) {
      addFlag("wide_spread", `P25 ${sub.p25} to P95 ${sub.p95} — wide relative to P50 ${sub.p50}`);
    }

    const base = state.baselineById.get(id);
    if (base) {
      const rel = Math.abs(sub.p50 - base.p50) / Math.max(base.p50, 1);
      if (rel >= 1.0 && Math.abs(sub.p50 - base.p50) >= 3) {
        addFlag("baseline_deviation", `Submission P50 ${sub.p50} vs. baseline P50 ${base.p50} (${Math.round(rel * 100)}% relative difference)`);
      }
    }
  }

  return flags;
}

// ---------- Store/product pick lists & info panels ----------

let selectedStoreId = null;
let selectedProductId = null;
let selectedShapQuantile = "0.5";

function sortedIds(map) {
  return [...map.keys()].sort((a, b) => +a - +b);
}

function fillSelectOptions(selectEl, values, placeholder) {
  selectEl.textContent = "";
  const opt = document.createElement("option");
  opt.value = "";
  opt.textContent = placeholder;
  selectEl.appendChild(opt);
  values.forEach((v) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    selectEl.appendChild(o);
  });
}

function populateFilterOptions() {
  const cities = [...new Set([...state.stores.values()].map((s) => s.city))].sort();
  const clusters = [...new Set([...state.stores.values()].map((s) => s.store_cluster))].sort((a, b) => +a - +b);
  const formats = [...new Set([...state.stores.values()].map((s) => s.store_type))].sort();
  fillSelectOptions(document.getElementById("storeCityFilter"), cities, "All cities");
  fillSelectOptions(document.getElementById("storeClusterFilter"), clusters, "All clusters");
  fillSelectOptions(document.getElementById("storeFormatFilter"), formats, "All formats");

  const families = [...new Set([...state.products.values()].map((p) => p.product_family))].sort();
  const classes = [...new Set([...state.products.values()].map((p) => p.product_class))].sort((a, b) => +a - +b);
  fillSelectOptions(document.getElementById("productFamilyFilter"), families, "All families");
  fillSelectOptions(document.getElementById("productClassFilter"), classes, "All classes");

  ["storeSearch", "storeCityFilter", "storeClusterFilter", "storeFormatFilter"].forEach((id) =>
    document.getElementById(id).addEventListener("input", renderStoreList));
  ["productSearch", "productFamilyFilter", "productClassFilter", "productPerishableFilter"].forEach((id) =>
    document.getElementById(id).addEventListener("input", renderProductList));
}

function renderPickList(containerId, ids, selectedId, labelFn, onSelect) {
  const container = document.getElementById(containerId);
  container.textContent = "";
  if (ids.length === 0) {
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = "No matches.";
    container.appendChild(hint);
    return;
  }
  ids.forEach((id) => {
    const row = document.createElement("div");
    row.className = "pick-item" + (id === selectedId ? " selected" : "");
    const [main, sub] = labelFn(id);
    row.appendChild(document.createTextNode(main));
    const subEl = document.createElement("span");
    subEl.className = "pick-sub";
    subEl.textContent = sub;
    row.appendChild(subEl);
    row.addEventListener("click", () => onSelect(id));
    container.appendChild(row);
  });
}

function renderStoreList() {
  const q = document.getElementById("storeSearch").value.trim().toLowerCase();
  const city = document.getElementById("storeCityFilter").value;
  const cluster = document.getElementById("storeClusterFilter").value;
  const format = document.getElementById("storeFormatFilter").value;

  const ids = sortedIds(state.stores).filter((id) => {
    const s = state.stores.get(id);
    if (city && s.city !== city) return false;
    if (cluster && s.store_cluster !== cluster) return false;
    if (format && s.store_type !== format) return false;
    if (q && !(id.includes(q) || s.city.toLowerCase().includes(q))) return false;
    return true;
  });

  renderPickList("storeList", ids, selectedStoreId, (id) => {
    const s = state.stores.get(id);
    return [`${id} · ${s.city}`, `${s.department} · format ${s.store_type} · cluster ${s.store_cluster}`];
  }, selectStore);
}

function renderProductList() {
  const q = document.getElementById("productSearch").value.trim().toLowerCase();
  const family = document.getElementById("productFamilyFilter").value;
  const cls = document.getElementById("productClassFilter").value;
  const perishable = document.getElementById("productPerishableFilter").value;

  const ids = sortedIds(state.products).filter((id) => {
    const p = state.products.get(id);
    if (family && p.product_family !== family) return false;
    if (cls && p.product_class !== cls) return false;
    if (perishable && p.is_perishable !== perishable) return false;
    if (q && !(id.includes(q) || p.product_family.toLowerCase().includes(q))) return false;
    return true;
  });

  renderPickList("productList", ids, selectedProductId, (id) => {
    const p = state.products.get(id);
    return [`${id} · ${p.product_family}`, `class ${p.product_class}${p.is_perishable === "1" ? " · perishable" : ""}`];
  }, selectProduct);
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

function refreshSelection() {
  renderInfoPanels(selectedStoreId, selectedProductId);
  renderChart(selectedStoreId, selectedProductId);
  renderShapPanel(selectedStoreId, selectedProductId);
}

function selectStore(id) {
  selectedStoreId = id;
  renderStoreList();
  refreshSelection();
}

function selectProduct(id) {
  selectedProductId = id;
  renderProductList();
  refreshSelection();
}

function jumpToSeries(storeId, productId) {
  selectedStoreId = storeId;
  selectedProductId = productId;
  renderStoreList();
  renderProductList();
  refreshSelection();
  switchTab("forecasts");
  document.getElementById("chart").scrollIntoView({ behavior: "smooth", block: "center" });
}

// ---------- Exceptions table ----------

let lastExceptionFlags = [];
let exceptionFilterType = "all";
const MAX_EXCEPTION_ROWS = 300;

function renderExceptions() {
  const card = document.getElementById("exceptionsCard");
  if (!state.submissionById) { card.style.display = "none"; renderOverviewSummary(null); return; }
  card.style.display = "block";

  lastExceptionFlags = scanExceptions();

  const counts = {};
  const bySeverity = { critical: 0, warning: 0, info: 0 };
  lastExceptionFlags.forEach((f) => {
    counts[f.type] = (counts[f.type] || 0) + 1;
    bySeverity[FLAG_TYPES[f.type].severity]++;
  });

  document.getElementById("exceptionsSummary").textContent =
    `${lastExceptionFlags.length.toLocaleString()} flagged row(s) — ` +
    `${bySeverity.critical} critical, ${bySeverity.warning} warning, ${bySeverity.info} info`;

  renderOverviewSummary(bySeverity);

  const filterSelect = document.getElementById("exceptionFilter");
  filterSelect.textContent = "";
  const allOpt = document.createElement("option");
  allOpt.value = "all";
  allOpt.textContent = `All (${lastExceptionFlags.length})`;
  filterSelect.appendChild(allOpt);
  Object.keys(FLAG_TYPES).forEach((type) => {
    if (!counts[type]) return;
    const opt = document.createElement("option");
    opt.value = type;
    opt.textContent = `${FLAG_TYPES[type].label} (${counts[type]})`;
    filterSelect.appendChild(opt);
  });
  filterSelect.value = exceptionFilterType;
  filterSelect.onchange = () => {
    exceptionFilterType = filterSelect.value;
    renderExceptionTable();
  };

  ["exceptionPromoFilter", "exceptionFromDate", "exceptionToDate"].forEach((id) =>
    document.getElementById(id).onchange = renderExceptionTable);
  document.getElementById("exportExceptionsBtn").onclick = exportFilteredExceptions;

  renderExceptionTable();
}

function filteredExceptions() {
  const promo = document.getElementById("exceptionPromoFilter").value;
  const fromDate = document.getElementById("exceptionFromDate").value;
  const toDate = document.getElementById("exceptionToDate").value;

  return lastExceptionFlags
    .filter((f) => exceptionFilterType === "all" || f.type === exceptionFilterType)
    .filter((f) => !promo || f.promotion === promo)
    .filter((f) => !fromDate || f.date >= fromDate)
    .filter((f) => !toDate || f.date <= toDate)
    .sort((a, b) => SEVERITY_RANK[FLAG_TYPES[a.type].severity] - SEVERITY_RANK[FLAG_TYPES[b.type].severity] || a.date.localeCompare(b.date));
}

function renderExceptionTable() {
  const tbody = document.getElementById("exceptionTableBody");
  tbody.textContent = "";

  const filtered = filteredExceptions();
  const shown = filtered.slice(0, MAX_EXCEPTION_ROWS);
  shown.forEach((f) => {
    const sev = FLAG_TYPES[f.type].severity;
    const tr = document.createElement("tr");
    tr.className = "exception-row";

    const flagCell = document.createElement("td");
    const wrap = document.createElement("span");
    wrap.className = "status-item status-" + sev;
    const icon = document.createElement("span");
    icon.className = "status-icon";
    icon.textContent = sev === "critical" ? "✕" : sev === "warning" ? "⚠" : "ⓘ";
    const label = document.createElement("span");
    label.textContent = FLAG_TYPES[f.type].label;
    wrap.appendChild(icon);
    wrap.appendChild(label);
    flagCell.appendChild(wrap);

    const actionTextCell = document.createElement("td");
    actionTextCell.textContent = ACTION_BY_FLAG[f.type];

    const storeCell = document.createElement("td");
    storeCell.textContent = f.storeId;
    const productCell = document.createElement("td");
    productCell.textContent = f.productId;
    const dateCell = document.createElement("td");
    dateCell.textContent = f.date;
    const detailCell = document.createElement("td");
    detailCell.textContent = f.detail;

    const viewCell = document.createElement("td");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "link-button";
    btn.textContent = "View";
    btn.addEventListener("click", () => jumpToSeries(f.storeId, f.productId));
    viewCell.appendChild(btn);

    tr.append(flagCell, actionTextCell, storeCell, productCell, dateCell, detailCell, viewCell);
    tbody.appendChild(tr);
  });

  const hint = document.getElementById("exceptionMoreHint");
  hint.textContent = filtered.length > shown.length
    ? `Showing first ${shown.length.toLocaleString()} of ${filtered.length.toLocaleString()} — narrow the filter to see more.`
    : "";
}

// bySeverity is null before a submission file is loaded — hides the card.
function renderOverviewSummary(bySeverity) {
  const card = document.getElementById("overviewSummaryCard");
  if (!bySeverity) { card.style.display = "none"; return; }
  card.style.display = "block";
  const total = bySeverity.critical + bySeverity.warning + bySeverity.info;
  document.getElementById("overviewSummaryText").textContent = total === 0
    ? "No exceptions found — this submission looks clean."
    : `${total.toLocaleString()} row(s) flagged for review — ${bySeverity.critical} critical, ${bySeverity.warning} warning, ${bySeverity.info} info.`;
}

function exportFilteredExceptions() {
  const rows = filteredExceptions();
  const header = ["id", "store_id", "product_id", "date", "promotion", "flag", "severity", "action", "detail"];
  const csvLines = [header.join(",")];
  rows.forEach((f) => {
    const cells = [f.id, f.storeId, f.productId, f.date, f.promotion, f.type, FLAG_TYPES[f.type].severity, ACTION_BY_FLAG[f.type], f.detail];
    csvLines.push(cells.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","));
  });
  const blob = new Blob([csvLines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "exception_rows.csv";
  a.click();
  URL.revokeObjectURL(url);
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

function addDays(dateStr, n) {
  const [y, m, d] = dateStr.split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  dt.setUTCDate(dt.getUTCDate() + n);
  return dt.toISOString().slice(0, 10);
}

// Payday dates aren't in any data file — Gustavo/Sanne flagged the 3rd and
// 18th of each month as operationally visible in-store even though it's not
// a field in the extracts.
function isPayday(dateStr) {
  const day = +dateStr.slice(8, 10);
  return day === 3 || day === 18;
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

  const history = (state.historyIndex.get(storeId + "|" + productId) || [])
    .slice()
    .sort((a, b) => a.date.localeCompare(b.date));

  const rows = [
    ...history.map((h) => ({ date: h.date, promotion: h.promotion, actual: h.sales_count, sub: null, base: null })),
    ...points.map((pt) => ({
      ...pt,
      sub: state.submissionById ? state.submissionById.get(pt.id) : null,
      base: state.baselineById.get(pt.id),
      actual: null,
    })),
  ];
  const forecastStartIdx = history.length;

  let yMax = 0;
  rows.forEach((r) => {
    if (r.sub) yMax = Math.max(yMax, r.sub.p95);
    if (r.base) yMax = Math.max(yMax, r.base.p95);
    if (r.actual !== null) yMax = Math.max(yMax, r.actual);
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

  // x tick labels — spread out so a longer history+forecast range doesn't crowd
  const labelStep = Math.max(1, Math.ceil(rows.length / 10));
  rows.forEach((r, i) => {
    if (i % labelStep !== 0 && i !== rows.length - 1) return;
    const label = svgEl("text", { x: x(i), y: MARGIN.top + PLOT_H + 18, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11 });
    label.textContent = formatDate(r.date);
    svg.appendChild(label);
  });

  // payday ticks — routine, so a small mark on the axis rather than a full dashed line
  rows.forEach((r, i) => {
    if (!isPayday(r.date)) return;
    const gx = x(i), gy = MARGIN.top + PLOT_H;
    svg.appendChild(svgEl("line", { x1: gx, x2: gx, y1: gy, y2: gy + 5, stroke: "var(--text-muted)", "stroke-width": 1.5 }));
  });

  // forecast-start divider
  if (forecastStartIdx > 0 && forecastStartIdx < rows.length) {
    const gx = x(forecastStartIdx);
    svg.appendChild(svgEl("line", { x1: gx, x2: gx, y1: MARGIN.top, y2: MARGIN.top + PLOT_H, stroke: "var(--axis)", "stroke-width": 1, "stroke-dasharray": "1,3" }));
  }

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

  // recent actual sales — neutral ink, not a categorical hue (it's reference
  // context, not a series competing with submission/baseline)
  const withActual = rows.map((r, i) => ({ i, r })).filter(({ r }) => r.actual !== null);
  if (withActual.length > 0) {
    const actualPath = linePathFor(withActual, x, y, (r) => r.actual);
    svg.appendChild(svgEl("path", { d: actualPath, fill: "none", stroke: "var(--text-secondary)", "stroke-width": 1.5 }));
    withActual.forEach(({ i, r }) => {
      svg.appendChild(svgEl("circle", { cx: x(i), cy: y(r.actual), r: 2.5, fill: "var(--text-secondary)" }));
    });
  }

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
  dateRow.textContent = r.date +
    (r.promotion === "True" ? " · promotion" : "") +
    (holiday ? ` · ${holiday.event_type}` : "") +
    (isPayday(r.date) ? " · payday" : "");
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
  if (r.actual !== null && r.actual !== undefined) {
    addRow("var(--text-secondary)", "Actual sales", r.actual + (r.actual === 0 ? " (zero ≠ proven no demand)" : ""));
  }

  tooltip.style.left = (left + 12) + "px";
  tooltip.style.top = (top + 12) + "px";
  tooltip.style.display = "block";
}

function renderLegend() {
  renderLegendInto("legend", [
    { swatch: "line", color: "var(--series-1)", label: "Submission (P50, P25–P95 band)" },
    { swatch: "dashed", color: "var(--series-2)", label: "Baseline P50" },
    { swatch: "line", color: "var(--text-secondary)", label: `Actual sales, last ${HISTORY_LOOKBACK_DAYS}d` },
    { swatch: "dot", color: "var(--series-3)", label: "Promotion day" },
  ]);
}

function renderLegendInto(containerId, items) {
  const legend = document.getElementById(containerId);
  legend.textContent = "";
  items.forEach((it) => {
    const div = document.createElement("div");
    div.className = "legend-item";
    const svg = svgEl("svg", { width: 16, height: 10, class: "legend-swatch" });
    if (it.swatch === "dot") {
      svg.appendChild(svgEl("circle", { cx: 8, cy: 5, r: 4, fill: it.color }));
    } else if (it.swatch === "square") {
      svg.appendChild(svgEl("rect", { x: 2, y: 1, width: 12, height: 8, fill: it.color }));
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

// ---------- SHAP explanation panel ----------

function renderShapQuantileTabs() {
  const wrap = document.getElementById("shapQuantileTabs");
  wrap.textContent = "";
  const labels = { "0.25": "P25", "0.5": "P50", "0.75": "P75", "0.95": "P95" };
  SHAP_QUANTILES.forEach((q) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "quantile-tab" + (q === selectedShapQuantile ? " active" : "");
    btn.textContent = labels[q];
    btn.addEventListener("click", () => {
      selectedShapQuantile = q;
      renderShapPanel(selectedStoreId, selectedProductId);
    });
    wrap.appendChild(btn);
  });
}

const SHAP_MARGIN = { top: 20, right: 20, bottom: 28, left: 50 };
const SHAP_HEIGHT = 200;
const SHAP_PLOT_H = SHAP_HEIGHT - SHAP_MARGIN.top - SHAP_MARGIN.bottom;

function renderShapPanel(storeId, productId) {
  const card = document.getElementById("shapCard");
  if (state.shapIndex.size === 0) { card.style.display = "none"; return; }
  card.style.display = "block";
  renderShapQuantileTabs();

  const svg = document.getElementById("shapChart");
  svg.textContent = "";

  const points = state.seriesIndex.get(storeId + "|" + productId) || [];
  const rows = points.map((pt) => ({
    date: pt.date,
    shap: (state.shapIndex.get(pt.id) || new Map()).get(selectedShapQuantile) || null,
  }));

  if (rows.every((r) => !r.shap)) {
    const empty = svgEl("text", { x: WIDTH / 2, y: SHAP_HEIGHT / 2, "text-anchor": "middle", fill: "var(--text-muted)" });
    empty.textContent = "No SHAP rows for this store × product in the loaded file.";
    svg.appendChild(empty);
    return;
  }

  const plotW = WIDTH - SHAP_MARGIN.left - SHAP_MARGIN.right;
  const x = (i) => SHAP_MARGIN.left + (rows.length === 1 ? 0 : (i / (rows.length - 1)) * plotW);
  const barWidth = Math.min(20, plotW / rows.length - 2);

  // stacked, signed totals per row — feeds both the y-scale and the bars
  const stacks = rows.map((r) => {
    if (!r.shap) return null;
    let pos = 0, neg = 0;
    const segments = [];
    SHAP_BUCKETS.forEach(({ key }) => {
      const v = r.shap.contribs[key] || 0;
      if (v > 0) { segments.push({ key, from: pos, to: pos + v, v }); pos += v; }
      else if (v < 0) { segments.push({ key, from: neg, to: neg + v, v }); neg += v; }
    });
    return { segments, pos, neg };
  });

  let magnitude = 0;
  stacks.forEach((s) => { if (s) magnitude = Math.max(magnitude, s.pos, -s.neg); });
  const yMax = niceMax(magnitude * 1.15);
  const centerY = SHAP_MARGIN.top + SHAP_PLOT_H / 2;
  const y = (v) => centerY - (v / yMax) * (SHAP_PLOT_H / 2);

  // zero baseline + magnitude reference ticks
  svg.appendChild(svgEl("line", { x1: SHAP_MARGIN.left, x2: WIDTH - SHAP_MARGIN.right, y1: centerY, y2: centerY, stroke: "var(--axis)", "stroke-width": 1 }));
  [yMax, -yMax].forEach((v) => {
    const label = svgEl("text", { x: SHAP_MARGIN.left - 8, y: y(v) + 4, "text-anchor": "end", fill: "var(--text-muted)", "font-size": 11 });
    label.textContent = (v > 0 ? "+" : "") + Math.round(v);
    svg.appendChild(label);
  });

  // x tick labels
  const labelStep = Math.max(1, Math.ceil(rows.length / 10));
  rows.forEach((r, i) => {
    if (i % labelStep !== 0 && i !== rows.length - 1) return;
    const label = svgEl("text", { x: x(i), y: SHAP_HEIGHT - 8, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 11 });
    label.textContent = formatDate(r.date);
    svg.appendChild(label);
  });

  // bars
  stacks.forEach((s, i) => {
    if (!s) return;
    const barX = x(i) - barWidth / 2;
    s.segments.forEach((seg) => {
      const y1 = y(seg.from), y2 = y(seg.to);
      svg.appendChild(svgEl("rect", {
        x: barX, y: Math.min(y1, y2), width: barWidth, height: Math.max(1, Math.abs(y2 - y1)),
        fill: seg.v > 0 ? "var(--shap-pos)" : "var(--shap-neg)",
      }));
    });
  });

  // hover hit area + tooltip
  const hit = svgEl("rect", { x: SHAP_MARGIN.left, y: SHAP_MARGIN.top, width: plotW, height: SHAP_PLOT_H, fill: "transparent" });
  svg.appendChild(hit);
  const tooltip = document.getElementById("shapTooltip");
  hit.addEventListener("pointermove", (e) => {
    const rect = svg.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * WIDTH;
    let idx = Math.round(((mx - SHAP_MARGIN.left) / plotW) * (rows.length - 1));
    idx = Math.max(0, Math.min(rows.length - 1, idx));
    showShapTooltip(tooltip, rows[idx], e.clientX - rect.left, e.clientY - rect.top);
  });
  hit.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });

  renderLegendInto("shapLegend", [
    { swatch: "square", color: "var(--shap-pos)", label: "Pushes forecast up" },
    { swatch: "square", color: "var(--shap-neg)", label: "Pushes forecast down" },
  ]);
}

function showShapTooltip(tooltip, row, left, top) {
  tooltip.textContent = "";
  const dateRow = document.createElement("div");
  dateRow.className = "tt-date";
  dateRow.textContent = row.date;
  tooltip.appendChild(dateRow);

  if (!row.shap) {
    const none = document.createElement("div");
    none.className = "tt-row";
    none.textContent = "No SHAP row for this date.";
    tooltip.appendChild(none);
    tooltip.style.left = (left + 12) + "px";
    tooltip.style.top = (top + 12) + "px";
    tooltip.style.display = "block";
    return;
  }

  function addRow(color, label, value) {
    const rowEl = document.createElement("div");
    rowEl.className = "tt-row";
    const leftEl = document.createElement("span");
    leftEl.className = "tt-label";
    if (color) {
      const key = document.createElement("span");
      key.className = "tt-key";
      key.style.background = color;
      leftEl.appendChild(key);
    }
    leftEl.appendChild(document.createTextNode(label));
    const rightEl = document.createElement("span");
    rightEl.className = "tt-value";
    rightEl.textContent = value;
    rowEl.appendChild(leftEl);
    rowEl.appendChild(rightEl);
    tooltip.appendChild(rowEl);
  }

  addRow(null, "Prediction", Math.round(row.shap.prediction));
  addRow(null, "Baseline (no features)", Math.round(row.shap.base));
  SHAP_BUCKETS
    .map((b) => ({ ...b, v: row.shap.contribs[b.key] || 0 }))
    .filter((b) => Math.abs(b.v) >= 0.05)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .forEach((b) => addRow(b.v > 0 ? "var(--shap-pos)" : "var(--shap-neg)", b.label, (b.v > 0 ? "+" : "") + b.v.toFixed(1)));

  tooltip.style.left = (left + 12) + "px";
  tooltip.style.top = (top + 12) + "px";
  tooltip.style.display = "block";
}

// ---------- Tab navigation ----------

const TABS = ["overview", "exceptions", "forecasts"];

function switchTab(tab) {
  TABS.forEach((t) => {
    document.getElementById("tab-" + t).hidden = t !== tab;
  });
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
}

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  document.getElementById("goToExceptionsBtn").addEventListener("click", () => switchTab("exceptions"));
  switchTab("overview");
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
      renderExceptions();
      status.textContent = "Loaded " + file.name;
      if (selectedStoreId && selectedProductId) renderChart(selectedStoreId, selectedProductId);
    };
    reader.readAsText(file);
  });

  document.getElementById("shapInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const status = document.getElementById("shapLoadStatus");
    status.textContent = "Reading " + file.name + "…";
    const reader = new FileReader();
    reader.onload = () => {
      state.shapIndex = parseShapCsv(reader.result);
      status.textContent = `Loaded ${file.name} (${state.shapIndex.size.toLocaleString()} ids)`;
      if (selectedStoreId && selectedProductId) renderShapPanel(selectedStoreId, selectedProductId);
    };
    reader.readAsText(file);
  });

  (async function init() {
    initTabs();
    const status = document.getElementById("loadStatus");
    status.textContent = "Loading context data…";
    await loadContext();
    status.textContent = "";
    populateFilterOptions();
    selectedStoreId = sortedIds(state.stores)[0];
    selectedProductId = sortedIds(state.products)[0];
    renderStoreList();
    renderProductList();
    renderInfoPanels(selectedStoreId, selectedProductId);
    renderChart(selectedStoreId, selectedProductId);
  })();
}

// Exposed for the self-check in test_app.js — no effect in the browser.
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    splitCSVLine, eventAppliesToStore, niceMax, formatDate, addDays, isPayday,
    validateAndIndexSubmission, scanExceptions, FLAG_TYPES, ACTION_BY_FLAG,
    parseShapCsv, SHAP_BUCKETS, state,
  };
}
