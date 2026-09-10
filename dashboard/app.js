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

// Severity now drives a single reserved alert colour, not a 3-hue traffic
// light (Gustavo: "one strong alert colour for genuinely important cases —
// not a children's birthday party"). "critical" = the three cases he named
// explicitly (wide uncertainty, sharp changes, weak promoted performance)
// plus data-integrity errors; everything else is a quieter, neutral note.
const FLAG_TYPES = {
  monotonicity: { label: "Numbers don't add up", severity: "critical" },
  wide_spread: { label: "Unusually wide uncertainty", severity: "critical" },
  baseline_deviation: { label: "Sharp change from the usual model", severity: "critical" },
  promo_no_lift: { label: "Weak promoted performance", severity: "critical" },
  perishable_spike: { label: "Perishable item, big swing possible", severity: "warning" },
  holiday_flat: { label: "Holiday treated like a normal day", severity: "warning" },
  essential_low: { label: "Everyday item forecast at zero", severity: "warning" },
  promo_zero_history: { label: "Promoted day sold zero, recently", severity: "warning" },
  zero_forecast: { label: "Forecast is zero", severity: "info" },
};
const SEVERITY_RANK = { critical: 0, warning: 1, info: 2 };
const SEVERITY_LABEL = { critical: "Alert", warning: "Review", info: "Note" };

// What a reviewer should actually do about each flag — Gustavo's "practical
// action view" ask, folded into the exceptions table rather than a second page.
const ACTION_BY_FLAG = {
  monotonicity: "Data problem — fix or reject before using this row",
  wide_spread: "Uncertain forecast — use your judgement, not just the number",
  baseline_deviation: "Double-check this — it's very different from the usual model",
  promo_no_lift: "Check this promotion — no sales boost is expected",
  perishable_spike: "Perishable item — double-check you won't overstock",
  holiday_flat: "Check holiday staffing & stock — forecast doesn't reflect the holiday",
  essential_low: "Everyday item — confirm zero is right before it runs out",
  promo_zero_history: "Not a confirmed stockout — flagged for review, check before assuming no demand",
  zero_forecast: "Double-check: is zero really expected here?",
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
  trafficIndex: new Map(),  // store_id -> Map(date -> transaction_count)
  submissionById: null,     // id -> {p25,p50,p75,p95}, set once a file is loaded
  shapIndex: new Map(),     // id -> Map("0.25"|"0.5"|"0.75"|"0.95" -> {base, prediction, contribs: {bucket: value}})
  storeRollups: new Map(),  // store_id -> {totalP50, totalP95, exceptions, critical}, built by computeRollups()
  productRollups: new Map(),// "estate"|store_id -> Map(product_id -> {sumP50, minP25, maxP95, anyPromo, exceptions})
  focusedStoreId: null,     // drill-down: Stores tab -> Products tab scope; null = estate-wide
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
  const [storeTxt, itemTxt, eventsTxt, trafficTxt] = await Promise.all([
    loadText("../gustavo_context/store_info.csv"),
    loadText("../gustavo_context/item_catalogue.csv"),
    loadText("../gustavo_context/events_and_holidays.csv"),
    loadText("../gustavo_context/store_traffic.csv"),
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
  forEachRow(trafficTxt, (f) => {
    const [date, storeId, transactionCount] = f;
    let byDate = state.trafficIndex.get(storeId);
    if (!byDate) { byDate = new Map(); state.trafficIndex.set(storeId, byDate); }
    byDate.set(date, +transactionCount);
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

const WIDE_SPREAD_PERCENTILE = 0.95;

// Real quantile forecasts on sparse grocery demand are wide by default —
// the median P25-P95 spread relative to P50 in a real submission run was
// already ~3x, so a fixed multiplier flagged 62% of all rows as "critical."
// "Unusually wide" has to mean unusual *for this submission*, so the
// threshold is the submission's own top-tail (95th percentile), not a
// magic constant tuned to one dataset.
function wideSpreadThreshold() {
  const ratios = [];
  for (const sub of state.submissionById.values()) {
    ratios.push((sub.p95 - sub.p25) / Math.max(sub.p50, 1));
  }
  ratios.sort((a, b) => a - b);
  const idx = Math.min(ratios.length - 1, Math.floor(ratios.length * WIDE_SPREAD_PERCENTILE));
  return ratios[idx];
}

function scanExceptions() {
  const flags = [];
  if (!state.submissionById) return flags;

  const holidaySetsByStore = buildHolidayAdjacentSetsByStore();
  const seriesStats = buildSeriesStats(holidaySetsByStore);
  const wideSpreadCutoff = wideSpreadThreshold();

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
      addFlag("monotonicity", `Low estimate ${sub.p25}, typical ${sub.p50}, high estimate ${sub.p75}, worst case ${sub.p95} — should rise in that order, but doesn't`);
    }

    if (product.is_perishable === "1") {
      const spiked = sub.p50 === 0 ? sub.p95 >= 3 : sub.p95 / sub.p50 >= 2;
      if (spiked) addFlag("perishable_spike", `Typical estimate is ${sub.p50}, but the worst case is ${sub.p95} — much higher`);
    }

    const stats = seriesStats.get(t.store_id + "|" + t.product_id);
    if (t.promotion === "True" && stats && stats.nonPromoAvg !== null && stats.nonPromoAvg > 0.5 &&
        sub.p50 <= stats.nonPromoAvg * 1.05) {
      addFlag("promo_no_lift", `Forecast is ${sub.p50}, about the same as a normal day (~${stats.nonPromoAvg.toFixed(1)}) despite the promotion`);
    }

    const holidaySet = holidaySetsByStore.get(t.store_id);
    if (holidaySet.has(t.date) && stats && stats.nonHolidayAvg !== null && stats.nonHolidayAvg > 0.5) {
      const diffRatio = Math.abs(sub.p50 - stats.nonHolidayAvg) / stats.nonHolidayAvg;
      if (diffRatio < 0.1) addFlag("holiday_flat", `Forecast is ${sub.p50}, about the same as a normal day (~${stats.nonHolidayAvg.toFixed(1)}) even though this is a holiday`);
    }

    if (sub.p50 === 0 && sub.p95 > 0) {
      if (STAPLE_FAMILIES.has(product.product_family)) {
        addFlag("essential_low", `Typical forecast is 0 units (worst case up to ${sub.p95}) for an everyday ${product.product_family.toLowerCase()} item`);
      } else {
        addFlag("zero_forecast", `Typical forecast is 0 units (worst case up to ${sub.p95})`);
      }
    }

    if ((sub.p95 - sub.p25) / Math.max(sub.p50, 1) >= wideSpreadCutoff) {
      addFlag("wide_spread", `Could be anywhere from ${sub.p25} to ${sub.p95} around a typical estimate of ${sub.p50} — unusually wide even for this submission`);
    }

    const base = state.baselineById.get(id);
    if (base) {
      const rel = Math.abs(sub.p50 - base.p50) / Math.max(base.p50, 1);
      if (rel >= 1.0 && Math.abs(sub.p50 - base.p50) >= 3) {
        addFlag("baseline_deviation", `Forecast is ${sub.p50}, vs. ${base.p50} from the simple baseline model — a ${Math.round(rel * 100)}% difference`);
      }
    }
  }

  // Promoted day, zero units actually sold — in the recent history window
  // (last HISTORY_LOOKBACK_DAYS, same window the chart overlays). Not
  // treated as a stockout, just surfaced for a human to check.
  for (const [key, points] of state.seriesIndex) {
    const [storeId, productId] = key.split("|");
    const history = state.historyIndex.get(key);
    if (!history) continue;
    history.forEach((h) => {
      if (h.promotion === "True" && h.sales_count === 0) {
        flags.push({
          id: null, storeId, productId, date: h.date, promotion: "True",
          type: "promo_zero_history",
          detail: `Promoted on ${h.date}, but 0 units sold — worth checking before assuming no demand`,
        });
      }
    });
  }

  return flags;
}

// ---------- Store/product rollups (control-room views) ----------

// One pass over the submission + exception flags to build the estate-level
// numbers the Stores and Products tabs need: total forecast exposure and
// exception counts, rolled up by store and by store×product.
function computeRollups(flags) {
  const storeRollups = new Map();   // store_id -> {totalP50, totalP95, exceptions, critical}
  const byStoreProduct = new Map(); // "store|product" -> {sumP50, minP25, maxP95, anyPromo, exceptions, critical}

  function storeRow(storeId) {
    let r = storeRollups.get(storeId);
    if (!r) { r = { totalP50: 0, totalP95: 0, exceptions: 0, critical: 0 }; storeRollups.set(storeId, r); }
    return r;
  }
  function spRow(key) {
    let r = byStoreProduct.get(key);
    if (!r) { r = { sumP50: 0, minP25: Infinity, maxP95: -Infinity, anyPromo: false, exceptions: 0, critical: 0 }; byStoreProduct.set(key, r); }
    return r;
  }

  for (const [id, sub] of state.submissionById) {
    const t = state.testById.get(id);
    if (!t) continue;
    const key = t.store_id + "|" + t.product_id;
    const s = storeRow(t.store_id);
    s.totalP50 += sub.p50;
    s.totalP95 += sub.p95;
    const sp = spRow(key);
    sp.sumP50 += sub.p50;
    sp.minP25 = Math.min(sp.minP25, sub.p25);
    sp.maxP95 = Math.max(sp.maxP95, sub.p95);
    if (t.promotion === "True") sp.anyPromo = true;
  }

  flags.forEach((f) => {
    const sev = FLAG_TYPES[f.type].severity;
    storeRow(f.storeId).exceptions++;
    if (sev === "critical") storeRow(f.storeId).critical++;
    const sp = spRow(f.storeId + "|" + f.productId);
    sp.exceptions++;
    if (sev === "critical") sp.critical++;
  });

  // roll store×product up to an estate-wide per-product view (summed
  // across all stores), for the Products tab when no store is focused
  const productRollups = new Map([["estate", new Map()]]);
  for (const [key, sp] of byStoreProduct) {
    const [storeId, productId] = key.split("|");
    let perStore = productRollups.get(storeId);
    if (!perStore) { perStore = new Map(); productRollups.set(storeId, perStore); }
    perStore.set(productId, sp);

    const estate = productRollups.get("estate");
    let e = estate.get(productId);
    if (!e) { e = { sumP50: 0, minP25: Infinity, maxP95: -Infinity, anyPromo: false, exceptions: 0, critical: 0 }; estate.set(productId, e); }
    e.sumP50 += sp.sumP50;
    e.minP25 = Math.min(e.minP25, sp.minP25);
    e.maxP95 = Math.max(e.maxP95, sp.maxP95);
    e.anyPromo = e.anyPromo || sp.anyPromo;
    e.exceptions += sp.exceptions;
    e.critical += sp.critical;
  }

  state.storeRollups = storeRollups;
  state.productRollups = productRollups;
}

function topRiskProducts(storeId) {
  const perStore = state.productRollups.get(storeId);
  if (!perStore) return [];
  return [...perStore.entries()]
    .filter(([, r]) => r.exceptions > 0)
    .sort((a, b) => b[1].critical - a[1].critical || b[1].exceptions - a[1].exceptions)
    .slice(0, 3)
    .map(([pid]) => {
      const p = state.products.get(pid);
      return p ? `${p.product_family} (${pid})` : pid;
    });
}

function focusStore(storeId) {
  state.focusedStoreId = storeId;
  renderProductsTab();
  switchTab("products");
}

let storeSortBy = "risk";

// Estate summary: total expected/exposure units and alert counts per store —
// the "where do I need to act" answer at a glance. Sorted by risk (alerts)
// by default so the stores needing attention are on top, not alphabetical.
function renderStoresTab() {
  const tbody = document.getElementById("storeRollupBody");
  if (!tbody) return;
  tbody.textContent = "";

  const rows = sortedIds(state.stores)
    .filter((id) => state.storeRollups.has(id))
    .map((id) => ({ id, s: state.stores.get(id), r: state.storeRollups.get(id) }));

  rows.sort((a, b) => storeSortBy === "risk"
    ? (b.r.critical - a.r.critical) || (b.r.exceptions - a.r.exceptions)
    : (b.r.totalP95 - a.r.totalP95));

  rows.forEach(({ id, s, r }) => {
    const tr = document.createElement("tr");
    tr.className = "exception-row";

    const cells = [id, s.city, s.store_type, s.store_cluster,
      Math.round(r.totalP50).toLocaleString(), Math.round(r.totalP95).toLocaleString()]
      .map((v) => { const td = document.createElement("td"); td.textContent = v; return td; });

    const alertCell = document.createElement("td");
    if (r.critical > 0) {
      const chip = document.createElement("span");
      chip.className = "status-item status-critical";
      const icon = document.createElement("span");
      icon.className = "status-icon";
      icon.textContent = "✕";
      chip.append(icon, document.createTextNode(`${r.critical} alert${r.critical === 1 ? "" : "s"}`));
      alertCell.appendChild(chip);
      if (r.exceptions > r.critical) alertCell.appendChild(document.createTextNode(` · ${r.exceptions - r.critical} more`));
    } else {
      alertCell.textContent = r.exceptions > 0 ? `${r.exceptions} to review` : "Clean";
      alertCell.className = "muted";
    }

    const topCell = document.createElement("td");
    topCell.textContent = topRiskProducts(id).join(", ") || "—";

    const viewCell = document.createElement("td");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "link-button";
    btn.textContent = "Drill in";
    btn.addEventListener("click", () => focusStore(id));
    viewCell.appendChild(btn);

    tr.append(...cells, alertCell, topCell, viewCell);
    tbody.appendChild(tr);
  });
}

// Product view: family, class, perishability, promotion status, and forecast
// range — scoped to whichever store was drilled into from the Stores tab,
// or estate-wide if none. Fresh (perishable) items sort first, since
// shortage and waste both hurt there.
function renderProductsTab() {
  const tbody = document.getElementById("productRollupBody");
  if (!tbody) return;
  tbody.textContent = "";

  const hint = document.getElementById("productsScopeHint");
  if (state.focusedStoreId) {
    const s = state.stores.get(state.focusedStoreId);
    hint.textContent = "";
    hint.innerHTML = `Showing <strong>${s.city} (store ${state.focusedStoreId})</strong> — format ${s.store_type}, cluster ${s.store_cluster}. `;
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "link-button";
    clearBtn.textContent = "Show all stores";
    clearBtn.addEventListener("click", () => { state.focusedStoreId = null; renderProductsTab(); });
    hint.appendChild(clearBtn);
  } else {
    hint.textContent = "Showing all stores (estate-wide) — drill in from the Stores tab to focus on one store, to compare like with like.";
  }

  const scope = state.focusedStoreId || "estate";
  const rollup = state.productRollups.get(scope) || new Map();
  const rows = [...rollup.entries()]
    .map(([pid, r]) => ({ pid, p: state.products.get(pid), r }))
    .filter((row) => row.p);

  rows.sort((a, b) =>
    (b.p.is_perishable === "1") - (a.p.is_perishable === "1") ||
    b.r.critical - a.r.critical ||
    b.r.exceptions - a.r.exceptions);

  rows.forEach(({ pid, p, r }) => {
    const tr = document.createElement("tr");
    tr.className = "exception-row";
    if (p.is_perishable === "1") tr.classList.add("fresh-row");

    const cells = [pid, p.product_family, p.product_class,
      p.is_perishable === "1" ? "Fresh" : "—",
      r.anyPromo ? "Promoted" : "—",
      `${Math.round(r.minP25)}–${Math.round(r.maxP95)} (Σ P50 ${Math.round(r.sumP50).toLocaleString()})`]
      .map((v) => { const td = document.createElement("td"); td.textContent = v; return td; });

    const alertCell = document.createElement("td");
    alertCell.textContent = r.critical > 0 ? `${r.critical} alert${r.critical === 1 ? "" : "s"}`
      : r.exceptions > 0 ? `${r.exceptions} to review` : "Clean";
    alertCell.className = r.critical > 0 ? "alert-text" : "muted";

    const viewCell = document.createElement("td");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "link-button";
    btn.textContent = "View forecast";
    btn.addEventListener("click", () => {
      const storeId = state.focusedStoreId ||
        sortedIds(state.stores).find((sid) => (state.productRollups.get(sid) || new Map()).has(pid)) ||
        sortedIds(state.stores)[0];
      jumpToSeries(storeId, pid);
    });
    viewCell.appendChild(btn);

    tr.append(...cells, alertCell, viewCell);
    tbody.appendChild(tr);
  });
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
  computeRollups(lastExceptionFlags);

  const counts = {};
  const bySeverity = { critical: 0, warning: 0, info: 0 };
  lastExceptionFlags.forEach((f) => {
    counts[f.type] = (counts[f.type] || 0) + 1;
    bySeverity[FLAG_TYPES[f.type].severity]++;
  });

  document.getElementById("exceptionsSummary").textContent = lastExceptionFlags.length === 0
    ? "No exceptions found — this submission looks clean."
    : `${lastExceptionFlags.length.toLocaleString()} row(s) flagged — ` +
      `${bySeverity.critical} alert, ${bySeverity.warning} review, ${bySeverity.info} note`;

  renderOverviewSummary(bySeverity);
  renderStoresTab();
  renderProductsTab();

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

    // "What to do" first (the actionable line), the flag type as a small
    // status chip underneath (why it was picked up) — leads with the action
    // a non-technical reviewer needs, not the internal rule name.
    const actionCell = document.createElement("td");
    const actionText = document.createElement("div");
    actionText.textContent = ACTION_BY_FLAG[f.type];
    const chip = document.createElement("span");
    chip.className = "status-item status-" + sev;
    const icon = document.createElement("span");
    icon.className = "status-icon";
    icon.textContent = sev === "critical" ? "✕" : sev === "warning" ? "⚠" : "ⓘ";
    const chipLabel = document.createElement("span");
    chipLabel.textContent = `${SEVERITY_LABEL[sev]} — ${FLAG_TYPES[f.type].label}`;
    chip.appendChild(icon);
    chip.appendChild(chipLabel);
    actionCell.appendChild(actionText);
    actionCell.appendChild(chip);

    const store = state.stores.get(f.storeId);
    const product = state.products.get(f.productId);
    const storeCell = document.createElement("td");
    storeCell.textContent = store ? `${store.city} (${f.storeId})` : f.storeId;
    const productCell = document.createElement("td");
    productCell.textContent = product ? `${product.product_family} (${f.productId})` : f.productId;
    const dateCell = document.createElement("td");
    dateCell.textContent = f.date;
    const detailCell = document.createElement("td");
    detailCell.textContent = f.detail;

    const viewCell = document.createElement("td");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "link-button";
    btn.textContent = "View chart";
    btn.addEventListener("click", () => jumpToSeries(f.storeId, f.productId));
    viewCell.appendChild(btn);

    tr.append(actionCell, storeCell, productCell, dateCell, detailCell, viewCell);
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

function eventScopeMatches(event, store) {
  if (event.scope === "National") return true;
  if (event.scope === "Regional") return event.location === store.department;
  if (event.scope === "Local") return event.location === store.city;
  return false;
}

// Whether an event is a live holiday for demand purposes at this store — a
// transferred holiday isn't (its date moved elsewhere), even though it's
// still shown on the chart for calendar context via eventScopeMatches.
function eventAppliesToStore(event, store) {
  if (event.is_transferred === "True") return false;
  return eventScopeMatches(event, store);
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
    renderForecastTable([], storeId);
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

  // calendar context — holidays, local/regional events, and transferred
  // holidays are all shown (scope + location included, so a town event is
  // never mistaken for a national one). Transferred holidays are shown for
  // context only — eventAppliesToStore excludes them from the "is this
  // actually a holiday here" logic used by holiday_flat / the shading below.
  const store = state.stores.get(storeId);
  const rangeStart = rows[0].date, rangeEnd = rows[rows.length - 1].date;
  const calendarByDate = new Map();
  state.events.forEach((ev) => {
    if (ev.date < rangeStart || ev.date > rangeEnd) return;
    if (!eventScopeMatches(ev, store)) return;
    calendarByDate.set(ev.date, ev);
  });

  // pre-holiday shading — the day leading into a real (non-transferred)
  // holiday, so the "surrounding period" reads as calendar context, not
  // just the day itself.
  rows.forEach((r, i) => {
    if (i === 0) return;
    const ev = calendarByDate.get(r.date);
    if (!ev || ev.is_transferred === "True") return;
    const bandLeft = x(i - 1), bandRight = x(i);
    svg.appendChild(svgEl("rect", {
      x: bandLeft, y: MARGIN.top, width: Math.max(1, bandRight - bandLeft), height: PLOT_H,
      fill: "var(--text-muted)", "fill-opacity": 0.08,
    }));
  });

  rows.forEach((r, i) => {
    const ev = calendarByDate.get(r.date);
    if (!ev) return;
    const gx = x(i);
    const transferred = ev.is_transferred === "True";
    svg.appendChild(svgEl("line", {
      x1: gx, x2: gx, y1: MARGIN.top, y2: MARGIN.top + PLOT_H,
      stroke: "var(--text-muted)", "stroke-width": 1,
      "stroke-dasharray": transferred ? "1,2" : "3,3", "stroke-opacity": transferred ? 0.6 : 1,
    }));
    const scopeText = ev.scope === "National" ? "National" : `${ev.scope} · ${ev.location}`;
    const label = svgEl("text", { x: gx, y: MARGIN.top - 8, "text-anchor": "middle", fill: "var(--text-muted)", "font-size": 9 });
    label.textContent = `${ev.event_type}${transferred ? " (moved)" : ""} · ${scopeText}`;
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
    showTooltip(tooltip, r, calendarByDate.get(r.date), storeId, e.clientX - rect.left, e.clientY - rect.top);
  });
  hit.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    tooltip.style.display = "none";
  });

  renderLegend();
  renderForecastTable(rows, storeId);
}

// Every store×product×date row, P25/P50/P75/P95 always shown in that order
// — Gustavo: "do not hide the range behind one forecast number." Promoted
// rows are tinted so they're easy to spot against ordinary days.
function renderForecastTable(rows, storeId) {
  const tbody = document.getElementById("forecastRowBody");
  if (!tbody) return;
  tbody.textContent = "";
  const trafficByDate = state.trafficIndex.get(storeId) || new Map();

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = "exception-row" + (r.promotion === "True" ? " promo-row" : "");

    const dateCell = document.createElement("td");
    dateCell.textContent = r.date;
    const promoCell = document.createElement("td");
    promoCell.textContent = r.promotion === "True" ? "PROMOTION" : "—";
    const actualCell = document.createElement("td");
    actualCell.textContent = (r.actual !== null && r.actual !== undefined) ? r.actual : "—";
    const trafficCell = document.createElement("td");
    const traffic = trafficByDate.get(r.date);
    trafficCell.textContent = traffic !== undefined ? traffic.toLocaleString() : "—";

    const quantileCells = ["p25", "p50", "p75", "p95"].map((k) => {
      const td = document.createElement("td");
      td.textContent = r.sub ? Math.round(r.sub[k]) : "—";
      return td;
    });

    tr.append(dateCell, promoCell, actualCell, trafficCell, ...quantileCells);
    tbody.appendChild(tr);
  });
}

function bandPath(items, x, y, lowFn, highFn) {
  const forward = items.map(({ i, r }) => `${x(i)},${y(lowFn(r))}`);
  const backward = items.slice().reverse().map(({ i, r }) => `${x(i)},${y(highFn(r))}`);
  return `M ${forward.join(" L ")} L ${backward.join(" L ")} Z`;
}

function linePathFor(items, x, y, valFn) {
  return items.map(({ i, r }, idx) => `${idx === 0 ? "M" : "L"} ${x(i)},${y(valFn(r))}`).join(" ");
}

function showTooltip(tooltip, r, calendarEvent, storeId, left, top) {
  tooltip.textContent = "";
  const dateRow = document.createElement("div");
  dateRow.className = "tt-date";
  dateRow.textContent = r.date +
    (r.promotion === "True" ? " · PROMOTION" : "") +
    (isPayday(r.date) ? " · payday" : "");
  tooltip.appendChild(dateRow);

  if (calendarEvent) {
    const scopeText = calendarEvent.scope === "National" ? "National" : `${calendarEvent.scope} · ${calendarEvent.location}`;
    const evRow = document.createElement("div");
    evRow.className = "tt-row";
    evRow.textContent = `${calendarEvent.event_type}${calendarEvent.is_transferred === "True" ? " (moved — not an active holiday here)" : ""} — ${scopeText}`;
    tooltip.appendChild(evRow);
  }

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
    addRow("var(--series-1)", "P25", Math.round(r.sub.p25));
    addRow(null, "P50", Math.round(r.sub.p50));
    addRow(null, "P75", Math.round(r.sub.p75));
    addRow(null, "P95", Math.round(r.sub.p95));
  }
  if (r.base) {
    addRow("var(--series-2)", "Baseline P50", Math.round(r.base.p50));
  }
  if (r.actual !== null && r.actual !== undefined) {
    addRow("var(--text-secondary)", "Actual sales", r.actual + (r.actual === 0 ? " (zero ≠ proven no demand)" : ""));
  }
  const traffic = (state.trafficIndex.get(storeId) || new Map()).get(r.date);
  if (traffic !== undefined) {
    addRow(null, "Store transactions that day", traffic.toLocaleString());
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
  const labels = { "0.25": "Low estimate", "0.5": "Typical", "0.75": "High estimate", "0.95": "Worst case" };
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
    renderShapDetail(null);
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
    renderShapDetail(rows[idx]);
  });
  hit.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });

  renderLegendInto("shapLegend", [
    { swatch: "square", color: "var(--shap-pos)", label: "Pushes forecast up" },
    { swatch: "square", color: "var(--shap-neg)", label: "Pushes forecast down" },
  ]);

  // default detail panel — most reviewers won't think to hover a chart, so
  // show the first day's breakdown up front rather than an empty panel.
  renderShapDetail(rows.find((r) => r.shap) || null);
}

// Shared with the tooltip: plain-language "what's driving this day's number"
// used by the always-visible panel below the chart, updated on hover.
function renderShapDetail(row) {
  const el = document.getElementById("shapDetail");
  if (!row || !row.shap) {
    el.innerHTML = '<span class="muted">Point at a bar in the chart above to see what\'s driving that day\'s forecast.</span>';
    return;
  }
  const factors = SHAP_BUCKETS
    .map((b) => ({ ...b, v: row.shap.contribs[b.key] || 0 }))
    .filter((b) => Math.abs(b.v) >= 0.05)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .slice(0, 4);

  let html = `<strong>${formatDate(row.date)}</strong> — forecast ${Math.round(row.shap.prediction)}, starting point ${Math.round(row.shap.base)}.<br>`;
  html += factors.length === 0
    ? "No single factor stands out — the forecast is close to the starting point."
    : "Biggest reasons: " + factors
        .map((f) => `<strong>${f.label}</strong> ${f.v > 0 ? "pushes it up" : "pushes it down"} by ${Math.abs(f.v).toFixed(1)}`)
        .join(", ") + ".";
  el.innerHTML = html;
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

  addRow(null, "Forecast for this day", Math.round(row.shap.prediction));
  addRow(null, "Starting point (before adjustments)", Math.round(row.shap.base));
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

const TABS = ["overview", "stores", "products", "exceptions", "forecasts"];

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
  document.getElementById("storeSortBy").addEventListener("change", (e) => {
    storeSortBy = e.target.value;
    renderStoresTab();
  });
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
    parseShapCsv, SHAP_BUCKETS, computeRollups, state,
  };
}
