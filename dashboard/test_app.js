// Minimal self-check for the non-trivial logic in app.js (CSV quoting,
// monotonicity validation, holiday scope matching). Run: node test_app.js
const assert = require("assert");
const { splitCSVLine, eventAppliesToStore, validateAndIndexSubmission, scanExceptions, isPayday, FLAG_TYPES, ACTION_BY_FLAG, parseShapCsv, state } = require("./app.js");

// SHAP CSV parsing (format written by scripts/export_shap_for_dashboard.py):
// nested by id -> quantile -> {base, prediction, contribs}, and bucket sums
// must reproduce prediction - base (that's the whole point of SHAP additivity).
const shapCsv = [
  "id,quantile,base_value,prediction,contrib_sales_trend,contrib_promotion",
  "t1,0.5,100,150,40,10",
  "t1,0.25,90,120,20,10",
].join("\n");
const shapIndex = parseShapCsv(shapCsv);
const t1p50 = shapIndex.get("t1").get("0.5");
assert.strictEqual(t1p50.base, 100);
assert.strictEqual(t1p50.prediction, 150);
assert.strictEqual(t1p50.contribs.sales_trend, 40);
assert.strictEqual(t1p50.contribs.promotion, 10);
assert.strictEqual(t1p50.contribs.sales_trend + t1p50.contribs.promotion, t1p50.prediction - t1p50.base);
assert.strictEqual(shapIndex.get("t1").get("0.25").base, 90);

// Payday dates (3rd/18th of any month).
assert.strictEqual(isPayday("2026-07-03"), true);
assert.strictEqual(isPayday("2026-07-18"), true);
assert.strictEqual(isPayday("2026-07-04"), false);

// Every flag type has an action mapped to it — the exceptions table renders
// ACTION_BY_FLAG[f.type] unconditionally, so a gap here would render "undefined".
Object.keys(FLAG_TYPES).forEach((type) => {
  assert.ok(ACTION_BY_FLAG[type], `missing ACTION_BY_FLAG entry for ${type}`);
});

// CSV parsing: plain line and a quoted field with an embedded comma
// (matches the real row in gustavo_context/item_catalogue.csv).
assert.deepStrictEqual(splitCSVLine("a,b,c"), ["a", "b", "c"]);
assert.deepStrictEqual(
  splitCSVLine('1004550,"LIQUOR,WINE,BEER",1318,0'),
  ["1004550", "LIQUOR,WINE,BEER", "1318", "0"]
);

// Holiday scope matching.
const store = { city: "Medellín", department: "Antioquia" };
assert.strictEqual(eventAppliesToStore({ scope: "National", is_transferred: "False" }, store), true);
assert.strictEqual(eventAppliesToStore({ scope: "Regional", location: "Antioquia", is_transferred: "False" }, store), true);
assert.strictEqual(eventAppliesToStore({ scope: "Regional", location: "Atlántico", is_transferred: "False" }, store), false);
assert.strictEqual(eventAppliesToStore({ scope: "Local", location: "Medellín", is_transferred: "False" }, store), true);
assert.strictEqual(eventAppliesToStore({ scope: "National", is_transferred: "True" }, store), false);

// Submission validation: schema, id matching, monotonicity.
state.testById.set("1", {});
state.testById.set("2", {});
const csv = [
  "id,sales_count_p25,sales_count_p50,sales_count_p75,sales_count_p95",
  "1,10,20,30,40",   // valid
  "2,50,20,30,40",   // p25 > p50 -> violation
  "3,1,2,3,4",        // id not in testById -> extra
].join("\n");
const result = validateAndIndexSubmission(csv);
assert.strictEqual(result.schemaOk, true);
assert.strictEqual(result.monotonicViolations, 1);
assert.strictEqual(result.missing, 0); // both "1" and "2" present
assert.strictEqual(result.extra, 1);   // "3" is unexpected

// Exception scanning: essential-low + baseline-deviation on a single row,
// and confirm a valid quantile order doesn't falsely trigger monotonicity.
state.stores.set("S1", { city: "TestCity", department: "TestDept", store_type: "A", store_cluster: "1" });
state.products.set("P1", { product_family: "BREAD/BAKERY", product_class: "1", is_perishable: "0" });
state.testById.set("t1", { date: "2026-07-10", store_id: "S1", product_id: "P1", promotion: "False" });
state.seriesIndex.set("S1|P1", [{ id: "t1", date: "2026-07-10", promotion: "False" }]);
state.baselineById.set("t1", { p25: 5, p50: 10, p75: 15, p95: 20 });
state.submissionById = new Map([["t1", { p25: 0, p50: 0, p75: 0, p95: 3 }]]);

let flags = scanExceptions();
assert.ok(flags.some((f) => f.type === "essential_low" && f.id === "t1"));
assert.ok(flags.some((f) => f.type === "baseline_deviation" && f.id === "t1"));
assert.ok(!flags.some((f) => f.type === "monotonicity"));

// Exception scanning: promo-day-with-no-lift and holiday-treated-as-normal,
// both of which need the per-series non-promo/non-holiday P50 average.
state.products.set("P2", { product_family: "DAIRY", product_class: "1", is_perishable: "0" });
state.events.push({ date: "2026-07-15", event_type: "Holiday", scope: "Local", location: "TestCity", is_transferred: "False" });
const p2Rows = [
  ["t2", "2026-07-10", "False", 10],
  ["t3", "2026-07-11", "False", 10],
  ["t4", "2026-07-14", "False", 10], // day before the 07-15 holiday
  ["t5", "2026-07-20", "True", 10],  // promoted, but no lift over the ~10 baseline
];
state.seriesIndex.set("S1|P2", p2Rows.map(([id, date, promotion]) => ({ id, date, promotion })));
const subMap = new Map([["t1", { p25: 0, p50: 0, p75: 0, p95: 3 }]]);
p2Rows.forEach(([id, date, promotion, p50]) => {
  state.testById.set(id, { date, store_id: "S1", product_id: "P2", promotion });
  state.baselineById.set(id, { p25: p50, p50, p75: p50, p95: p50 });
  subMap.set(id, { p25: p50, p50, p75: p50, p95: p50 });
});
state.submissionById = subMap;

flags = scanExceptions();
assert.ok(flags.some((f) => f.type === "promo_no_lift" && f.id === "t5"));
assert.ok(flags.some((f) => f.type === "holiday_flat" && f.id === "t4"));

// Exception scanning: zero-forecast + wide-spread for a non-staple product
// (these must NOT be gated to staple families / perishables the way
// essential_low / perishable_spike are).
state.products.set("P3", { product_family: "LIQUOR,WINE,BEER", product_class: "1", is_perishable: "0" });
state.testById.set("t6", { date: "2026-07-10", store_id: "S1", product_id: "P3", promotion: "False" });
state.testById.set("t7", { date: "2026-07-11", store_id: "S1", product_id: "P3", promotion: "False" });
state.seriesIndex.set("S1|P3", [
  { id: "t6", date: "2026-07-10", promotion: "False" },
  { id: "t7", date: "2026-07-11", promotion: "False" },
]);
subMap.set("t6", { p25: 0, p50: 0, p75: 0, p95: 5 });   // zero median, non-staple family
subMap.set("t7", { p25: 0, p50: 2, p75: 10, p95: 30 }); // spread 30 >> 2.5 * p50
state.submissionById = subMap;

flags = scanExceptions();
assert.ok(flags.some((f) => f.type === "zero_forecast" && f.id === "t6"));
assert.ok(!flags.some((f) => f.type === "essential_low" && f.id === "t6"));
assert.ok(flags.some((f) => f.type === "wide_spread" && f.id === "t7"));

console.log("All self-checks passed.");
