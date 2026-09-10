// Minimal self-check for the non-trivial logic in app.js (CSV quoting,
// monotonicity validation, holiday scope matching). Run: node test_app.js
const assert = require("assert");
const { splitCSVLine, eventAppliesToStore, validateAndIndexSubmission, state } = require("./app.js");

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

console.log("All self-checks passed.");
