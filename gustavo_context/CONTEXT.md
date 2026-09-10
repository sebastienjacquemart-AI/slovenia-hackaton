# Gustavo Context — tracking file

Running notes on everything Gustavo shares, so it's easy to hand off to Jan
(forecasting) without re-reading every raw file. Update this whenever a new
file lands in this folder.

## Files received

### `store_info.csv` (20 rows)
Store metadata: `store_id, city, department, store_type, store_cluster`.

### `item_catalogue.csv` (500 rows)
Product metadata: `product_id, product_family, product_class, is_perishable`.

### `events_and_holidays.csv` (24 rows)
Calendar events: `date, event_type, scope, location, is_transferred`.
`scope` is National / Regional / Local; `is_transferred` flags a holiday
moved to a different date.

### `index_oil.csv` (149 rows)
Daily macro indicator: `date, oil_price`. Market-day based (weekends absent)
and has **6 missing observations** — Gustavo flagged this explicitly, treat
as a data-quality caveat, not something to silently forward-fill without
noting it.

### `store_traffic.csv` (3,600 rows, added 2026-09-10)
Daily footfall per store: `date, store_id, transaction_count`. Same date
range as `sales_history.csv` (2026-01-11 to 2026-07-09). Useful as a
denominator/sanity-check: if a store's overall traffic is normal on a day
one product records zero sales, that argues against a store-wide
disruption (closure, weather, local event) and toward something
product-specific (stockout or genuine zero demand for that item) — see the
store 44 case study below for a worked example. It does **not** by itself
prove either stockout or zero demand for a single product.

## Open questions for Gustavo

- (none yet)

## Team

- **Gustavo** — client contact, source of the `gustavo_context` files.
- **Olivier Libert** — team, gave store-segmentation and promotion context.
- **Sanne Van Ransbeeck** — team, gave holiday/transfer-date context.

## Competition scoring (verified 2026-09-10)

The participant CLI's local scorer confirms that the leaderboard uses
row-weighted pinball loss after applying `log1p` to both actual and predicted
sales. For row `i` and quantile probability `q`:

```text
z_i       = ln(1 + actual_i)
z_hat_i,q = ln(1 + prediction_i,q)
error_i,q = z_i - z_hat_i,q

pinball_i,q = max(q * error_i,q, (q - 1) * error_i,q)

score = sum_i(weight_i * sum_q(pinball_i,q))
        / (4 * sum_i(weight_i))
```

The four submitted quantiles, P25, P50, P75, and P95, receive equal weight in
the final mean. Their pinball penalties are directionally asymmetric:

| Quantile | Underforecast coefficient | Overforecast coefficient |
|---|---:|---:|
| P25 | 0.25 | 0.75 |
| P50 | 0.50 | 0.50 |
| P75 | 0.75 | 0.25 |
| P95 | 0.95 | 0.05 |

P95 therefore penalizes underforecasting 19 times more than overforecasting.
P25 penalizes overforecasting three times more than underforecasting. There is
no evidence of an extra quantile-specific multiplier beyond these standard
pinball coefficients.

This was checked with a synthetic one-row ground-truth file. With actual sales
of 10, unit row weight, and all four predictions set to zero, the CLI returned
`1.468710854589002`, exactly equal to
`ln(11) * (0.25 + 0.50 + 0.75 + 0.95) / 4`. A second exact-prediction row with
weight 3 reduced the score to one quarter, confirming the row-weighted
denominator.

The public leaderboard scores only 20% of the evaluation data. The site does
not disclose how it selects that subset. After the competition timer ends, the
private leaderboard uses the full evaluation dataset. The scorer supports row
weights, but there is no evidence that production uses unequal positive row
weights beyond selecting the public subset.

The validation functions in `pipeline/stages/model_training.py` and
`pipeline/forecast.py` apply `log1p` before calculating pinball loss. Local validation
uses equal row weights because the production row weights are not available.

## Domain notes from the team (Olivier & Sanne, 2026-09-10)

**Holidays (`events_and_holidays.csv`) — critical for feature engineering:**
- `is_transferred = True` marks the *original* official date of a holiday
  that got moved. That original date trades like an ordinary day — do
  **not** treat it as a holiday.
- The actual day off (where demand shifts) is the separate row with
  `event_type = Transfer`, a few days later. Confirmed in the data, e.g.
  `2026-04-15` (`Holiday`, `is_transferred=True`) → `2026-04-16`
  (`Transfer`, the real day off).
- Don't double-count: only the `Transfer` row should be treated as the
  holiday for demand purposes, not both dates.
- Days immediately *before* a real holiday can see a demand bump (people
  stocking up/preparing) — worth a feature.

**Promotions:**
- `promotion = True` in the sales data means the product was actively
  pushed in that specific store on that date (placement, signage,
  sometimes a different price) — not just "on sale" generically. Treat
  promo days as a distinct demand regime, not ordinary shelf demand.

**Store fields — complementary, not redundant:**
- `city` — local customer base, habits, local events.
- `department` — regional context, which region-level events apply.
- `store_type` — format (e.g. large-format vs. neighbourhood store) drives
  different trading patterns.
- `store_cluster` — Olivier's own operational grouping by comparable size,
  customer base, and trading behaviour. **Not geographic** — clusters can
  cross city/department lines. Treat as the most informative single
  store-level feature.
- Event applicability cascades by `scope` (National > Regional > Local) —
  join holidays to stores via matching `location`/`department`/nationwide.
- Same product can play a different role (staple vs. rarely-moving) in
  different stores — store×product interactions matter, not just store or
  product effects alone.
- **Sample caveat**: the 20 stores in `store_info.csv` are the estate's
  20 *highest-volume* stores — not a random sample. Don't assume findings
  generalize to a "typical" store.
- No demographic data available/will be provided — don't infer or assume
  demographics.

**Oil price (`index_oil.csv`) — Sanne, 2026-09-10:**
- Effect is **lagged**, not a same-day/till event. Sanne expects the impact
  to show up in baskets months after a sharp oil-price fall, once income
  and local spending confidence soften — no precise lag figure given, she
  was explicit about not inventing one.
- The visible effect is customers **trading down**: smaller basket
  quantities, less discretionary/premium buying — not a uniform sales drop.
- Impact likely varies by product family and by store (local customer
  base / economic exposure), but there's no quantified category×store
  breakdown available from the team.

**Product risk tiers (staple vs. discretionary) — Gustavo & Olivier, 2026-09-10:**
- `item_catalogue.csv` has **no** essential/discretionary flag — only
  `product_family` / `product_class`. The staple-vs-discretionary split
  below is the team's commercial judgement, not something encoded in the
  file.
- **Staples/essentials**: bread and other basic food lines, household
  cleaning (e.g. detergent), everyday replenishment items.
- **Discretionary/premium**: wine and other treats, premium versions of
  ordinary goods, anything customers can easily postpone or trade down
  from.
- **Middle ground**: many categories depend on pack size, brand
  positioning, and the store's customer base — a basic item and a premium
  version of it can sit in the same `product_family`. Gustavo was clear
  he can't classify all 500 products precisely from the catalogue alone.

**Under- vs. over-forecast risk asymmetry — Olivier, 2026-09-10:**
Directly relevant to the P25/P50/P75/P95 quantile task — which tail matters
more isn't uniform across products/stores:
- **Protect hardest against under-forecasting** (favor higher quantiles)
  for: promoted lines (promo already creates demand — empty shelf wastes
  the promo spend), perishables/fresh (a missed sale is gone, often the
  whole basket goes with it), staples/high-volume items especially around
  holidays/payday/local events, large-format stores (higher baseline
  volume and promo exposure).
- **Protect harder against over-forecasting** (favor lower quantiles) for:
  perishables (surplus becomes waste fast — note this cuts *both* ways
  with the point above: fresh needs a tight band, not just a high one),
  discretionary goods like wine (more variable demand, excess ties up
  cash), items in smaller/lower-volume stores, products tied to uncertain
  local events (don't assume the event will drive footfall).
- Practical read for Jan: a single symmetric quantile spread per
  store×product is probably wrong. Fresh + promoted + staple should skew
  wider/higher; discretionary + low-volume-store should skew tighter/lower.

## Q&A with Gustavo (2026-09-10)

**Stock-availability priorities:**
- Most critical to keep in stock despite uncertainty: perishables, staples/
  high-volume products (esp. ahead of a holiday, **payday — the 3rd and
  18th of the month**, or a local event), promoted lines, products in
  large-format stores, essentials in high-traffic urban stores (Medellín,
  Barranquilla).
- Risk compounds: a promoted fresh staple in a large-format store before a
  holiday is the "protect hardest" case — not where a cautious median
  forecast belongs.
- Won't name specific critical product IDs from the catalogue alone — that
  needs actual sales behaviour, not just metadata.

**Promotion incrementality vs. cannibalization:**
- Not all promotions create *new* demand — varies by product/store.
- More likely **incremental** (genuinely additional demand): staples/
  essentials, strong visibility + real price/placement change, timed
  before a holiday/payday/local event, enough store traffic and normal
  stock levels.
- More likely **pull-forward / cannibalized** (shifts sales from nearby
  days, spike now + dip after): discretionary items, planned household
  purchases, goods bought in bulk because temporarily cheap.
- The `promotion` flag only tells you we pushed the item — not whether it
  created net-new demand. That has to be inferred from sales history
  (e.g. compare sales in the days immediately after a promo ends).

**Promotion mechanics, more detail (Sanne relaying Gustavo, 2026-09-10):**
- `promotion=True` means the product was actively pushed in that store on
  that date — better placement, signage, sometimes a different price. The
  data does **not** say which mechanism was used or how strong the offer
  was.
- Execution varies by store — the same product can react very differently
  depending on format, cluster, customer base, local context.
- Three possible effects: genuinely incremental demand, pull-forward
  (softer sales afterwards), or cannibalization (customers switch
  brand/pack size/tier).
- Product-specific: cooking oil/staples often show a strong unit lift
  (stock-up behaviour), typically followed by softer sales after; wine/
  premium/discretionary react strongly but inconsistently (clientele,
  occasion, placement, price all matter); perishables are operationally
  most sensitive — a strong promo moves volume fast, and availability/waste
  make misses expensive.
- **No universal promotion multiplier** — Gustavo explicitly warned against
  one uplift factor across all products/stores. Promoted demand is a
  different regime from normal shelf demand.
- A promoted zero is especially suspicious but still doesn't *prove* a
  stockout or failed promo — no shelf-availability/execution data exists to
  confirm it.
- **Modeling suggestion:** think in terms of promotion × product family and
  promotion × store/cluster interactions at minimum, and consider testing
  for post-promotion softness rather than a single True/False uplift term.

**Lead time before holiday/payday:** no fixed number of days — varies by
occasion, product, store. Fresh/staples/beverages/entertaining goods react
earlier than routine household lines; payday effects split between the
exact date and the surrounding days depending on product/location. Avoid
hardcoding a single lead-time window as a feature.

**Regional/local sensitivity by product family:** fresh/perishables,
beverages/entertaining (incl. wine), staples/essentials (regional income &
payday timing shift basket size/brand choice), and "local habit" products
(routine in one city, occasional elsewhere) are most sensitive. Expects
stronger differences across **store cluster/format** than from product
family alone.

**Zero sales ≠ zero demand:** a `0` in `sales_count` only means no units
were recorded sold — not that the shelf was stocked and no one wanted it.
Common causes besides genuine low demand: out of stock, poor placement/
visibility, no promotion when one was expected, product not relevant to
that store's customer base, unusual calendar day. **The data doesn't
distinguish these** — treating every zero as true zero demand is a
confident way to be wrong.

**Not derivable from these extracts** (asked directly, Gustavo declined to
guess): perishable-category shelf-life/waste tolerance, routine
product-pairing/basket co-purchase effects, which named store clusters
behave differently under promo/holiday/payday. All would require checking
raw sales history, not something to reverse-engineer from the catalogue.

**Strategic importance beyond average volume** — a modest-volume
store×product pair still matters when it's perishable, an essential line,
under active promotion, holiday/event-relevant, important to one
particular store's local customer base (even if not estate-wide), or
high-variance/surprising (e.g. wine, premium lines).

**Where to be conservative (favor high quantiles) vs. accept risk (favor
low quantiles)** — reinforces the earlier asymmetry notes, with specifics:
- Conservative: perishables, promoted lines (esp. cooking oil, household
  staples), holiday/event periods, essential everyday products, store-
  product pairs important to a specific local base.
- More risk-tolerant: non-perishable, demand driven by a one-off planned
  promotion, discretionary/unpredictable premium lines (wine) where
  overcommitting inventory just leaves stock sitting.

**Morning review-exception checklist** (what a store manager checks first
— maps directly to dashboard "flag" ideas, see Notes for Jan):
1. Perishables with high P95 but much lower P50 (stockout risk on fresh).
2. Promoted lines with elevated P75/P95, esp. cooking oil, staples, wine
   (empty shelf under a promo sign).
3. Dates around a holiday/Transfer/additional day/major event — including
   the day before, not just the day itself.
4. Essential everyday products with a low P25 or P50 (a modest staple
   missing still damages the whole trip).
5. Store-product forecasts that look unusually different from that
   store's normal behaviour.
6. High-uncertainty lines where P95 is dramatically above P50 (esp. wine,
   premium, promoted perishables) — needs human judgement, not blind trust.
7. Non-perishable discretionary lines with high P95 but weak middle
   figures — acceptable stockout risk, don't over-stock for them.

**What would make Gustavo distrust a forecast even if the estate total
looks fine:**
- Store-level picture wrong even though the total nets out okay.
- Perishables with low P25/P50 but high P95 around promos/events.
- Promoted staples (esp. cooking oil) forecast like an ordinary day.
- Wine/premium lines shown with implausibly tight/precise quantiles.
- Holiday/Transfer/pre-holiday days treated like a normal weekday.
- Essential products with implausibly low P25/P50.
- Forecasts that ignore the store's own character (city/format/cluster).
- Too many identical-looking forecasts across products/stores ("my shops
  are not photocopies").
- Systematic zero-handling bias (treating every zero as lost demand, or
  every zero as no interest).
- **Hard rule**: quantiles must be monotonic — `P25 ≤ P50 ≤ P75 ≤ P95` for
  every row. A violation means the forecast contradicts itself.

**Gustavo's consolidated caveat list (2026-09-10)** — sent as "the useful
caveat list" for the business side; mostly reinforces notes already above,
kept together here since he framed it as the canonical set:
1. A zero sale is not proof of zero demand — shelf availability isn't recorded.
2. Promotion changes the selling situation (placement/signage/price); a
   promoted zero deserves attention but doesn't prove a failed promo.
3. Promotions can shift or substitute demand (stock up, buy less after,
   switch brand/size/tier).
4. Holidays must be read by date + scope + location — a transferred holiday
   trades on the transfer date, not the original; a local event doesn't
   apply to every store.
5. Payday (3rd/18th, and surrounding days) is operationally visible.
6. Stores aren't interchangeable — city, format, department, cluster matter;
   compare like with like.
7. Products aren't interchangeable — family, class, perishability matter;
   fresh products have tighter consequences on both shortage and waste.
8. Oil price is context, not sales — market-day series, weekends absent, 6
   missing observations.
9. **The records show outcomes, not explanations** — they can't tell you
   whether an abnormal result came from weak demand, an empty shelf, a
   promotion problem, or an operational incident. His words: "The technical
   interpretation is why you have an analytics team. I run the stores; I do
   not pretend to be a machine-learning engineer in a cheaper suit."

**Case study — store 44 / product 1473478, 2026-02-08 to 09:** Gustavo
asked whether this was a stockout or genuine zero demand: sales ran
~150-200 units/day, dropped to 14.8 on 02-07, hit **0 on both 02-08 and
02-09**, then recovered (50.5 → 207, the latter a promo day) by 02-11.
Gustavo's own answer: **cannot confirm from the extracts** — "consistent
with an availability interruption, but not proof of one... the sales
history contains no stock-availability or replenishment field." He'd need
store 44's operational records to establish start/end times.
- Checked `store_traffic.csv` for store 44 on those dates: transaction
  counts were 5,316 (02-08) and 3,866 (02-09) — both in line with, and one
  of them above, the surrounding days (range ~3,600-5,400 across 02-01 to
  02-15). **The store itself wasn't quiet** — this rules out a store-wide
  disruption (closure, weather, local event) as the explanation, and points
  toward something product-specific. It does not distinguish "stockout" from
  "genuine demand collapse for this one item" — traffic only says people
  came to the store, not that this product was on the shelf or wanted.
- Pattern shape (gradual fall 02-05→02-07, two flat zeros, gradual — not
  instant — recovery 02-10→02-11) is the kind of shape a stockout produces,
  but as Gustavo said, this is suggestive, not proof.

## External links

- Sanne shared a Google Drive folder (2026-09-10):
  https://drive.google.com/drive/folders/1euBBwLbm_jl-TPafqiXJeD9603TEqPsJ?usp=drive_link
  — contents not yet reviewed/downloaded into this repo.

## Notes for Jan (forecasting)

- These files are keyed on `store_id` / `product_id` / `date`, matching the
  join keys in `data/sales_history.csv` and `data/test.csv` — usable as
  external features (store attributes, product attributes, holiday
  calendar, oil price) alongside the core sales data.
- Holiday feature: use `Transfer` rows (and pre-holiday days) as the
  demand-affecting dates, not `is_transferred=True` rows — see domain
  notes above. Getting this backwards will actively hurt the model.
- `store_cluster` is likely the single most informative store-level
  categorical feature (per Olivier) — worth prioritizing over raw
  `store_id`.
- Data covers only the 20 highest-volume stores — keep that in mind when
  reasoning about generalization.
- Oil price is a lagged, indirect signal (trading-down effect, months-long
  lag) — don't expect it to correlate same-day/same-week with sales; a
  rolling/lagged transform is more plausible than the raw level.
- Consider building a simple staple/discretionary tier feature from
  `is_perishable` + `product_family`/`product_class` (see risk-tier notes
  above) — no ready-made flag exists, this would need to be engineered.
- Quantile asymmetry: the "right" spread between P25/P75/P95 probably
  differs by product/store risk tier (see under/over-forecast notes
  above) rather than being a fixed offset from P50.
- Payday dates (3rd and 18th of each month) are a concrete, checkable
  calendar feature — distinct from the holiday calendar file.
- Promotion feature should ideally distinguish likely-incremental vs.
  likely-cannibalized promos (see Q&A above), not treat `promotion=True`
  as one uniform effect.
- Zeros in `sales_count` are censored/ambiguous (could be stockout, not
  true zero demand) — worth flagging as a modeling caveat, not something
  this data can resolve on its own.
- Quantile outputs must satisfy `P25 ≤ P50 ≤ P75 ≤ P95` per row — Gustavo
  called this out as a hard sanity check, and it's a natural validation
  rule for the dashboard too (see below).
- `store_traffic.csv` (daily transaction counts per store) is a candidate
  normalizing feature — e.g. sales-per-transaction, or a same-day-traffic
  check to help separate a store-wide disruption from a product-specific
  one (see the store 44 case study above). Not yet used anywhere.

## Front-end requirements — Sanne relaying Gustavo's ask, 2026-09-10

Framing: the context files explain *what* was happening around a sale, not
*why* a low/zero sale happened — no shelf-availability, promo-execution,
distribution-centre-incident, or manager's-explanation data exists. The
dashboard should not pretend otherwise.

Gustavo's brief for what a good front end does — "helps my team make a
decision before the coffee goes cold, not a museum of colourful charts":
- A clear daily planning view per store/product/date showing P25, P50, P75,
  P95 together, always in that order — the range visible at a glance, not
  just one number.
- Filters for: store, city, cluster, format, product family, product class,
  perishability, promotion, date.
- Calendar context shown beside the forecast — holiday, transfer,
  additional day, local scope, payday timing — and scope must be respected:
  **a Pereira event must not be presented as if it affects Medellín.**
- Promotion clearly marked — promoted demand is not ordinary shelf demand.
- An exceptions page for lines needing attention: unusually high
  uncertainty, forecasted zeros, promoted items with weak expected sales,
  perishables, sharp changes around holidays.
- Store and product drill-downs — go from an estate-level issue to the
  exact store-product-date row without hunting through ten screens.
- Comparison against recent recorded sales — while making clear a zero
  sale doesn't prove weak demand (empty shelf ≠ weak demand).
- A practical action view: what to review for replenishment, promotion
  readiness, staffing, or a manager call.
- Exportable detail — eventually someone needs the exact rows for the
  store team.
- Restrained colour: low/middle/high demand, not a "Christmas tree" —
  abnormal cases should be obvious, ordinary cases quiet.

## Dashboard notes (validation ideas from Gustavo's review checklist)

Gustavo's "morning review exceptions" and "what would make me distrust a
forecast" answers (Q&A above) double as a spec for what the submission
dashboard should flag, beyond basic schema validation:
- Quantile monotonicity violation (`P25 ≤ P50 ≤ P75 ≤ P95`) — hard error.
- Perishable rows with a large P95−P50 gap.
- Promoted rows (join `test.csv.promotion`) with high P75/P95, esp. for
  staple-ish/staple product families.
- Rows on/around a holiday `Transfer` date or the day before.
- Essential/staple rows with low P25/P50.
- Rows with an unusually wide P95−P25 spread relative to similar products
  (possible "false confidence" or "no confidence" outlier).
- Compare against `predictions_baseline.csv` per row to surface large
  deviations for a human to sanity-check, not to auto-correct.

**v2 shipped (2026-09-10):** `dashboard/app.js`'s `scanExceptions()` now
runs six of these checks across the full submission (not just the row
currently on screen) and surfaces them in a filterable, sortable exception
table above the store×product chart, with a "View" action that jumps the
chart to that row: monotonicity violations, perishable P95≫P50 spikes,
promoted days with no forecast lift vs. that series' own non-promo average,
holiday/pre-holiday days that look like a normal day for that series,
essential-family (staple-by-`product_family`) rows with a P50 of 0, and
large deviations from `predictions_baseline.csv`. "Store-product looks
unusually different from that store's normal behaviour" and duplicate-
forecast detection ("my shops are not photocopies") are not implemented —
they'd need a cross-series baseline of "normal" that v1/v2 don't build.

**v3 shipped (2026-09-10), closing the gaps against Sanne/Gustavo's
front-end brief above:**
- Store/product dropdowns replaced with searchable, filterable pick lists
  (city/cluster/format; family/class/perishability).
- Chart now overlays the last 30 days of `data/sales_history.csv` as a
  neutral-toned actuals line ahead of the forecast, with a persistent
  "zero ≠ proven no demand" caveat and a per-point tooltip note on zeros.
- Payday (3rd/18th) shown as a small axis tick — deliberately subtle,
  unlike the holiday dashed lines, since it's routine not exceptional.
- Two more exception checks, generalized beyond perishables/staples:
  `zero_forecast` (any product, not just staples) and `wide_spread`
  (P95−P25 spread unusually wide relative to P50, for any product) — this
  covers Gustavo's "high-uncertainty wine/premium lines" checklist item
  that v2 didn't reach.
- Exceptions table gained promotion and date-range filters, an "Action"
  column (`ACTION_BY_FLAG` — replenishment / promotion readiness /
  calendar readiness / manager review / data integrity) as a lightweight
  stand-in for a separate action view, and a CSV export of the current
  filtered rows.
- UX fix: the exceptions panel is now a collapsed `<details>` showing only
  the flag-count summary until clicked — team feedback was that the full
  table on load was too much at once ("if every row screams, none of them
  does").
- Still not built: the Colombia map (discussed, deprioritized — most
  effort for the least explicitly-requested payoff), and "store-product
  looks unusually different from that store's normal behaviour" /
  duplicate-forecast detection from v2's gap list.
