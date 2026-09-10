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
Daily macro indicator: `date, oil_price`.

## Open questions for Gustavo

- (none yet)

## Team

- **Gustavo** — client contact, source of the `gustavo_context` files.
- **Olivier Libert** — team, gave store-segmentation and promotion context.
- **Sanne Van Ransbeeck** — team, gave holiday/transfer-date context.

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
