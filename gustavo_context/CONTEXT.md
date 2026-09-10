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
