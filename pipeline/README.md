# Forecasting pipeline

This package runs the training workflow in three cached stages:

1. Data processing reads the challenge CSV files, validates them, and writes one merged
   Parquet dataset.
2. Feature engineering encodes the selected feature groups and writes a training-ready
   Parquet table.
3. Model training makes a chronological holdout split, fits the selected model, and
   writes a JSON evaluation report.

Run commands from the `hackathon-slovenia` directory.

## Setup

Install the Python environment with uv:

```bash
uv sync
```

The pipeline expects these files under `data/`:

```text
data/
├── sales_history.csv
├── test.csv
├── item_catalogue.csv
├── store_info.csv
├── events_and_holidays.csv
├── index_oil.csv
└── sample_submission.csv
```

Stage 1 also reads `gustavo_context/store_traffic.csv`. Override that directory with
`--context-dir`. Traffic is observed for the training period only; forecast rows keep a
missing value rather than assuming future transaction counts.

## Run the full pipeline

```bash
uv run python -m pipeline all
```

The default run selects the locally validated `best` feature preset, trains LightGBM,
and holds out the final 28 days for evaluation. Pass `--features all` to include every
registered group, including rejected experiments and raw identifiers.

Use the historical quantile baseline for a fast end-to-end check:

```bash
uv run python -m pipeline all --model historical_quantile
```

## Run individual stages

The positional argument selects the last stage to run. The command checks its
prerequisites first, so stale upstream artifacts rebuild automatically.

```bash
# Stage 1 only
uv run python -m pipeline data

# Stages 1 and 2
uv run python -m pipeline features

# All stages, ending with model training
uv run python -m pipeline train
```

Generated files live under `.cache/pipeline/`:

```text
.cache/pipeline/
├── stage1_merged.parquet
├── stage1_manifest.json
├── stage2_training_features.parquet
├── stage2_manifest.json
├── stage3_<model>_report.json
└── stage3_<model>_manifest.json
```

## Select features

Pass a comma-separated list to `--features`:

```bash
uv run python -m pipeline features \
  --features sales_history,calendar,promotion,store
```

The registered groups are:

| Group | Contents |
|---|---|
| `identity` | Encoded store and product identifiers |
| `promotion` | Promotion flag |
| `calendar` | Year, month, day, weekday, week number, and weekend flag |
| `product` | Product family, class, and perishability |
| `store` | City, department, store type, and cluster |
| `external` | Oil, store traffic, and event or holiday context |
| `sales_history` | Sales lags plus rolling means and standard deviations |
| `promotion_regime` | Recent promotion history, starts, post-promotion days, and interactions |
| `seasonal_history` | Extra lags, matching-weekday history, rolling medians, and trend |
| `intermittency` | Zero rates, nonzero means, stockout rates, and coefficient of variation |
| `payday` | Flags around the 3rd and 18th, plus a promotion interaction |
| `holiday_regime` | Effective-holiday window interactions |
| `behavioral_profile` | Rolling sales quantiles and promotion response without product ID |
| `oil_trend` | Lagged oil levels, moving averages, and longer-run change |
| `hierarchy` | Lagged product, store, and store-family demand averages |

Traffic features are leakage-safe: the model uses prior-day and trailing store
transaction counts, plus a 7-day/28-day ratio. Future dates have no observed traffic;
the pipeline carries forward the last as-of store baseline and marks those rows with
`traffic_source_missing` rather than treating missing traffic as zero.

`--features best` selects the feature groups that improved the recorded holdout score.
It excludes raw product ID, payday, and the extra holiday interactions. `--features all`
selects every group. Stage 2 retains identifiers, the date, and the target alongside
columns prefixed with `feature_`.

The feature-selection results are recorded in
`experiments/FEATURE_ABLATION.md` and `experiments/lightgbm_holdout_results.json`.

To add a group, write a feature-expression builder in `features.py` and register it in
`FEATURE_GROUPS`. The selected group names are part of the Stage 2 cache key.

## Select a model

Choose a registered model with `--model`:

```bash
uv run python -m pipeline train --model lightgbm
uv run python -m pipeline train --model autogluon
uv run python -m pipeline train --model historical_quantile
```

`lightgbm` fits one quantile model for P25, P50, P75, and P95. The
`autogluon` model fits a tabular quantile ensemble on the same Stage 2 feature columns.
Set its training budget and preset with `--autogluon-time-limit` and
`--autogluon-presets`. The default budget is 300 seconds with the `medium_quality`
preset. The
`historical_quantile` model predicts the observed per-series quantiles and falls back to
global quantiles for an unseen series.

To add a model, implement the prediction function in `models.py` and add it to `MODELS`.
The model name, split settings, seed, feature selection, model code, and feature artifact
all contribute to the Stage 3 cache key.

## Configure the split

Stage 3 uses the latest dates as the validation period:

```bash
uv run python -m pipeline train \
  --model lightgbm \
  --holdout-days 28 \
  --seed 42 \
  --num-boost-round 500 \
  --learning-rate 0.03 \
  --num-leaves 31 \
  --min-data-in-leaf 20 \
  --feature-fraction 1.0 \
  --bagging-fraction 1.0 \
  --bagging-freq 0
```

The report contains the split dates and row counts, pinball loss for each quantile, and
mean pinball loss. The loss applies `log1p` to actual and predicted sales, matching the
competition scorer's target scale.

The pipeline trains on `log1p(sales_count)` by default and converts predictions back to
sales units before scoring them. Pass `--no-log-sales` to train on raw sales instead.
The setting is part of the Stage 3 cache key and appears as `target_transform` in the
report.

The LightGBM settings are included in the report metadata and cache key. The defaults
use 500 boosting rounds at a 0.03 learning rate, based on local holdout comparison.

## Train and create a submission

The recursive forecaster carries the strongest behavior from the earlier LightGBM
experiment into this package. By default it evaluates three backward rolling 14-day
windows, then trains four quantile models for the final forecast, feeds each predicted
median into the next day's lag features, validates submission IDs and quantile order, and
writes the prediction file. Use `--eval-windows 2` for a faster check or
`--eval-windows 1` to reproduce a single holdout.

To avoid feeding predictions back into later sales-history features, select the direct
strategy:

```bash
uv run python -m pipeline.forecast \
  --feature-groups all \
  --forecast-strategy direct
```

Direct forecasting trains a separate set of four quantile models for every lead in the
test window. A lead-30 model learns from historical 30-day-ahead targets and forecasts
from the last observed sales date. Known target-date context is still included. This is
much more expensive than the default recursive strategy: a 30-day window fits 120 models
for the final forecast, plus another 120 when holdout evaluation is enabled. Use
`--skip-holdout` when only a submission is needed.

Build Stage 1 first, then run:

```bash
uv run python -m pipeline data
uv run python -m pipeline.forecast --feature-groups all
```

Submission models also use `log1p` sales by default. Pass `--no-log-sales` to use raw
sales. Recursive lag features and submitted quantiles remain in ordinary sales units.

Predictions go to `pipeline/predictions/`. Models, holdout metrics, and diagnostics go to
`pipeline/artifacts/`. Pass `--output predictions.csv` to choose an exact submission
path. The command refuses to overwrite an existing prediction file.

Pass `--shap` to write `<prediction-name>.shap.parquet` beside the submission. SHAP
output is disabled by default because it makes recursive forecasting much slower. The
sidecar has one row per submission ID and quantile, with raw and submitted predictions,
the source quantile model, the SHAP base value, and one `shap_<feature>` column per model
feature. `target_transform` records the model scale, while `raw_model_prediction` keeps
the value explained by the additive SHAP columns. `postprocessing_adjustment` records
negative-value clipping. When quantile
crossing correction reorders predictions, `source_model_quantile` identifies which model
produced each submitted quantile.

## Cache behavior

Each manifest fingerprints:

- direct input file contents;
- stage parameters;
- files that implement the stage;
- the upstream artifact for Stage 2 and Stage 3.

Changing Stage 1 code rebuilds all stages. Changing only model code reuses Stages 1 and
2, then rebuilds Stage 3. The cache also checks the output file digest before accepting a
hit.

Force a rebuild with:

```bash
uv run python -m pipeline all --force
```

Use another cache directory with `--cache-dir /path/to/cache`.

## Tests

Run the complete test suite:

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover cache invalidation, selectable feature groups, leakage-safe lag
construction, chronological splitting, the historical model, LightGBM, AutoGluon,
event handling, and source validation.
