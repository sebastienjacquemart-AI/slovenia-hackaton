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

## Run the full pipeline

```bash
uv run python -m pipeline all
```

The default run selects every feature group, trains LightGBM, and holds out the final 28
days for evaluation.

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
| `external` | Oil and event or holiday context |
| `sales_history` | Sales lags plus rolling means and standard deviations |

`--features all` selects every group. Stage 2 retains identifiers, the date, and the
target alongside columns prefixed with `feature_`.

To add a group, write a feature-expression builder in `features.py` and register it in
`FEATURE_GROUPS`. The selected group names are part of the Stage 2 cache key.

## Select a model

Choose a registered model with `--model`:

```bash
uv run python -m pipeline train --model lightgbm
uv run python -m pipeline train --model historical_quantile
```

`lightgbm` fits one quantile model for P25, P50, P75, and P95. The
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
  --seed 42
```

The report contains the split dates and row counts, pinball loss for each quantile, and
mean pinball loss.

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
construction, chronological splitting, the historical model, LightGBM, event handling,
and source validation.
