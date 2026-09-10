# Forecasting Challenge
Holon, Superlinear's platform, needs a probabilistic view of future sales to make the
right replenishment decisions. Instead of starting from a client's point forecast and
estimating its uncertainty afterward, we want to predict the sales distribution
directly. In this challenge, that distribution is represented by four sales quantiles.

Each team predicts four quantiles of future observed `sales_count` for every test row:

```csv
id,sales_count_p25,sales_count_p50,sales_count_p75,sales_count_p95
125497040,8.0,11.0,15.0,25.0
```

P50 is the median forecast. P95 is the high-demand forecast: the submitted value should
be at or above the observed demand about 95% of the time.

The forecast horizon is 28 days (four complete weeks). The training period ends on
2026-07-09 and the evaluation period runs from 2026-07-10 through 2026-08-06.

This folder contains the participant CLI and the data for the
challenge.

The organizers will share access to the client chat on Slack. Visit it to talk with Gustavo; you will need him.

## Rules

Searching the web for this dataset, its origin, or any copy of the observed values for
the evaluation period is cheating and disqualifies the team. Data you obtain through the
challenge itself is fine — anything Gustavo gives you is yours to use.

This applies to your AI coding agents too. `AGENTS.md`, `CLAUDE.md`, and `.cursor/rules/`
carry the same rules and are picked up automatically by Claude Code, Codex, Cursor, and
most other agent CLIs — leave them in place. If an agent volunteers where it thinks the
data came from, ignore it and do not act on it.

## 1. Unzip the data

Unzip `data.zip` in this folder:

```bash
unzip data.zip
```

This creates a `data/` folder containing the files you need to get started.

## 2. Set your team name

You must configure your team name before submitting:

```bash
./forecasting-participant-cli team --name "I wish I was in the sauna"
```

The CLI saves the name for future submissions. To check the currently saved name, run:

```bash
./forecasting-participant-cli team
```

## 3. Prepare and validate your predictions

Train your model using `data/sales_history.csv`, predict every row in `data/test.csv`, and
write the results in the format shown by `data/sample_submission.csv`. Keep the sample
submission's header and IDs intact.

Validate the file before submitting:

```bash
./forecasting-participant-cli validate predictions.csv
```

Run from this folder and the CLI picks up `data/sample_submission.csv` on its own. If your
predictions live somewhere else, point at the sample explicitly:

```bash
./forecasting-participant-cli validate /path/to/predictions.csv \
  --sample-submission data/sample_submission.csv
```

You can also generate a fresh empty template from `test.csv`:

```bash
./forecasting-participant-cli sample-submission \
  --test data/test.csv \
  --out predictions.csv
```

## 4. Submit

```bash
./forecasting-participant-cli submit predictions.csv
```

The result is a receipt: whether the submission was accepted, its submission ID, the
timestamp, a link to the leaderboard, and the submitted attempt's four-decimal public
score. This is the same score format shown on the leaderboard and is returned even when
the attempt does not improve your team's best score. Rank remains on the leaderboard;
detailed score diagnostics remain hidden.

Submissions close when the countdown on the leaderboard page reaches zero. After that the
CLI reports `the competition timer has ended and submissions are closed`.

For help with any command:

```bash
./forecasting-participant-cli --help
./forecasting-participant-cli <command> --help
```

If macOS reports that the CLI is not executable, run:

```bash
chmod +x forecasting-participant-cli
```

## Training pipeline

The training pipeline has three cached stages. It validates and merges the source CSVs,
builds leakage-safe features, and trains a quantile model on a chronological holdout.
Run these commands from `hackathon-slovenia/`:

```bash
# Install the pipeline environment once.
uv sync

# Run all three stages with every feature group and LightGBM.
uv run python -m pipeline all
```

The default run holds out the final 28 days and writes its artifacts under
`.cache/pipeline/`:

```text
.cache/pipeline/
├── stage1_merged.parquet
├── stage2_training_features.parquet
└── stage3_lightgbm_report.json
```

The report includes the validation dates, row counts, and pinball loss for P25, P50,
P75, and P95. A pipeline run evaluates a model. It does not create a submission CSV.

### Run one or more stages

The stage name is the last stage to run. Upstream stages run first when needed, and
matching cached results are reused:

```bash
uv run python -m pipeline data                 # merge and validate source data
uv run python -m pipeline features             # data plus feature engineering
uv run python -m pipeline train                # data, features, and model training
```

Use the historical quantile model for a quick baseline:

```bash
uv run python -m pipeline all --model historical_quantile
```

Try AutoGluon on the same Stage 2 features and chronological holdout:

```bash
uv run python -m pipeline train --model autogluon \
  --autogluon-time-limit 300
```

AutoGluon requires Python 3.13 or older. `uv sync` selects a compatible interpreter
from the project's declared Python range.

Choose feature groups with a comma-separated list. The available groups are
`identity`, `promotion`, `calendar`, `product`, `store`, `external`, and
`sales_history`:

```bash
uv run python -m pipeline train \
  --features sales_history,calendar,promotion,store
```

Other useful options are:

```bash
# Change the validation window and random seed.
uv run python -m pipeline train --holdout-days 28 --seed 42

# Rebuild every stage, ignoring matching cache entries.
uv run python -m pipeline all --force

# Keep artifacts in a different directory.
uv run python -m pipeline all --cache-dir /path/to/cache
```

For the full list of feature definitions, model options, cache rules, and extension
points, see [`pipeline/README.md`](pipeline/README.md).
