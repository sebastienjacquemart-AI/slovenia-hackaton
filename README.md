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
./forecasting-participant-cli team --name "Your Team Name"
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
