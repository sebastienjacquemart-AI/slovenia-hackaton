"""Train AutoGluon on the existing forecast features and create a submission."""

from __future__ import annotations

import argparse
import random
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .forecast import (
    FEATURE_NAMES,
    QUANTILES,
    features_for_next,
    load_context_panel,
    load_sales_panel,
    make_supervised,
    parse_feature_groups,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    pipeline_dir = Path(__file__).resolve().parent
    challenge_dir = pipeline_dir.parent
    parser = argparse.ArgumentParser(
        description="Train an AutoGluon quantile model and create a submission."
    )
    parser.add_argument("--data-dir", type=Path, default=challenge_dir / "data")
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=challenge_dir / ".cache" / "pipeline" / "stage1_merged.parquet",
    )
    parser.add_argument("--feature-groups", default="all")
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=pipeline_dir / "artifacts" / "autogluon"
    )
    return parser.parse_args()


def _frame(
    values: np.ndarray,
    feature_names: tuple[str, ...],
    categorical_names: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=feature_names)
    for name in categorical_names:
        frame[name] = frame[name].astype("category")
    return frame


def recursive_autogluon_forecast(
    predictor: object,
    history: np.ndarray,
    horizon: int,
    feature_names: tuple[str, ...],
    categorical_names: tuple[str, ...],
    future_context: np.ndarray | None,
) -> np.ndarray:
    forecasts = []
    working_history = history.copy()
    for day in range(horizon):
        context_row = None if future_context is None else future_context[day]
        features = features_for_next(working_history, context_row)
        predicted = predictor.predict(
            _frame(features, feature_names, categorical_names)
        )
        columns = [
            next(column for column in predicted.columns if float(column) == quantile)
            for quantile in QUANTILES
        ]
        coherent = np.maximum.accumulate(
            np.maximum(predicted[columns].to_numpy(), 0.0), axis=1
        ).astype(np.float32)
        forecasts.append(coherent)
        working_history = np.vstack((working_history, coherent[:, 1]))
        print(f"Forecast day {day + 1}/{horizon}", end="\r", flush=True)
    print()
    return np.stack(forecasts)


def main() -> None:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as error:
        raise RuntimeError("Run `uv sync` before the AutoGluon forecast") from error

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    panel = load_sales_panel(args.data_dir / "sales_history.csv")
    feature_groups = parse_feature_groups(args.feature_groups)
    context = load_context_panel(args.processed_data, panel, feature_groups)
    context_values = None if context is None else context.values
    feature_names = (
        FEATURE_NAMES if context is None else FEATURE_NAMES + context.feature_names
    )
    categorical_names = () if context is None else context.categorical_feature_names
    features, targets = make_supervised(
        panel.values, panel.values.shape[0], context_values
    )
    train_data = _frame(features, feature_names, categorical_names)
    train_data["sales_count"] = targets

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    model_path = args.artifacts_dir / run_id
    print(
        f"Training AutoGluon on {len(train_data):,} rows and {len(feature_names)} features"
    )
    predictor = TabularPredictor(
        label="sales_count",
        problem_type="quantile",
        quantile_levels=list(QUANTILES),
        eval_metric="pinball_loss",
        path=str(model_path),
        verbosity=2,
    ).fit(
        train_data=train_data,
        presets=args.presets,
        time_limit=args.time_limit,
        num_cpus=args.threads if args.threads > 0 else "auto",
        num_gpus=0,
    )

    test = pd.read_csv(args.data_dir / "test.csv", usecols=["date"])
    horizon = test["date"].nunique()
    future_context = (
        None
        if context is None
        else context.values[panel.values.shape[0] : panel.values.shape[0] + horizon]
    )
    forecasts = recursive_autogluon_forecast(
        predictor,
        panel.values,
        horizon,
        feature_names,
        categorical_names,
        future_context,
    )
    output_path = args.output or (
        Path(__file__).resolve().parent
        / "predictions"
        / f"{run_id}_autogluon_recursive_{args.feature_groups}.csv"
    )
    write_submission(
        panel,
        forecasts,
        args.data_dir / "test.csv",
        args.data_dir / "sample_submission.csv",
        output_path,
    )
    print(f"Best model: {predictor.model_best}")


if __name__ == "__main__":
    main()
