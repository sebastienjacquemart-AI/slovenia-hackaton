"""Train LightGBM quantile models from sales history and create a submission."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from .stockouts import likely_stockout_for_last_day


QUANTILES = (0.25, 0.50, 0.75, 0.95)
LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)
FEATURE_NAMES = tuple(
    [f"sales_lag_{lag}" for lag in LAGS]
    + [
        feature
        for window in ROLLING_WINDOWS
        for feature in (f"sales_mean_{window}", f"sales_std_{window}")
    ]
    + ["likely_stockout_lag_1"]
)
MIN_HISTORY = max(max(LAGS), max(ROLLING_WINDOWS))
CONTEXT_FEATURE_GROUPS = {
    "promotion": ("promotion",),
    "product": ("product_family", "product_class", "is_perishable"),
    "store": ("store_id", "city", "department", "store_type", "store_cluster"),
    "calendar": ("year", "month", "day", "day_of_week", "week_of_year", "is_weekend"),
    "event": (
        "event_count",
        "effective_event_count",
        "is_event_day",
        "is_effective_event_day",
        "has_holiday",
        "is_effective_holiday",
        "days_until_effective_holiday",
        "days_since_effective_holiday",
        "has_transfer",
        "has_additional",
        "has_special_event",
        "has_national_event",
        "has_regional_event",
        "has_local_event",
        "has_effective_national_event",
        "has_effective_regional_event",
        "has_effective_local_event",
        "has_transferred_event",
    ),
    "oil": ("oil_price", "oil_price_source_missing"),
    "traffic": ("transaction_count",),
}
CATEGORICAL_CONTEXT_FEATURES = {
    "store_id",
    "product_family",
    "product_class",
    "city",
    "department",
    "store_type",
    "store_cluster",
}


@dataclass(frozen=True)
class SalesPanel:
    dates: pd.DatetimeIndex
    pairs: pd.MultiIndex
    values: np.ndarray


@dataclass(frozen=True)
class ContextPanel:
    dates: pd.DatetimeIndex
    values: np.ndarray
    feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ForecastExplanations:
    raw_predictions: np.ndarray
    shap_values: np.ndarray
    base_values: np.ndarray
    source_quantiles: np.ndarray


def parse_args() -> argparse.Namespace:
    approach_dir = Path(__file__).resolve().parent
    challenge_dir = approach_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a historical-sales LightGBM model on a 150/30 split, "
            "then retrain on all history and create a quantile submission."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=challenge_dir / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Exact output path. The command refuses to overwrite it. By default, "
            "a unique timestamped file is created under predictions/."
        ),
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=approach_dir / "artifacts"
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=challenge_dir / ".cache" / "pipeline" / "stage1_merged.parquet",
    )
    parser.add_argument(
        "--feature-groups",
        default="sales",
        help=(
            "Comma-separated feature groups. Sales features are always enabled. "
            "Optional groups: promotion, product, store, calendar, event, oil, traffic, or all."
        ),
    )
    parser.add_argument("--train-days", type=int, default=150)
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip the 150/30 evaluation and only build the submission.",
    )
    return parser.parse_args()


def parse_feature_groups(raw_groups: str) -> tuple[str, ...]:
    requested = {group.strip().lower() for group in raw_groups.split(",") if group.strip()}
    requested.discard("sales")
    if "all" in requested:
        requested.remove("all")
        requested.update(CONTEXT_FEATURE_GROUPS)
    unknown = requested.difference(CONTEXT_FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    return tuple(group for group in CONTEXT_FEATURE_GROUPS if group in requested)


def unique_prediction_path(approach_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = uuid4().hex[:8]
    return approach_dir / "predictions" / f"{timestamp}_{run_name}_{run_id}.csv"


def load_sales_panel(path: Path) -> SalesPanel:
    frame = pd.read_csv(
        path,
        usecols=["date", "store_id", "product_id", "sales_count"],
        parse_dates=["date"],
        dtype={"store_id": "int32", "product_id": "int32", "sales_count": "float32"},
    )
    duplicate_count = int(frame.duplicated(["date", "store_id", "product_id"]).sum())
    if duplicate_count:
        raise ValueError(f"sales history has {duplicate_count} duplicate date/pair rows")

    wide = frame.pivot(index="date", columns=["store_id", "product_id"], values="sales_count")
    wide = wide.sort_index().sort_index(axis=1)
    if wide.isna().any().any():
        raise ValueError("sales history is not a complete date by store/product panel")
    if (wide.to_numpy() < 0).any():
        raise ValueError("sales history contains negative values")
    return SalesPanel(
        dates=pd.DatetimeIndex(wide.index),
        pairs=pd.MultiIndex.from_tuples(wide.columns.tolist(), names=["store_id", "product_id"]),
        values=wide.to_numpy(dtype=np.float32, copy=True),
    )


def load_context_panel(
    path: Path, sales_panel: SalesPanel, feature_groups: tuple[str, ...]
) -> ContextPanel | None:
    if not feature_groups:
        return None
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run `uv run python -m pipeline data` first"
        )
    feature_names = tuple(
        feature
        for group in feature_groups
        for feature in CONTEXT_FEATURE_GROUPS[group]
    )
    categorical_names = tuple(
        feature for feature in feature_names if feature in CATEGORICAL_CONTEXT_FEATURES
    )
    source = pl.scan_parquet(path)
    expressions: list[pl.Expr] = []
    for feature in feature_names:
        expression = pl.col(feature)
        if feature in CATEGORICAL_CONTEXT_FEATURES:
            categories = (
                source.select(pl.col(feature).cast(pl.String).unique().sort())
                .collect()[feature]
                .to_list()
            )
            expression = (
                expression.cast(pl.String).cast(pl.Enum(categories)).to_physical()
            )
        elif feature in {"oil_price", "transaction_count"}:
            expression = expression.fill_null(0)
        expressions.append(expression.cast(pl.Float32).alias(feature))
    frame = (
        source
        .sort(["date", "store_id", "product_id"])
        .select(
            "date",
            pl.col("store_id").alias("_store_key"),
            pl.col("product_id").alias("_product_key"),
            *expressions,
        )
        .collect()
    )
    if frame.select(feature_names).null_count().row(0) != (0,) * len(feature_names):
        raise ValueError("processed context contains null feature values")
    dates = pd.DatetimeIndex(frame["date"].unique(maintain_order=True).to_list())
    series_count = len(sales_panel.pairs)
    if frame.height != len(dates) * series_count:
        raise ValueError("processed context is not a complete date by store/product panel")
    first_date_pairs = frame.filter(pl.col("date") == frame["date"][0]).select(
        "_store_key", "_product_key"
    ).to_numpy()
    expected_pairs = np.column_stack(
        [
            sales_panel.pairs.get_level_values("store_id"),
            sales_panel.pairs.get_level_values("product_id"),
        ]
    )
    if not np.array_equal(first_date_pairs, expected_pairs):
        raise ValueError("processed context store/product pairs do not align with sales history")
    if not dates[: len(sales_panel.dates)].equals(sales_panel.dates):
        raise ValueError("processed context dates do not begin with the sales history dates")
    values = frame.select(feature_names).to_numpy().reshape(
        len(dates), series_count, len(feature_names)
    )
    return ContextPanel(
        dates=dates,
        values=values.astype(np.float32, copy=False),
        feature_names=feature_names,
        categorical_feature_names=categorical_names,
    )


def features_for_next(
    history: np.ndarray, context_row: np.ndarray | None = None
) -> np.ndarray:
    """Build one feature row per series using values before the target date."""
    if history.shape[0] < MIN_HISTORY:
        raise ValueError(f"at least {MIN_HISTORY} history days are required")
    columns = [history[-lag] for lag in LAGS]
    for window in ROLLING_WINDOWS:
        recent = history[-window:]
        columns.extend((recent.mean(axis=0), recent.std(axis=0)))
    columns.append(likely_stockout_for_last_day(history).astype(np.float32))
    sales_features = np.column_stack(columns).astype(np.float32, copy=False)
    if context_row is None:
        return sales_features
    if context_row.shape[0] != history.shape[1]:
        raise ValueError("context row does not match the number of sales series")
    return np.column_stack((sales_features, context_row)).astype(np.float32, copy=False)


def make_supervised(
    values: np.ndarray,
    target_end: int,
    context_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create features for target dates [MIN_HISTORY, target_end)."""
    if not MIN_HISTORY < target_end <= values.shape[0]:
        raise ValueError(
            f"target_end must be between {MIN_HISTORY + 1} and {values.shape[0]}"
        )
    feature_blocks = []
    target_blocks = []
    for target_index in range(MIN_HISTORY, target_end):
        context_row = None if context_values is None else context_values[target_index]
        feature_blocks.append(features_for_next(values[:target_index], context_row))
        target_blocks.append(values[target_index])
    return np.vstack(feature_blocks), np.concatenate(target_blocks)


def train_models(
    features: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
    model_dir: Path,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    categorical_feature_names: tuple[str, ...] = (),
) -> dict[float, lgb.Booster]:
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset = lgb.Dataset(
        features,
        label=targets,
        feature_name=list(feature_names),
        categorical_feature=list(categorical_feature_names),
        free_raw_data=False,
    )
    models: dict[float, lgb.Booster] = {}
    for quantile in QUANTILES:
        print(f"Training q={quantile:.2f} on {len(targets):,} rows")
        params = {
            "objective": "quantile",
            "alpha": quantile,
            "metric": "quantile",
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "seed": args.seed,
            "feature_fraction_seed": args.seed,
            "bagging_seed": args.seed,
            "num_threads": args.threads,
            "verbosity": -1,
        }
        model = lgb.train(params, dataset, num_boost_round=args.num_boost_round)
        model.save_model(model_dir / f"q{int(quantile * 100):02d}.txt")
        models[quantile] = model
    return models


def _recursive_forecast(
    models: dict[float, lgb.Booster],
    initial_history: np.ndarray,
    horizon: int,
    future_context: np.ndarray | None = None,
    include_shap: bool = False,
) -> tuple[np.ndarray, ForecastExplanations | None]:
    """Forecast recursively, feeding the median prediction into every next-day feature."""
    history = initial_history.copy()
    forecasts = []
    raw_forecasts = []
    shap_steps = []
    base_steps = []
    source_quantile_steps = []
    quantile_values = np.asarray(QUANTILES, dtype=np.float32)
    for step in range(horizon):
        context_row = None if future_context is None else future_context[step]
        features = features_for_next(history, context_row)
        raw = np.column_stack([models[q].predict(features) for q in QUANTILES])
        clipped = np.maximum(raw, 0.0)
        order = np.argsort(clipped, axis=1, stable=True)
        coherent = np.take_along_axis(clipped, order, axis=1).astype(np.float32)
        forecasts.append(coherent)
        if include_shap:
            contributions = np.stack(
                [models[q].predict(features, pred_contrib=True) for q in QUANTILES],
                axis=1,
            ).astype(np.float32)
            ordered_contributions = np.take_along_axis(
                contributions, order[:, :, np.newaxis], axis=1
            )
            raw_forecasts.append(
                np.take_along_axis(raw, order, axis=1).astype(np.float32)
            )
            shap_steps.append(ordered_contributions[:, :, :-1])
            base_steps.append(ordered_contributions[:, :, -1])
            source_quantile_steps.append(quantile_values[order])
        history = np.vstack((history, coherent[:, 1]))
        print(f"Forecast day {step + 1}/{horizon}", end="\r", flush=True)
    print()
    forecast_array = np.stack(forecasts)
    if not include_shap:
        return forecast_array, None
    return forecast_array, ForecastExplanations(
        raw_predictions=np.stack(raw_forecasts),
        shap_values=np.stack(shap_steps),
        base_values=np.stack(base_steps),
        source_quantiles=np.stack(source_quantile_steps),
    )


def recursive_forecast(
    models: dict[float, lgb.Booster],
    initial_history: np.ndarray,
    horizon: int,
    future_context: np.ndarray | None = None,
) -> np.ndarray:
    forecasts, _ = _recursive_forecast(
        models, initial_history, horizon, future_context, include_shap=False
    )
    return forecasts


def recursive_forecast_with_shap(
    models: dict[float, lgb.Booster],
    initial_history: np.ndarray,
    horizon: int,
    future_context: np.ndarray | None = None,
) -> tuple[np.ndarray, ForecastExplanations]:
    forecasts, explanations = _recursive_forecast(
        models, initial_history, horizon, future_context, include_shap=True
    )
    if explanations is None:
        raise RuntimeError("SHAP explanations were not generated")
    return forecasts, explanations


def pinball_loss(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> float:
    error = np.log1p(actual) - np.log1p(np.maximum(forecast, 0.0))
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def evaluate_holdout(
    panel: SalesPanel,
    args: argparse.Namespace,
    artifacts_dir: Path,
    context: ContextPanel | None = None,
) -> dict[str, float | str]:
    split_end = args.train_days + args.holdout_days
    if split_end > panel.values.shape[0]:
        raise ValueError(
            f"requested {split_end} split days but history has {panel.values.shape[0]}"
        )
    context_values = None if context is None else context.values
    feature_names = FEATURE_NAMES if context is None else FEATURE_NAMES + context.feature_names
    categorical_names = () if context is None else context.categorical_feature_names
    features, targets = make_supervised(panel.values, args.train_days, context_values)
    models = train_models(
        features,
        targets,
        args,
        artifacts_dir / "holdout_models",
        feature_names,
        categorical_names,
    )
    holdout_context = (
        None
        if context is None
        else context.values[args.train_days : args.train_days + args.holdout_days]
    )
    forecast = recursive_forecast(
        models,
        panel.values[: args.train_days],
        args.holdout_days,
        holdout_context,
    )
    actual = panel.values[args.train_days:split_end]
    metrics = {
        f"pinball_p{int(quantile * 100):02d}": pinball_loss(
            actual, forecast[:, :, index], quantile
        )
        for index, quantile in enumerate(QUANTILES)
    }
    metrics["mean_pinball"] = float(np.mean(list(metrics.values())))
    metrics["train_start"] = panel.dates[0].date().isoformat()
    metrics["train_end"] = panel.dates[args.train_days - 1].date().isoformat()
    metrics["holdout_start"] = panel.dates[args.train_days].date().isoformat()
    metrics["holdout_end"] = panel.dates[split_end - 1].date().isoformat()
    metrics_path = artifacts_dir / "holdout_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def write_submission(
    panel: SalesPanel,
    forecasts: np.ndarray,
    test_path: Path,
    sample_path: Path,
    output_path: Path,
) -> None:
    test = pd.read_csv(
        test_path,
        parse_dates=["date"],
        dtype={"store_id": "int32", "product_id": "int32"},
    )
    sample = pd.read_csv(sample_path)
    expected_columns = ["id"] + [f"sales_count_p{int(q * 100):02d}" for q in QUANTILES]
    if list(sample.columns) != expected_columns:
        raise ValueError(f"unexpected sample submission columns: {list(sample.columns)}")
    if len(test) != len(sample) or not np.array_equal(test["id"], sample["id"]):
        raise ValueError("test IDs do not exactly match the sample submission")

    forecast_dates = pd.DatetimeIndex(sorted(test["date"].unique()))
    if len(forecast_dates) != forecasts.shape[0]:
        raise ValueError("test date count does not match forecast horizon")
    date_positions = forecast_dates.get_indexer(test["date"])
    test_pairs = pd.MultiIndex.from_frame(test[["store_id", "product_id"]])
    pair_positions = panel.pairs.get_indexer(test_pairs)
    if (date_positions < 0).any() or (pair_positions < 0).any():
        raise ValueError("test contains dates or store/product pairs missing from the forecast")

    ordered_forecasts = forecasts[date_positions, pair_positions]
    submission = pd.DataFrame({"id": sample["id"]})
    for index, column in enumerate(expected_columns[1:]):
        submission[column] = ordered_forecasts[:, index]
    if submission.isna().any().any():
        raise ValueError("submission contains missing values")
    if not np.all(np.diff(submission[expected_columns[1:]].to_numpy(), axis=1) >= 0):
        raise ValueError("submission quantiles cross")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing prediction file: {output_path}")
    submission.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Wrote {len(submission):,} rows to {output_path}")


def write_shap_values(
    panel: SalesPanel,
    forecasts: np.ndarray,
    explanations: ForecastExplanations,
    feature_names: tuple[str, ...],
    test_path: Path,
    output_path: Path,
) -> None:
    """Write one row per submitted ID and quantile to a Parquet sidecar."""
    test = pd.read_csv(
        test_path,
        usecols=["id", "date", "store_id", "product_id"],
        parse_dates=["date"],
        dtype={"store_id": "int32", "product_id": "int32"},
    )
    forecast_dates = pd.DatetimeIndex(sorted(test["date"].unique()))
    date_positions = forecast_dates.get_indexer(test["date"])
    test_pairs = pd.MultiIndex.from_frame(test[["store_id", "product_id"]])
    pair_positions = panel.pairs.get_indexer(test_pairs)
    if (date_positions < 0).any() or (pair_positions < 0).any():
        raise ValueError("test contains rows missing from the SHAP forecast")

    ordered_forecasts = forecasts[date_positions, pair_positions]
    ordered_raw = explanations.raw_predictions[date_positions, pair_positions]
    ordered_shap = explanations.shap_values[date_positions, pair_positions]
    ordered_base = explanations.base_values[date_positions, pair_positions]
    ordered_sources = explanations.source_quantiles[date_positions, pair_positions]
    if ordered_shap.shape[-1] != len(feature_names):
        raise ValueError("SHAP feature count does not match model feature names")
    reconstructed = ordered_base + ordered_shap.sum(axis=-1)
    if not np.allclose(reconstructed, ordered_raw, rtol=1e-4, atol=1e-4):
        raise ValueError("SHAP values do not reconstruct the raw model predictions")

    row_count = len(test)
    columns: dict[str, object] = {
        "id": np.repeat(test["id"].to_numpy(), len(QUANTILES)),
        "submitted_quantile": np.tile(
            np.asarray(QUANTILES, dtype=np.float32), row_count
        ),
        "source_model_quantile": ordered_sources.reshape(-1),
        "raw_prediction": ordered_raw.reshape(-1),
        "submitted_prediction": ordered_forecasts.reshape(-1),
        "postprocessing_adjustment": (
            ordered_forecasts - ordered_raw
        ).reshape(-1),
        "base_value": ordered_base.reshape(-1),
    }
    for index, feature_name in enumerate(feature_names):
        columns[f"shap_{feature_name}"] = ordered_shap[:, :, index].reshape(-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing SHAP file: {output_path}")
    pl.DataFrame(columns).write_parquet(output_path, compression="zstd")
    print(f"Wrote {row_count * len(QUANTILES):,} SHAP rows to {output_path}")


def main() -> None:
    args = parse_args()
    panel = load_sales_panel(args.data_dir / "sales_history.csv")
    feature_groups = parse_feature_groups(args.feature_groups)
    context = load_context_panel(args.processed_data, panel, feature_groups)
    print(
        f"Loaded {panel.values.shape[0]} days and {panel.values.shape[1]:,} "
        "store/product series"
    )
    enabled_features = ["sales", *feature_groups]
    print(f"Feature groups: {', '.join(enabled_features)}")
    run_name = "lgbm_" + "_".join(enabled_features)
    output_path = args.output or unique_prediction_path(
        Path(__file__).resolve().parent, run_name
    )
    run_artifacts_dir = (
        args.artifacts_dir
        if not feature_groups
        else args.artifacts_dir / "_".join(enabled_features)
    )
    if not args.skip_holdout:
        evaluate_holdout(panel, args, run_artifacts_dir, context)

    context_values = None if context is None else context.values
    feature_names = FEATURE_NAMES if context is None else FEATURE_NAMES + context.feature_names
    categorical_names = () if context is None else context.categorical_feature_names
    features, targets = make_supervised(
        panel.values, panel.values.shape[0], context_values
    )
    models = train_models(
        features,
        targets,
        args,
        run_artifacts_dir / "final_models",
        feature_names,
        categorical_names,
    )
    test = pd.read_csv(args.data_dir / "test.csv", usecols=["date"], parse_dates=["date"])
    horizon = test["date"].nunique()
    future_context = (
        None
        if context is None
        else context.values[panel.values.shape[0] : panel.values.shape[0] + horizon]
    )
    if future_context is not None and future_context.shape[0] != horizon:
        raise ValueError("processed context does not cover the full forecast horizon")
    forecasts, explanations = recursive_forecast_with_shap(
        models, panel.values, horizon, future_context
    )
    write_submission(
        panel,
        forecasts,
        args.data_dir / "test.csv",
        args.data_dir / "sample_submission.csv",
        output_path,
    )
    write_shap_values(
        panel,
        forecasts,
        explanations,
        feature_names,
        args.data_dir / "test.csv",
        output_path.with_suffix(".shap.parquet"),
    )


if __name__ == "__main__":
    main()
