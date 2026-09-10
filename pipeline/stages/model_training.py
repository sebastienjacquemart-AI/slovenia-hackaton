"""Stage 3 makes a chronological split, fits a model, and reports its score."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.models import (
    QUANTILES,
    AutoGluonConfig,
    LightGBMConfig,
    ModelConfig,
    get_model,
)


def _pinball(actual: pl.Series, predicted: pl.Series, quantile: float) -> float:
    error = actual.log1p() - predicted.clip(lower_bound=0).log1p()
    loss = error.clip(lower_bound=0) * quantile + (-error).clip(lower_bound=0) * (
        1 - quantile
    )
    return float(loss.mean())


def train_and_evaluate(
    features_path: Path,
    report_path: Path,
    model_name: str,
    holdout_days: int,
    seed: int,
    model_config: ModelConfig | None = None,
    log_sales: bool = False,
) -> dict[str, Any]:
    if holdout_days < 1:
        raise ValueError("holdout_days must be at least 1")
    dataset = pl.read_parquet(features_path)
    maximum_date = dataset.get_column("date").max()
    minimum_date = dataset.get_column("date").min()
    if maximum_date is None or minimum_date is None:
        raise ValueError("The feature dataset is empty")
    validation_start = maximum_date - timedelta(days=holdout_days - 1)
    training = dataset.filter(pl.col("date") < validation_start)
    validation = dataset.filter(pl.col("date") >= validation_start)
    if training.is_empty() or validation.is_empty():
        raise ValueError("The chronological split produced an empty partition")

    feature_columns = [
        column for column in dataset.columns if column.startswith("feature_")
    ]
    predictor = get_model(model_name)
    if model_config is None:
        model_config = (
            AutoGluonConfig() if model_name == "autogluon" else LightGBMConfig()
        )
    model_training = training
    if log_sales:
        model_training = training.with_columns(pl.col("sales_count").log1p())
    prediction_rows, model_metadata = predictor(
        model_training, validation, feature_columns, seed, model_config
    )
    prediction_columns = [
        f"prediction_p{int(quantile * 100):02d}" for quantile in QUANTILES
    ]
    predictions = pl.DataFrame(
        prediction_rows,
        schema=prediction_columns,
        orient="row",
    )
    if log_sales:
        predictions = predictions.select(pl.all().exp() - 1).select(
            pl.all().clip(lower_bound=0)
        )
    losses = {
        f"pinball_p{int(quantile * 100):02d}": _pinball(
            validation.get_column("sales_count"),
            predictions.get_column(column),
            quantile,
        )
        for quantile, column in zip(QUANTILES, prediction_columns, strict=True)
    }
    report: dict[str, Any] = {
        "model": model_name,
        "target_transform": "log1p" if log_sales else "identity",
        "feature_count": len(feature_columns),
        "training_rows": training.height,
        "validation_rows": validation.height,
        "training_start": str(training.get_column("date").min()),
        "training_end": str(training.get_column("date").max()),
        "validation_start": str(validation.get_column("date").min()),
        "validation_end": str(validation.get_column("date").max()),
        **losses,
        "mean_pinball": sum(losses.values()) / len(losses),
        "model_metadata": model_metadata,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
