"""Selectable model implementations for stage 3."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

QUANTILES = (0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class LightGBMConfig:
    num_boost_round: int = 250
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PredictionFunction = Callable[
    [pl.DataFrame, pl.DataFrame, list[str], int, LightGBMConfig],
    tuple[list[list[float]], dict[str, Any]],
]


def historical_quantile_predictions(
    training: pl.DataFrame,
    validation: pl.DataFrame,
    feature_columns: list[str],
    seed: int,
    config: LightGBMConfig,
) -> tuple[list[list[float]], dict[str, Any]]:
    del feature_columns, seed, config
    prediction_names = [
        f"prediction_p{int(quantile * 100):02d}" for quantile in QUANTILES
    ]
    per_series = training.group_by("store_id", "product_id").agg(
        *[
            pl.col("sales_count").quantile(quantile, interpolation="linear").alias(name)
            for quantile, name in zip(QUANTILES, prediction_names, strict=True)
        ]
    )
    global_values = {
        name: float(training.get_column("sales_count").quantile(quantile))
        for quantile, name in zip(QUANTILES, prediction_names, strict=True)
    }
    predicted = validation.join(
        per_series, on=["store_id", "product_id"], how="left", validate="m:1"
    ).with_columns(
        *[pl.col(name).fill_null(value) for name, value in global_values.items()]
    )
    return predicted.select(prediction_names).to_numpy().tolist(), {
        "fit_series": per_series.height
    }


def lightgbm_predictions(
    training: pl.DataFrame,
    validation: pl.DataFrame,
    feature_columns: list[str],
    seed: int,
    config: LightGBMConfig,
) -> tuple[list[list[float]], dict[str, Any]]:
    try:
        import numpy as np
        from lightgbm import Dataset, train
    except ImportError as error:
        raise RuntimeError(
            "The lightgbm model needs the root project dependencies. "
            "Run `uv sync` from the challenge directory."
        ) from error

    train_features = training.select(feature_columns).to_numpy()
    train_targets = training.get_column("sales_count").to_numpy()
    validation_features = validation.select(feature_columns).to_numpy()
    columns = []
    for quantile in QUANTILES:
        model = train(
            {
                "objective": "quantile",
                "alpha": quantile,
                "learning_rate": config.learning_rate,
                "num_leaves": config.num_leaves,
                "min_data_in_leaf": config.min_data_in_leaf,
                "feature_fraction": config.feature_fraction,
                "bagging_fraction": config.bagging_fraction,
                "bagging_freq": config.bagging_freq,
                "seed": seed,
                "verbosity": -1,
            },
            Dataset(train_features, label=train_targets),
            num_boost_round=config.num_boost_round,
        )
        columns.append(np.maximum(model.predict(validation_features), 0.0))
    predictions = np.maximum.accumulate(np.column_stack(columns), axis=1)
    return predictions.tolist(), {
        **config.as_dict(),
        "feature_count": len(feature_columns),
    }


MODELS: dict[str, PredictionFunction] = {
    "historical_quantile": historical_quantile_predictions,
    "lightgbm": lightgbm_predictions,
}


def get_model(name: str) -> PredictionFunction:
    try:
        return MODELS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {name!r}. Choose from: {', '.join(MODELS)}"
        ) from error
