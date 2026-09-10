"""Selectable model implementations for stage 3."""

from __future__ import annotations

import random
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import polars as pl

QUANTILES = (0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class LightGBMConfig:
    num_boost_round: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    bagging_freq: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutoGluonConfig:
    time_limit: int | None = 300
    presets: str = "medium_quality"
    model_types: tuple[str, ...] = ()
    num_cpus: int | str = "auto"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


ModelConfig = LightGBMConfig | AutoGluonConfig


PredictionFunction = Callable[
    [pl.DataFrame, pl.DataFrame, list[str], int, ModelConfig],
    tuple[list[list[float]], dict[str, Any]],
]


def historical_quantile_predictions(
    training: pl.DataFrame,
    validation: pl.DataFrame,
    feature_columns: list[str],
    seed: int,
    config: ModelConfig,
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
    config: ModelConfig,
) -> tuple[list[list[float]], dict[str, Any]]:
    if not isinstance(config, LightGBMConfig):
        raise TypeError("lightgbm requires LightGBMConfig")
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


def autogluon_predictions(
    training: pl.DataFrame,
    validation: pl.DataFrame,
    feature_columns: list[str],
    seed: int,
    config: ModelConfig,
) -> tuple[list[list[float]], dict[str, Any]]:
    if not isinstance(config, AutoGluonConfig):
        raise TypeError("autogluon requires AutoGluonConfig")
    try:
        import numpy as np
        from autogluon.tabular import TabularPredictor
    except ImportError as error:
        raise RuntimeError(
            "The autogluon model needs the root project dependencies. "
            "Run `uv sync` from the challenge directory."
        ) from error

    random.seed(seed)
    np.random.seed(seed)
    label = "sales_count"
    train_data = training.select(*feature_columns, label).to_pandas()
    validation_data = validation.select(feature_columns).to_pandas()
    fit_options: dict[str, Any] = {
        "num_cpus": config.num_cpus,
        "num_gpus": 0,
        "presets": config.presets,
        "time_limit": config.time_limit,
    }
    if config.model_types:
        fit_options["hyperparameters"] = {
            model_type: {} for model_type in config.model_types
        }

    with tempfile.TemporaryDirectory(prefix="autogluon-quantile-") as model_path:
        predictor = TabularPredictor(
            label=label,
            problem_type="quantile",
            quantile_levels=list(QUANTILES),
            eval_metric="pinball_loss",
            path=model_path,
            verbosity=0,
        ).fit(train_data=train_data, **fit_options)
        predicted = predictor.predict(validation_data)
        prediction_columns = [
            next(column for column in predicted.columns if float(column) == quantile)
            for quantile in QUANTILES
        ]
        predictions = np.maximum.accumulate(
            np.maximum(predicted[prediction_columns].to_numpy(), 0.0), axis=1
        )
        model_names = predictor.model_names()
        best_model = predictor.model_best

    return predictions.tolist(), {
        **config.as_dict(),
        "feature_count": len(feature_columns),
        "fit_models": len(model_names),
        "best_model": best_model,
    }


MODELS: dict[str, PredictionFunction] = {
    "autogluon": autogluon_predictions,
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
