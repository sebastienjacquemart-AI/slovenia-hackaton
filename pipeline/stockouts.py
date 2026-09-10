"""Past-only heuristic for zeros that look like inventory stockouts."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl


LOOKBACK_DAYS = 28
MIN_POSITIVE_DAYS = 3
MIN_POSITIVE_RATE = 0.50
MIN_TYPICAL_SALES = 5.0
SERIES_KEYS = ["store_id", "product_id"]


def add_stockout_flag(dataset: pl.LazyFrame) -> pl.LazyFrame:
    """Add a training-only flag using observations before the current row."""
    ordered = dataset.sort([*SERIES_KEYS, "date"])
    positive_sales = pl.when(pl.col("sales_count") > 0).then(pl.col("sales_count"))
    ordered = ordered.with_columns(
        positive_sales.shift(1)
        .rolling_median(window_size=LOOKBACK_DAYS, min_samples=1)
        .over(SERIES_KEYS)
        .alias("_prior_positive_median"),
        pl.col("sales_count").gt(0).cast(pl.Int16).shift(1)
        .rolling_sum(window_size=LOOKBACK_DAYS, min_samples=1)
        .over(SERIES_KEYS).alias("_prior_positive_count"),
        pl.col("sales_count").is_not_null().cast(pl.Int16).shift(1)
        .rolling_sum(window_size=LOOKBACK_DAYS, min_samples=1)
        .over(SERIES_KEYS).alias("_prior_observation_count"),
    )
    likely_stockout = (
        pl.col("sales_count").eq(0)
        & (pl.col("_prior_positive_count") >= MIN_POSITIVE_DAYS)
        & (pl.col("_prior_positive_median") >= MIN_TYPICAL_SALES)
        & (pl.col("_prior_positive_count") / pl.col("_prior_observation_count") >= MIN_POSITIVE_RATE)
    ).fill_null(False)
    return ordered.with_columns(
        pl.when(pl.col("dataset_split") == "train").then(likely_stockout)
        .otherwise(pl.lit(None, dtype=pl.Boolean)).alias("is_likely_stockout")
    ).drop("_prior_positive_median", "_prior_positive_count", "_prior_observation_count")


def likely_stockout_for_last_day(history: np.ndarray) -> np.ndarray:
    """Flag each series' last observation using only earlier observations."""
    if history.ndim != 2:
        raise ValueError("sales history must have shape [date, series]")
    if history.shape[0] < 2:
        return np.zeros(history.shape[1], dtype=bool)
    prior = history[max(0, history.shape[0] - 1 - LOOKBACK_DAYS) : -1]
    positive = prior > 0
    positive_count = positive.sum(axis=0)
    observed_count = np.isfinite(prior).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        typical = np.nanmedian(np.where(positive, prior, np.nan), axis=0)
    positive_rate = np.divide(
        positive_count, observed_count,
        out=np.zeros_like(positive_count, dtype=np.float32), where=observed_count > 0,
    )
    return (
        (history[-1] == 0)
        & (positive_count >= MIN_POSITIVE_DAYS)
        & (positive_rate >= MIN_POSITIVE_RATE)
        & (typical >= MIN_TYPICAL_SALES)
    )


def likely_stockout_flags(values: np.ndarray) -> np.ndarray:
    flags = np.zeros_like(values, dtype=bool)
    for index in range(1, values.shape[0]):
        flags[index] = likely_stockout_for_last_day(values[: index + 1])
    return flags
