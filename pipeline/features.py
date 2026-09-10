"""Selectable feature groups for stage 2."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl


FeatureBuilder = Callable[[], list[pl.Expr]]
SERIES_KEYS = ["store_id", "product_id"]


def _encoded(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .cast(pl.Categorical)
        .to_physical()
        .cast(pl.Int32)
        .alias(f"feature_{column}")
    )


def _identity() -> list[pl.Expr]:
    return [
        pl.col("store_id").cast(pl.Int16).alias("feature_store_id"),
        pl.col("product_id").cast(pl.Int32).alias("feature_product_id"),
    ]


def _promotion() -> list[pl.Expr]:
    return [pl.col("promotion").cast(pl.Int8).alias("feature_promotion")]


def _calendar() -> list[pl.Expr]:
    return [
        pl.col(column).cast(pl.Int16).alias(f"feature_{column}")
        for column in ("year", "month", "day", "day_of_week", "week_of_year")
    ] + [pl.col("is_weekend").cast(pl.Int8).alias("feature_is_weekend")]


def _product() -> list[pl.Expr]:
    return [
        _encoded("product_family"),
        pl.col("product_class").cast(pl.Int32).alias("feature_product_class"),
        pl.col("is_perishable").cast(pl.Int8).alias("feature_is_perishable"),
    ]


def _store() -> list[pl.Expr]:
    return [
        _encoded("city"),
        _encoded("department"),
        _encoded("store_type"),
        pl.col("store_cluster").cast(pl.Int16).alias("feature_store_cluster"),
    ]


def _external() -> list[pl.Expr]:
    numeric = (
        "oil_price",
        "transaction_count",
        "event_count",
        "effective_event_count",
        "days_until_effective_holiday",
        "days_since_effective_holiday",
    )
    boolean = (
        "oil_price_source_missing",
        "is_event_day",
        "is_effective_event_day",
        "has_holiday",
        "is_effective_holiday",
        "is_1d_before_holiday",
        "is_2d_before_holiday",
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
    )
    return [
        pl.col(column).cast(pl.Float32).alias(f"feature_{column}")
        for column in numeric
    ] + [
        pl.col(column).cast(pl.Int8).alias(f"feature_{column}")
        for column in boolean
    ]


def _sales_history() -> list[pl.Expr]:
    shifted = pl.col("sales_count").shift(1)
    expressions = [
        pl.col("sales_count")
        .shift(lag)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias(f"feature_sales_lag_{lag}")
        for lag in (1, 7, 14, 28)
    ]
    for window in (7, 28):
        expressions.extend(
            [
                shifted.rolling_mean(window_size=window, min_samples=1)
                .over(SERIES_KEYS)
                .cast(pl.Float32)
                .alias(f"feature_sales_mean_{window}"),
                shifted.rolling_std(window_size=window, min_samples=2)
                .over(SERIES_KEYS)
                .cast(pl.Float32)
                .alias(f"feature_sales_std_{window}"),
            ]
        )
    expressions.append(
        pl.col("is_likely_stockout")
        .shift(1)
        .over(SERIES_KEYS)
        .fill_null(False)
        .cast(pl.Int8)
        .alias("feature_likely_stockout_lag_1")
    )
    return expressions


FEATURE_GROUPS: dict[str, FeatureBuilder] = {
    "identity": _identity,
    "promotion": _promotion,
    "calendar": _calendar,
    "product": _product,
    "store": _store,
    "external": _external,
    "sales_history": _sales_history,
}


def parse_feature_groups(raw_groups: str) -> tuple[str, ...]:
    requested = tuple(
        dict.fromkeys(group.strip() for group in raw_groups.split(",") if group.strip())
    )
    if requested == ("all",):
        return tuple(FEATURE_GROUPS)
    unknown = sorted(set(requested) - FEATURE_GROUPS.keys())
    if unknown:
        raise ValueError(
            f"Unknown feature groups: {', '.join(unknown)}. "
            f"Choose from: {', '.join(FEATURE_GROUPS)}"
        )
    if not requested:
        raise ValueError("Select at least one feature group")
    return requested


def feature_expressions(groups: tuple[str, ...]) -> list[pl.Expr]:
    return [expression for group in groups for expression in FEATURE_GROUPS[group]()]
