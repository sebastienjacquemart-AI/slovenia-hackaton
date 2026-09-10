"""Selectable feature groups for stage 2."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

FeatureBuilder = Callable[[], list[pl.Expr]]
SERIES_KEYS = ["store_id", "product_id"]
BASE_FEATURE_GROUPS = (
    "promotion",
    "calendar",
    "product",
    "store",
    "external",
    "sales_history",
)
# Updated after chronological ablation runs. Keep raw identifiers out of this preset.
BEST_FEATURE_GROUPS = BASE_FEATURE_GROUPS


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
        pl.col(column).cast(pl.Float32).alias(f"feature_{column}") for column in numeric
    ] + [pl.col(column).cast(pl.Int8).alias(f"feature_{column}") for column in boolean]


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


def _promotion_regime() -> list[pl.Expr]:
    promotion = pl.col("promotion").cast(pl.Int8)
    lagged = {
        lag: promotion.shift(lag).over(SERIES_KEYS).fill_null(0) for lag in (1, 2, 3, 7)
    }
    shifted = promotion.shift(1)
    return [
        *[
            value.cast(pl.Int8).alias(f"feature_promotion_lag_{lag}")
            for lag, value in lagged.items()
        ],
        shifted.rolling_sum(window_size=7, min_samples=1)
        .over(SERIES_KEYS)
        .cast(pl.Int8)
        .alias("feature_promotion_count_7"),
        shifted.rolling_sum(window_size=28, min_samples=1)
        .over(SERIES_KEYS)
        .cast(pl.Int8)
        .alias("feature_promotion_count_28"),
        ((promotion == 1) & (lagged[1] == 0))
        .cast(pl.Int8)
        .alias("feature_promotion_start"),
        ((promotion == 0) & (lagged[1] == 1))
        .cast(pl.Int8)
        .alias("feature_post_promotion_1d"),
        ((promotion == 0) & (lagged[1] == 0) & (lagged[2] == 1))
        .cast(pl.Int8)
        .alias("feature_post_promotion_2d"),
        ((promotion == 0) & (lagged[1] == 0) & (lagged[2] == 0) & (lagged[3] == 1))
        .cast(pl.Int8)
        .alias("feature_post_promotion_3d"),
        (pl.col("promotion") & pl.col("is_perishable"))
        .cast(pl.Int8)
        .alias("feature_promotion_perishable"),
        (pl.col("promotion") & pl.col("is_effective_holiday"))
        .cast(pl.Int8)
        .alias("feature_promotion_effective_holiday"),
    ]


def _seasonal_history() -> list[pl.Expr]:
    lagged = {
        lag: pl.col("sales_count").shift(lag).over(SERIES_KEYS)
        for lag in (2, 3, 4, 5, 6, 8, 21, 35, 42, 56)
    }
    shifted = pl.col("sales_count").shift(1)
    mean_7 = shifted.rolling_mean(window_size=7, min_samples=1).over(SERIES_KEYS)
    mean_28 = shifted.rolling_mean(window_size=28, min_samples=1).over(SERIES_KEYS)
    return [
        *[
            value.cast(pl.Float32).alias(f"feature_sales_lag_{lag}")
            for lag, value in lagged.items()
        ],
        pl.mean_horizontal(
            [
                pl.col("sales_count").shift(lag).over(SERIES_KEYS)
                for lag in (7, 14, 21, 28)
            ]
        )
        .cast(pl.Float32)
        .alias("feature_same_weekday_mean_4"),
        shifted.rolling_median(window_size=7, min_samples=1)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_sales_median_7"),
        shifted.rolling_median(window_size=28, min_samples=1)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_sales_median_28"),
        shifted.rolling_max(window_size=28, min_samples=1)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_sales_max_28"),
        (mean_7 - mean_28).cast(pl.Float32).alias("feature_sales_trend_7_28"),
        (mean_7 / (mean_28 + 1.0)).cast(pl.Float32).alias("feature_sales_ratio_7_28"),
    ]


def _intermittency() -> list[pl.Expr]:
    shifted_sales = pl.col("sales_count").shift(1)
    shifted_zero = (pl.col("sales_count") == 0).cast(pl.Float32).shift(1)
    shifted_stockout = pl.col("is_likely_stockout").cast(pl.Float32).shift(1)
    shifted_nonzero = (
        pl.when(pl.col("sales_count") > 0)
        .then(pl.col("sales_count"))
        .otherwise(None)
        .shift(1)
    )
    lag_1 = pl.col("sales_count").shift(1).over(SERIES_KEYS)
    lag_2 = pl.col("sales_count").shift(2).over(SERIES_KEYS)
    lag_3 = pl.col("sales_count").shift(3).over(SERIES_KEYS)
    return [
        *[
            shifted_zero.rolling_mean(window_size=window, min_samples=1)
            .over(SERIES_KEYS)
            .cast(pl.Float32)
            .alias(f"feature_zero_rate_{window}")
            for window in (7, 28, 56)
        ],
        *[
            shifted_nonzero.rolling_mean(window_size=window, min_samples=1)
            .over(SERIES_KEYS)
            .cast(pl.Float32)
            .alias(f"feature_nonzero_mean_{window}")
            for window in (28, 56)
        ],
        ((lag_1 == 0) & (lag_2 == 0) & (lag_3 == 0))
        .cast(pl.Int8)
        .alias("feature_zero_run_3"),
        shifted_stockout.rolling_mean(window_size=7, min_samples=1)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_stockout_rate_7"),
        shifted_stockout.rolling_mean(window_size=28, min_samples=1)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_stockout_rate_28"),
        (
            shifted_sales.rolling_std(window_size=28, min_samples=2).over(SERIES_KEYS)
            / (
                shifted_sales.rolling_mean(window_size=28, min_samples=1).over(
                    SERIES_KEYS
                )
                + 1.0
            )
        )
        .cast(pl.Float32)
        .alias("feature_sales_cv_28"),
    ]


def _payday() -> list[pl.Expr]:
    day = pl.col("day")
    return [
        day.is_in([3, 18]).cast(pl.Int8).alias("feature_is_payday"),
        day.is_in([2, 17]).cast(pl.Int8).alias("feature_is_1d_before_payday"),
        day.is_in([1, 16]).cast(pl.Int8).alias("feature_is_2d_before_payday"),
        day.is_in([4, 19]).cast(pl.Int8).alias("feature_is_1d_after_payday"),
        day.is_in([5, 20]).cast(pl.Int8).alias("feature_is_2d_after_payday"),
        (pl.col("promotion") & day.is_in([3, 18]))
        .cast(pl.Int8)
        .alias("feature_promotion_payday"),
    ]


def _holiday_regime() -> list[pl.Expr]:
    before = pl.col("is_1d_before_holiday") | pl.col("is_2d_before_holiday")
    holiday_window = pl.col("is_effective_holiday") | before
    return [
        holiday_window.cast(pl.Int8).alias("feature_effective_holiday_window"),
        (pl.col("promotion") & before)
        .cast(pl.Int8)
        .alias("feature_promotion_before_holiday"),
        (pl.col("is_perishable") & holiday_window)
        .cast(pl.Int8)
        .alias("feature_perishable_holiday_window"),
        (pl.col("has_effective_local_event") & pl.col("promotion"))
        .cast(pl.Int8)
        .alias("feature_local_event_promotion"),
    ]


def _behavioral_profile() -> list[pl.Expr]:
    shifted = pl.col("sales_count").shift(1)
    promoted_sales = (
        pl.when(pl.col("promotion"))
        .then(pl.col("sales_count"))
        .otherwise(None)
        .shift(1)
    )
    regular_sales = (
        pl.when(~pl.col("promotion"))
        .then(pl.col("sales_count"))
        .otherwise(None)
        .shift(1)
    )
    promoted_mean = promoted_sales.rolling_mean(window_size=56, min_samples=1).over(
        SERIES_KEYS
    )
    regular_mean = regular_sales.rolling_mean(window_size=56, min_samples=1).over(
        SERIES_KEYS
    )
    return [
        shifted.rolling_quantile(0.25, window_size=56, min_samples=7)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_profile_p25_56"),
        shifted.rolling_median(window_size=56, min_samples=7)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_profile_p50_56"),
        shifted.rolling_quantile(0.75, window_size=56, min_samples=7)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_profile_p75_56"),
        shifted.rolling_quantile(0.95, window_size=56, min_samples=7)
        .over(SERIES_KEYS)
        .cast(pl.Float32)
        .alias("feature_profile_p95_56"),
        promoted_mean.cast(pl.Float32).alias("feature_promoted_mean_56"),
        regular_mean.cast(pl.Float32).alias("feature_regular_mean_56"),
        (promoted_mean - regular_mean)
        .cast(pl.Float32)
        .alias("feature_promotion_uplift_56"),
    ]


def _oil_trend() -> list[pl.Expr]:
    oil = pl.col("oil_price")
    return [
        *[
            oil.shift(lag)
            .over(SERIES_KEYS)
            .cast(pl.Float32)
            .alias(f"feature_oil_lag_{lag}")
            for lag in (30, 60, 90)
        ],
        *[
            oil.shift(1)
            .rolling_mean(window_size=window, min_samples=7)
            .over(SERIES_KEYS)
            .cast(pl.Float32)
            .alias(f"feature_oil_mean_{window}")
            for window in (30, 60, 90)
        ],
        (oil.shift(30).over(SERIES_KEYS) - oil.shift(90).over(SERIES_KEYS))
        .cast(pl.Float32)
        .alias("feature_oil_change_30_90"),
    ]


FEATURE_GROUPS: dict[str, FeatureBuilder] = {
    "identity": _identity,
    "promotion": _promotion,
    "calendar": _calendar,
    "product": _product,
    "store": _store,
    "external": _external,
    "sales_history": _sales_history,
    "promotion_regime": _promotion_regime,
    "seasonal_history": _seasonal_history,
    "intermittency": _intermittency,
    "payday": _payday,
    "holiday_regime": _holiday_regime,
    "behavioral_profile": _behavioral_profile,
    "oil_trend": _oil_trend,
}


def parse_feature_groups(raw_groups: str) -> tuple[str, ...]:
    requested = tuple(
        dict.fromkeys(group.strip() for group in raw_groups.split(",") if group.strip())
    )
    if requested == ("best",):
        return BEST_FEATURE_GROUPS
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
