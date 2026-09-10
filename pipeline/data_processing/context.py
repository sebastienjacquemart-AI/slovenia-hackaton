from __future__ import annotations

from datetime import date

import polars as pl

from .config import EVENT_BOOLEAN_COLUMNS, EVENT_LIST_COLUMNS
from .validation import scalar


def effective_holiday_expression() -> pl.Expr:
    return (pl.col("event_type") == "Transfer") | (
        (pl.col("event_type") == "Holiday") & ~pl.col("is_transferred")
    )


def effective_event_expression() -> pl.Expr:
    return ~((pl.col("event_type") == "Holiday") & pl.col("is_transferred"))


def add_holiday_distances(
    context: pl.DataFrame,
    expanded_events: pl.DataFrame,
    stores: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    calendar = pl.DataFrame(
        {"date": pl.date_range(start_date, end_date, interval="1d", eager=True)}
    ).join(stores.select("store_id"), how="cross")
    holidays = (
        expanded_events.filter(effective_holiday_expression())
        .select("store_id", pl.col("date").alias("effective_holiday_date"))
        .unique()
    )
    candidates = calendar.join(holidays, on="store_id", how="left")
    future = (
        candidates.filter(pl.col("effective_holiday_date") >= pl.col("date"))
        .group_by("date", "store_id")
        .agg(
            (pl.col("effective_holiday_date") - pl.col("date"))
            .dt.total_days()
            .min()
            .cast(pl.Int16)
            .alias("days_until_effective_holiday")
        )
    )
    past = (
        candidates.filter(pl.col("effective_holiday_date") <= pl.col("date"))
        .group_by("date", "store_id")
        .agg(
            (pl.col("date") - pl.col("effective_holiday_date"))
            .dt.total_days()
            .min()
            .cast(pl.Int16)
            .alias("days_since_effective_holiday")
        )
    )
    return (
        calendar.join(context, on=["date", "store_id"], how="left", validate="1:1")
        .join(future, on=["date", "store_id"], how="left", validate="1:1")
        .join(past, on=["date", "store_id"], how="left", validate="1:1")
        .with_columns(
            pl.col("days_until_effective_holiday").fill_null(-1),
            pl.col("days_since_effective_holiday").fill_null(-1),
        )
        .with_columns(
            (pl.col("days_until_effective_holiday") == 1).alias("is_1d_before_holiday"),
            (pl.col("days_until_effective_holiday") == 2).alias("is_2d_before_holiday"),
        )
    )


def build_event_context(
    events: pl.DataFrame,
    stores: pl.DataFrame,
    start_date: date | None = None,
    end_date: date | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    events_with_id = events.with_row_index("_event_id")
    expanded = events_with_id.join(
        stores.select("store_id", "city", "department"), how="cross"
    ).filter(
        (pl.col("scope") == "National")
        | (
            (pl.col("scope") == "Regional")
            & (pl.col("location") == pl.col("department"))
        )
        | ((pl.col("scope") == "Local") & (pl.col("location") == pl.col("city")))
    )
    unmatched = events_with_id.join(
        expanded.select("_event_id").unique(), on="_event_id", how="anti"
    ).drop("_event_id")
    context = (
        expanded.group_by("date", "store_id")
        .agg(
            pl.len().cast(pl.Int16).alias("event_count"),
            effective_event_expression()
            .cast(pl.Int16)
            .sum()
            .cast(pl.Int16)
            .alias("effective_event_count"),
            pl.col("event_type").unique().sort().alias("event_types"),
            pl.col("scope").unique().sort().alias("event_scopes"),
            pl.col("location").unique().sort().alias("event_locations"),
            (pl.col("event_type") == "Holiday").any().alias("has_holiday"),
            effective_holiday_expression().any().alias("is_effective_holiday"),
            (pl.col("event_type") == "Transfer").any().alias("has_transfer"),
            (pl.col("event_type") == "Additional").any().alias("has_additional"),
            (pl.col("event_type") == "Event").any().alias("has_special_event"),
            (pl.col("scope") == "National").any().alias("has_national_event"),
            (pl.col("scope") == "Regional").any().alias("has_regional_event"),
            (pl.col("scope") == "Local").any().alias("has_local_event"),
            ((pl.col("scope") == "National") & effective_event_expression())
            .any()
            .alias("has_effective_national_event"),
            ((pl.col("scope") == "Regional") & effective_event_expression())
            .any()
            .alias("has_effective_regional_event"),
            ((pl.col("scope") == "Local") & effective_event_expression())
            .any()
            .alias("has_effective_local_event"),
            pl.col("is_transferred").any().alias("has_transferred_event"),
        )
        .with_columns(
            (pl.col("event_count") > 0).alias("is_event_day"),
            (pl.col("effective_event_count") > 0).alias("is_effective_event_day"),
        )
        .sort("date", "store_id")
    )
    if start_date is not None and end_date is not None:
        context = add_holiday_distances(context, expanded, stores, start_date, end_date)
    elif (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be provided together")
    return context, unmatched


def build_daily_oil_context(
    oil: pl.DataFrame, start_date: date, end_date: date
) -> pl.DataFrame:
    summary = oil.select(
        pl.len().alias("rows"),
        pl.col("date").n_unique().alias("unique_dates"),
        pl.col("date").null_count().alias("null_dates"),
        (
            pl.col("oil_price").is_not_null()
            & ((~pl.col("oil_price").is_finite()) | (pl.col("oil_price") <= 0))
        )
        .sum()
        .alias("invalid_prices"),
    )
    if int(scalar(summary, "rows")) != int(scalar(summary, "unique_dates")):
        raise ValueError("index_oil.csv contains duplicate dates")
    if int(scalar(summary, "null_dates")):
        raise ValueError("index_oil.csv contains null dates")
    if int(scalar(summary, "invalid_prices")):
        raise ValueError("index_oil.csv contains invalid non-null prices")
    dates = pl.DataFrame(
        {
            "date": pl.date_range(
                start=start_date, end=end_date, interval="1d", eager=True
            )
        }
    )
    return (
        dates.join(oil, on="date", how="left")
        .sort("date")
        .with_columns(pl.col("oil_price").is_null().alias("oil_price_source_missing"))
        .with_columns(pl.col("oil_price").forward_fill())
    )


def build_daily_traffic_context(
    traffic: pl.DataFrame,
    stores: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Build causal store-traffic features for the complete forecast calendar.

    Same-day traffic is retained for diagnostics, but model features use only
    traffic known before the target date. For dates after the traffic extract,
    the last observed store traffic is carried forward as an as-of baseline;
    ``traffic_source_missing`` makes that uncertainty explicit.
    """
    summary = traffic.select(
        pl.len().alias("rows"),
        pl.struct(["date", "store_id"]).n_unique().alias("unique_keys"),
        pl.col("date").null_count().alias("null_dates"),
        pl.col("store_id").null_count().alias("null_stores"),
        pl.col("transaction_count").null_count().alias("null_counts"),
        (pl.col("transaction_count") < 0)
        .fill_null(True)
        .sum()
        .alias("negative_counts"),
    )
    if int(scalar(summary, "rows")) != int(scalar(summary, "unique_keys")):
        raise ValueError("store_traffic.csv contains duplicate date/store rows")
    for column in ("null_dates", "null_stores", "null_counts", "negative_counts"):
        if int(scalar(summary, column)):
            raise ValueError(f"store_traffic.csv failed {column}")

    calendar = pl.DataFrame(
        {"date": pl.date_range(start_date, end_date, interval="1d", eager=True)}
    ).join(stores.select("store_id").unique(), how="cross")
    complete = (
        calendar.join(traffic, on=["date", "store_id"], how="left", validate="1:1")
        .sort(["store_id", "date"])
        .with_columns(
            pl.col("transaction_count").is_null().alias("traffic_source_missing"),
            pl.col("transaction_count")
            .forward_fill()
            .over("store_id")
            .alias("_traffic_last_observed"),
        )
        .with_columns(
            pl.col("_traffic_last_observed")
            .shift(1)
            .over("store_id")
            .alias("traffic_lag_1"),
            pl.col("_traffic_last_observed")
            .shift(1)
            .rolling_mean(window_size=7, min_samples=1)
            .over("store_id")
            .alias("traffic_mean_7"),
            pl.col("_traffic_last_observed")
            .shift(1)
            .rolling_mean(window_size=28, min_samples=1)
            .over("store_id")
            .alias("traffic_mean_28"),
        )
        .with_columns(
            (pl.col("traffic_mean_7") / (pl.col("traffic_mean_28") + 1.0)).alias(
                "traffic_ratio_7_28"
            )
        )
        .drop("_traffic_last_observed")
    )
    return complete


def add_event_defaults(frame: pl.LazyFrame) -> pl.LazyFrame:
    expressions: list[pl.Expr] = [
        pl.col("event_count").fill_null(0),
        pl.col("effective_event_count").fill_null(0),
        pl.col("days_until_effective_holiday").fill_null(-1),
        pl.col("days_since_effective_holiday").fill_null(-1),
    ]
    expressions.extend(
        pl.col(column).fill_null(False) for column in EVENT_BOOLEAN_COLUMNS
    )
    expressions.extend(
        pl.when(pl.col(column).is_null())
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col(column))
        .alias(column)
        for column in EVENT_LIST_COLUMNS
    )
    return frame.with_columns(expressions)
