from __future__ import annotations

from datetime import date
import polars as pl
from .config import EVENT_BOOLEAN_COLUMNS, EVENT_LIST_COLUMNS
from .validation import scalar

def build_event_context(events: pl.DataFrame, stores: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    events_with_id = events.with_row_index("_event_id")
    expanded = events_with_id.join(stores.select("store_id", "city", "department"), how="cross").filter((pl.col("scope") == "National") | ((pl.col("scope") == "Regional") & (pl.col("location") == pl.col("department"))) | ((pl.col("scope") == "Local") & (pl.col("location") == pl.col("city"))))
    unmatched = events_with_id.join(expanded.select("_event_id").unique(), on="_event_id", how="anti").drop("_event_id")
    context = expanded.group_by("date", "store_id").agg(pl.len().cast(pl.Int16).alias("event_count"), pl.col("event_type").unique().sort().alias("event_types"), pl.col("scope").unique().sort().alias("event_scopes"), pl.col("location").unique().sort().alias("event_locations"), (pl.col("event_type") == "Holiday").any().alias("has_holiday"), (pl.col("event_type") == "Transfer").any().alias("has_transfer"), (pl.col("event_type") == "Additional").any().alias("has_additional"), (pl.col("event_type") == "Event").any().alias("has_special_event"), (pl.col("scope") == "National").any().alias("has_national_event"), (pl.col("scope") == "Regional").any().alias("has_regional_event"), (pl.col("scope") == "Local").any().alias("has_local_event"), pl.col("is_transferred").any().alias("has_transferred_event")).with_columns((pl.col("event_count") > 0).alias("is_event_day")).sort("date", "store_id")
    return context, unmatched

def build_daily_oil_context(oil: pl.DataFrame, start_date: date, end_date: date) -> pl.DataFrame:
    summary = oil.select(pl.len().alias("rows"), pl.col("date").n_unique().alias("unique_dates"), pl.col("date").null_count().alias("null_dates"), ((pl.col("oil_price").is_not_null()) & ((~pl.col("oil_price").is_finite()) | (pl.col("oil_price") <= 0))).sum().alias("invalid_prices"))
    if int(scalar(summary, "rows")) != int(scalar(summary, "unique_dates")): raise ValueError("index_oil.csv contains duplicate dates")
    if int(scalar(summary, "null_dates")): raise ValueError("index_oil.csv contains null dates")
    if int(scalar(summary, "invalid_prices")): raise ValueError("index_oil.csv contains invalid non-null prices")
    dates = pl.DataFrame({"date": pl.date_range(start=start_date, end=end_date, interval="1d", eager=True)})
    return dates.join(oil, on="date", how="left").sort("date").with_columns(pl.col("oil_price").is_null().alias("oil_price_source_missing")).with_columns(pl.col("oil_price").forward_fill())

def add_event_defaults(frame: pl.LazyFrame) -> pl.LazyFrame:
    expressions = [pl.col("event_count").fill_null(0)]
    expressions.extend(pl.col(column).fill_null(False) for column in EVENT_BOOLEAN_COLUMNS)
    expressions.extend(pl.when(pl.col(column).is_null()).then(pl.lit([], dtype=pl.List(pl.String))).otherwise(pl.col(column)).alias(column) for column in EVENT_LIST_COLUMNS)
    return frame.with_columns(expressions)
