"""Build and validate the merged Stage 1 dataset."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from pipeline.stockouts import add_stockout_flag

from .config import (
    EXPECTED_OUTPUT_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    KEY_COLUMNS,
    OUTPUT_COLUMNS,
)
from .context import (
    add_event_defaults,
    build_daily_oil_context,
    build_daily_traffic_context,
    build_event_context,
)
from .validation import scalar


def build_dataset(
    sources: dict[str, pl.LazyFrame],
    history_stats: dict[str, object],
    test_stats: dict[str, object],
) -> tuple[pl.LazyFrame, pl.DataFrame]:
    history = sources["history"].with_columns(pl.lit("train").alias("dataset_split"))
    test = (
        sources["test"]
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("sales_count"),
            pl.lit("test").alias("dataset_split"),
        )
        .select(history.collect_schema().names())
    )
    base = add_stockout_flag(pl.concat([history, test], how="vertical"))

    items = sources["items"].with_columns(
        (pl.col("is_perishable") == 1).alias("is_perishable"),
        pl.lit(True).alias("_product_match"),
    )
    stores = sources["stores"].with_columns(pl.lit(True).alias("_store_match"))
    frame = base.join(items, on="product_id", how="left", validate="m:1").join(
        stores, on="store_id", how="left", validate="m:1"
    )
    start_date = min(history_stats["min_date"], test_stats["min_date"])
    end_date = max(history_stats["max_date"], test_stats["max_date"])
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise TypeError("Date bounds did not parse as dates")
    stores_eager = sources["stores"].collect()
    traffic_context = build_daily_traffic_context(
        sources["traffic"].collect(), stores_eager, start_date, end_date
    )
    event_context, unmatched_events = build_event_context(
        sources["events"].collect(), stores_eager, start_date, end_date
    )
    oil_context = build_daily_oil_context(
        sources["oil"].collect(), start_date, end_date
    )
    frame = (
        frame.join(
            traffic_context.lazy(), on=["date", "store_id"], how="left", validate="m:1"
        )
        .join(oil_context.lazy(), on="date", how="left", validate="m:1")
        .join(event_context.lazy(), on=["date", "store_id"], how="left", validate="m:1")
        .with_columns(
            pl.col("date").dt.year().cast(pl.Int16).alias("year"),
            pl.col("date").dt.month().cast(pl.Int8).alias("month"),
            pl.col("date").dt.day().cast(pl.Int8).alias("day"),
            pl.col("date").dt.weekday().cast(pl.Int8).alias("day_of_week"),
            pl.col("date").dt.week().cast(pl.Int8).alias("week_of_year"),
        )
        .with_columns((pl.col("day_of_week") >= 6).alias("is_weekend"))
    )
    frame = add_event_defaults(frame)

    audit = frame.select(
        pl.len().alias("row_count"),
        pl.struct(KEY_COLUMNS).n_unique().alias("unique_key_count"),
        pl.col("id").n_unique().alias("unique_id_count"),
        pl.col("_product_match").null_count().alias("missing_products"),
        pl.col("_store_match").null_count().alias("missing_stores"),
        ((pl.col("dataset_split") == "train") & pl.col("sales_count").is_null())
        .sum()
        .alias("missing_train_targets"),
        ((pl.col("dataset_split") == "test") & pl.col("sales_count").is_not_null())
        .sum()
        .alias("present_test_targets"),
        ((pl.col("dataset_split") == "train") & pl.col("transaction_count").is_null())
        .sum()
        .alias("missing_train_traffic"),
        ((pl.col("dataset_split") == "train") & pl.col("is_likely_stockout").is_null())
        .sum()
        .alias("missing_train_stockout_flags"),
        (
            (pl.col("dataset_split") == "test")
            & pl.col("is_likely_stockout").is_not_null()
        )
        .sum()
        .alias("present_test_stockout_flags"),
        (pl.col("dataset_split") == "train").sum().alias("train_rows"),
        (pl.col("dataset_split") == "test").sum().alias("test_rows"),
    ).collect()
    expected = {
        "row_count": EXPECTED_OUTPUT_ROWS,
        "unique_key_count": EXPECTED_OUTPUT_ROWS,
        "unique_id_count": EXPECTED_OUTPUT_ROWS,
        "missing_products": 0,
        "missing_stores": 0,
        "missing_train_targets": 0,
        "present_test_targets": 0,
        "missing_train_traffic": 0,
        "missing_train_stockout_flags": 0,
        "present_test_stockout_flags": 0,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "test_rows": EXPECTED_TEST_ROWS,
    }
    for column, expected_value in expected.items():
        actual = int(scalar(audit, column))
        if actual != expected_value:
            raise ValueError(
                f"Final validation failed for {column}: {actual} != {expected_value}"
            )
    return frame.select(OUTPUT_COLUMNS).sort(KEY_COLUMNS), unmatched_events


def write_parquet(
    frame: pl.LazyFrame, output_path: Path, compression: str
) -> dict[str, object]:
    frame.sink_parquet(output_path, compression=compression)
    written = pl.scan_parquet(output_path)
    if written.collect_schema().names() != OUTPUT_COLUMNS:
        raise ValueError("Written Parquet schema has an unexpected column order")
    stats = written.select(
        pl.len().alias("row_count"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("sales_count").null_count().alias("null_sales_count"),
    ).collect()
    if int(scalar(stats, "row_count")) != EXPECTED_OUTPUT_ROWS:
        raise ValueError("Written Parquet row count is incorrect")
    if int(scalar(stats, "null_sales_count")) != EXPECTED_TEST_ROWS:
        raise ValueError("Written Parquet target null count is incorrect")
    return stats.row(0, named=True)
