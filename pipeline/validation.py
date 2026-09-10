from __future__ import annotations

import polars as pl
from .config import KEY_COLUMNS

def scalar(frame: pl.DataFrame, column: str) -> object:
    return frame.get_column(column).item()

def validate_panel(frame: pl.LazyFrame, name: str, expected_rows: int, require_sales: bool) -> dict[str, object]:
    expressions = [pl.len().alias("row_count"), pl.struct(KEY_COLUMNS).n_unique().alias("unique_key_count"), pl.any_horizontal([pl.col(c).is_null() for c in KEY_COLUMNS]).sum().alias("null_key_count"), pl.col("id").null_count().alias("null_id_count"), pl.col("promotion").null_count().alias("null_promotion_count"), pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")]
    if require_sales:
        expressions += [pl.col("sales_count").null_count().alias("null_sales_count"), ((~pl.col("sales_count").is_finite()) | (pl.col("sales_count") < 0)).fill_null(True).sum().alias("invalid_sales_count")]
    summary = frame.select(expressions).collect()
    row_count, unique = int(scalar(summary, "row_count")), int(scalar(summary, "unique_key_count"))
    if row_count != expected_rows: raise ValueError(f"{name} has {row_count:,} rows; expected {expected_rows:,}")
    if unique != row_count: raise ValueError(f"{name} has {row_count - unique:,} duplicate keys")
    for column in ("null_key_count", "null_id_count", "null_promotion_count"):
        if int(scalar(summary, column)): raise ValueError(f"{name} failed {column}")
    if require_sales:
        for column in ("null_sales_count", "invalid_sales_count"):
            if int(scalar(summary, column)): raise ValueError(f"{name} failed {column}")
    return summary.row(0, named=True)

def validate_dimension(frame: pl.LazyFrame, name: str, key: str, expected_rows: int) -> None:
    summary = frame.select(pl.len().alias("row_count"), pl.col(key).n_unique().alias("unique_key_count"), pl.col(key).null_count().alias("null_key_count")).collect()
    if int(scalar(summary, "row_count")) != expected_rows: raise ValueError(f"{name} has {int(scalar(summary, 'row_count')):,} rows; expected {expected_rows:,}")
    if int(scalar(summary, "unique_key_count")) != expected_rows: raise ValueError(f"{name} contains duplicate {key} values")
    if int(scalar(summary, "null_key_count")): raise ValueError(f"{name} contains null {key} values")

def validate_context_sources(items: pl.LazyFrame, events: pl.LazyFrame) -> None:
    item = items.select(pl.any_horizontal([pl.col("product_family").is_null(), pl.col("product_class").is_null(), pl.col("is_perishable").is_null()]).sum().alias("null_context"), (~pl.col("is_perishable").is_in([0, 1])).fill_null(True).sum().alias("invalid_perishable")).collect()
    if int(scalar(item, "null_context")): raise ValueError("item_catalogue.csv contains null product context")
    if int(scalar(item, "invalid_perishable")): raise ValueError("item_catalogue.csv is_perishable must contain only 0 or 1")
    event = events.select(pl.any_horizontal([pl.col(c).is_null() for c in ("date", "event_type", "scope", "location", "is_transferred")]).sum().alias("null_fields"), (~pl.col("scope").is_in(["National", "Regional", "Local"])).fill_null(True).sum().alias("invalid_scopes")).collect()
    if int(scalar(event, "null_fields")): raise ValueError("events_and_holidays.csv contains null required fields")
    if int(scalar(event, "invalid_scopes")): raise ValueError("events_and_holidays.csv contains an unknown scope")

def validate_sample_ids(test: pl.LazyFrame, sample: pl.LazyFrame) -> None:
    if not test.select("id").collect().get_column("id").equals(sample.select("id").collect().get_column("id")):
        raise ValueError("sample_submission.csv IDs do not match test.csv IDs in order")
