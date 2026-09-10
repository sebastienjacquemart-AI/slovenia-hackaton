"""Build the typed, contextualized forecasting dataset with Polars."""

from __future__ import annotations

import argparse
import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from .context import (
    add_event_defaults as _add_event_defaults,
    build_daily_oil_context as _build_daily_oil_context,
    build_event_context as _build_event_context,
)
from .sources import parse_date as _parse_date, scan_sources as _scan_sources
from .validation import (
    scalar as _scalar,
    validate_context_sources as _validate_context_sources,
    validate_dimension as _validate_dimension,
    validate_panel as _validate_panel,
    validate_sample_ids as _validate_sample_ids,
)


KEY_COLUMNS = ["date", "product_id", "store_id"]
EXPECTED_TRAIN_ROWS = 1_800_000
EXPECTED_TEST_ROWS = 280_000
EXPECTED_OUTPUT_ROWS = EXPECTED_TRAIN_ROWS + EXPECTED_TEST_ROWS

HISTORY_SCHEMA = {
    "id": pl.Int64,
    "date": pl.String,
    "store_id": pl.Int16,
    "product_id": pl.Int32,
    "sales_count": pl.Float64,
    "promotion": pl.Boolean,
}
TEST_SCHEMA = {
    "id": pl.Int64,
    "date": pl.String,
    "store_id": pl.Int16,
    "product_id": pl.Int32,
    "promotion": pl.Boolean,
}
ITEM_SCHEMA = {
    "product_id": pl.Int32,
    "product_family": pl.String,
    "product_class": pl.Int32,
    "is_perishable": pl.Int8,
}
STORE_SCHEMA = {
    "store_id": pl.Int16,
    "city": pl.String,
    "department": pl.String,
    "store_type": pl.String,
    "store_cluster": pl.Int16,
}
EVENT_SCHEMA = {
    "date": pl.String,
    "event_type": pl.String,
    "scope": pl.String,
    "location": pl.String,
    "is_transferred": pl.Boolean,
}
OIL_SCHEMA = {"date": pl.String, "oil_price": pl.Float64}

EVENT_LIST_COLUMNS = ["event_types", "event_scopes", "event_locations"]
EVENT_BOOLEAN_COLUMNS = [
    "is_event_day",
    "is_effective_event_day",
    "is_effective_holiday",
    "is_1d_before_holiday",
    "is_2d_before_holiday",
    "has_holiday",
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
]

OUTPUT_COLUMNS = [
    "date",
    "product_id",
    "store_id",
    "id",
    "dataset_split",
    "sales_count",
    "promotion",
    "product_family",
    "product_class",
    "is_perishable",
    "city",
    "department",
    "store_type",
    "store_cluster",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "oil_price",
    "oil_price_source_missing",
    "event_count",
    "effective_event_count",
    "is_event_day",
    "is_effective_event_day",
    "event_types",
    "event_scopes",
    "event_locations",
    "has_holiday",
    "is_effective_holiday",
    "days_until_effective_holiday",
    "is_1d_before_holiday",
    "is_2d_before_holiday",
    "days_since_effective_holiday",
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
]


def parse_args() -> argparse.Namespace:
    challenge_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Merge challenge CSV files into one typed Parquet dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=challenge_dir / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=challenge_dir / "data" / "processed_dataset.parquet",
    )
    parser.add_argument(
        "--compression",
        choices=("zstd", "snappy", "lz4", "uncompressed"),
        default="zstd",
    )
    return parser.parse_args()


def parse_date(column: str = "date") -> pl.Expr:
    return pl.col(column).str.strptime(pl.Date, "%Y-%m-%d", strict=True)


def scan_sources(data_dir: Path) -> dict[str, pl.LazyFrame]:
    required = {
        "history": data_dir / "sales_history.csv",
        "test": data_dir / "test.csv",
        "items": data_dir / "item_catalogue.csv",
        "stores": data_dir / "store_info.csv",
        "events": data_dir / "events_and_holidays.csv",
        "oil": data_dir / "index_oil.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required data files: " + ", ".join(missing))

    return {
        "history": pl.scan_csv(required["history"], schema=HISTORY_SCHEMA).with_columns(
            parse_date()
        ),
        "test": pl.scan_csv(required["test"], schema=TEST_SCHEMA).with_columns(
            parse_date()
        ),
        "items": pl.scan_csv(required["items"], schema=ITEM_SCHEMA),
        "stores": pl.scan_csv(required["stores"], schema=STORE_SCHEMA),
        "events": pl.scan_csv(required["events"], schema=EVENT_SCHEMA).with_columns(
            parse_date()
        ),
        "oil": pl.scan_csv(required["oil"], schema=OIL_SCHEMA).with_columns(parse_date()),
        "sample": pl.scan_csv(required["sample"], schema_overrides={"id": pl.Int64}),
    }


def scalar(frame: pl.DataFrame, column: str) -> object:
    return frame.get_column(column).item()


def validate_panel(
    frame: pl.LazyFrame,
    name: str,
    expected_rows: int,
    require_sales: bool,
) -> dict[str, object]:
    expressions: list[pl.Expr] = [
        pl.len().alias("row_count"),
        pl.struct(KEY_COLUMNS).n_unique().alias("unique_key_count"),
        pl.any_horizontal([pl.col(column).is_null() for column in KEY_COLUMNS])
        .sum()
        .alias("null_key_count"),
        pl.col("id").null_count().alias("null_id_count"),
        pl.col("promotion").null_count().alias("null_promotion_count"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ]
    if require_sales:
        expressions.extend(
            [
                pl.col("sales_count").null_count().alias("null_sales_count"),
                (
                    (~pl.col("sales_count").is_finite())
                    | (pl.col("sales_count") < 0)
                )
                .fill_null(True)
                .sum()
                .alias("invalid_sales_count"),
            ]
        )

    summary = frame.select(expressions).collect()
    row_count = int(scalar(summary, "row_count"))
    unique_key_count = int(scalar(summary, "unique_key_count"))
    if row_count != expected_rows:
        raise ValueError(f"{name} has {row_count:,} rows; expected {expected_rows:,}")
    if unique_key_count != row_count:
        raise ValueError(f"{name} has {row_count - unique_key_count:,} duplicate keys")
    for column in ("null_key_count", "null_id_count", "null_promotion_count"):
        if int(scalar(summary, column)):
            raise ValueError(f"{name} failed {column}")
    if require_sales:
        for column in ("null_sales_count", "invalid_sales_count"):
            if int(scalar(summary, column)):
                raise ValueError(f"{name} failed {column}")

    return summary.row(0, named=True)


def validate_dimension(
    frame: pl.LazyFrame, name: str, key: str, expected_rows: int
) -> None:
    summary = frame.select(
        pl.len().alias("row_count"),
        pl.col(key).n_unique().alias("unique_key_count"),
        pl.col(key).null_count().alias("null_key_count"),
    ).collect()
    row_count = int(scalar(summary, "row_count"))
    if row_count != expected_rows:
        raise ValueError(f"{name} has {row_count:,} rows; expected {expected_rows:,}")
    if int(scalar(summary, "unique_key_count")) != row_count:
        raise ValueError(f"{name} contains duplicate {key} values")
    if int(scalar(summary, "null_key_count")):
        raise ValueError(f"{name} contains null {key} values")


def validate_context_sources(
    items: pl.LazyFrame, events: pl.LazyFrame
) -> None:
    item_summary = items.select(
        pl.any_horizontal(
            [
                pl.col("product_family").is_null(),
                pl.col("product_class").is_null(),
                pl.col("is_perishable").is_null(),
            ]
        )
        .sum()
        .alias("null_context"),
        (~pl.col("is_perishable").is_in([0, 1]))
        .fill_null(True)
        .sum()
        .alias("invalid_perishable"),
    ).collect()
    if int(scalar(item_summary, "null_context")):
        raise ValueError("item_catalogue.csv contains null product context")
    if int(scalar(item_summary, "invalid_perishable")):
        raise ValueError("item_catalogue.csv is_perishable must contain only 0 or 1")

    event_summary = events.select(
        pl.any_horizontal(
            [
                pl.col("date").is_null(),
                pl.col("event_type").is_null(),
                pl.col("scope").is_null(),
                pl.col("location").is_null(),
                pl.col("is_transferred").is_null(),
            ]
        )
        .sum()
        .alias("null_fields"),
        (~pl.col("scope").is_in(["National", "Regional", "Local"]))
        .fill_null(True)
        .sum()
        .alias("invalid_scopes"),
    ).collect()
    if int(scalar(event_summary, "null_fields")):
        raise ValueError("events_and_holidays.csv contains null required fields")
    if int(scalar(event_summary, "invalid_scopes")):
        raise ValueError("events_and_holidays.csv contains an unknown scope")


def validate_sample_ids(test: pl.LazyFrame, sample: pl.LazyFrame) -> None:
    test_ids = test.select("id").collect().get_column("id")
    sample_ids = sample.select("id").collect().get_column("id")
    if not test_ids.equals(sample_ids):
        raise ValueError("sample_submission.csv IDs do not match test.csv IDs in order")


def build_event_context(
    events: pl.DataFrame, stores: pl.DataFrame
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
        | (
            (pl.col("scope") == "Local")
            & (pl.col("location") == pl.col("city"))
        )
    )

    unmatched = events_with_id.join(
        expanded.select("_event_id").unique(), on="_event_id", how="anti"
    ).drop("_event_id")

    context = (
        expanded.group_by("date", "store_id")
        .agg(
            pl.len().cast(pl.Int16).alias("event_count"),
            pl.col("event_type").unique().sort().alias("event_types"),
            pl.col("scope").unique().sort().alias("event_scopes"),
            pl.col("location").unique().sort().alias("event_locations"),
            (pl.col("event_type") == "Holiday").any().alias("has_holiday"),
            (pl.col("event_type") == "Transfer").any().alias("has_transfer"),
            (pl.col("event_type") == "Additional").any().alias("has_additional"),
            (pl.col("event_type") == "Event").any().alias("has_special_event"),
            (pl.col("scope") == "National").any().alias("has_national_event"),
            (pl.col("scope") == "Regional").any().alias("has_regional_event"),
            (pl.col("scope") == "Local").any().alias("has_local_event"),
            pl.col("is_transferred").any().alias("has_transferred_event"),
        )
        .with_columns((pl.col("event_count") > 0).alias("is_event_day"))
        .sort("date", "store_id")
    )
    return context, unmatched


def build_daily_oil_context(
    oil: pl.DataFrame, start_date: date, end_date: date
) -> pl.DataFrame:
    summary = oil.select(
        pl.len().alias("rows"),
        pl.col("date").n_unique().alias("unique_dates"),
        pl.col("date").null_count().alias("null_dates"),
        (
            (pl.col("oil_price").is_not_null())
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

    daily_dates = pl.DataFrame(
        {
            "date": pl.date_range(
                start=start_date, end=end_date, interval="1d", eager=True
            )
        }
    )
    return (
        daily_dates.join(oil, on="date", how="left")
        .sort("date")
        .with_columns(pl.col("oil_price").is_null().alias("oil_price_source_missing"))
        .with_columns(pl.col("oil_price").forward_fill())
    )


def add_event_defaults(frame: pl.LazyFrame) -> pl.LazyFrame:
    expressions: list[pl.Expr] = [pl.col("event_count").fill_null(0)]
    expressions.extend(pl.col(column).fill_null(False) for column in EVENT_BOOLEAN_COLUMNS)
    expressions.extend(
        pl.when(pl.col(column).is_null())
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col(column))
        .alias(column)
        for column in EVENT_LIST_COLUMNS
    )
    return frame.with_columns(expressions)


def build_dataset(
    sources: dict[str, pl.LazyFrame],
    history_stats: dict[str, object],
    test_stats: dict[str, object],
) -> tuple[pl.LazyFrame, pl.DataFrame]:
    history = sources["history"].with_columns(
        pl.lit("train").alias("dataset_split")
    )
    test = sources["test"].with_columns(
        pl.lit(None, dtype=pl.Float64).alias("sales_count"),
        pl.lit("test").alias("dataset_split"),
    ).select(history.collect_schema().names())
    base = pl.concat([history, test], how="vertical")

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
    events_eager = sources["events"].collect()
    event_context, unmatched_events = build_event_context(
        events_eager, stores_eager, start_date, end_date
    )

    oil_context = build_daily_oil_context(sources["oil"].collect(), start_date, end_date)

    frame = (
        frame.join(oil_context.lazy(), on="date", how="left", validate="m:1")
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
        pl.any_horizontal(
            [
                pl.col("product_family").is_null(),
                pl.col("product_class").is_null(),
                pl.col("is_perishable").is_null(),
            ]
        )
        .sum()
        .alias("missing_product_context"),
        pl.any_horizontal(
            [
                pl.col("city").is_null(),
                pl.col("department").is_null(),
                pl.col("store_type").is_null(),
                pl.col("store_cluster").is_null(),
            ]
        )
        .sum()
        .alias("missing_store_context"),
        ((pl.col("dataset_split") == "train") & pl.col("sales_count").is_null())
        .sum()
        .alias("missing_train_targets"),
        ((pl.col("dataset_split") == "test") & pl.col("sales_count").is_not_null())
        .sum()
        .alias("present_test_targets"),
        (pl.col("dataset_split") == "train").sum().alias("train_rows"),
        (pl.col("dataset_split") == "test").sum().alias("test_rows"),
    ).collect()
    expected = {
        "row_count": EXPECTED_OUTPUT_ROWS,
        "unique_key_count": EXPECTED_OUTPUT_ROWS,
        "unique_id_count": EXPECTED_OUTPUT_ROWS,
        "missing_products": 0,
        "missing_stores": 0,
        "missing_product_context": 0,
        "missing_store_context": 0,
        "missing_train_targets": 0,
        "present_test_targets": 0,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "test_rows": EXPECTED_TEST_ROWS,
    }
    for column, expected_value in expected.items():
        actual = int(scalar(audit, column))
        if actual != expected_value:
            raise ValueError(f"Final validation failed for {column}: {actual} != {expected_value}")

    return frame.select(OUTPUT_COLUMNS).sort(KEY_COLUMNS), unmatched_events


def write_parquet(
    frame: pl.LazyFrame, output_path: Path, compression: str
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=f".{output_path.stem}-", suffix=".parquet", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    temporary_path.unlink()

    try:
        frame.sink_parquet(temporary_path, compression=compression)
        written = pl.scan_parquet(temporary_path)
        schema = written.collect_schema()
        if schema.names() != OUTPUT_COLUMNS:
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
        temporary_path.replace(output_path)
        return stats.row(0, named=True)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    sources = scan_sources(args.data_dir)

    history_stats = validate_panel(
        sources["history"], "sales_history.csv", EXPECTED_TRAIN_ROWS, require_sales=True
    )
    test_stats = validate_panel(
        sources["test"], "test.csv", EXPECTED_TEST_ROWS, require_sales=False
    )
    validate_dimension(sources["items"], "item_catalogue.csv", "product_id", 500)
    validate_dimension(sources["stores"], "store_info.csv", "store_id", 20)
    validate_context_sources(sources["items"], sources["events"])
    validate_sample_ids(sources["test"], sources["sample"])

    dataset, unmatched_events = build_dataset(sources, history_stats, test_stats)
    stats = write_parquet(dataset, args.output, args.compression)

    print(f"Wrote {int(stats['row_count']):,} rows to {args.output}")
    print(f"Date range: {stats['min_date']} through {stats['max_date']}")
    print(f"Columns: {len(OUTPUT_COLUMNS)}")
    print(f"Unmatched source events: {unmatched_events.height}")
    if unmatched_events.height:
        print(unmatched_events.select("date", "event_type", "scope", "location"))


# Keep the original public imports stable while the responsibilities live in modules.
parse_date = _parse_date
scan_sources = _scan_sources
scalar = _scalar
validate_panel = _validate_panel
validate_dimension = _validate_dimension
validate_context_sources = _validate_context_sources
validate_sample_ids = _validate_sample_ids
build_event_context = _build_event_context
build_daily_oil_context = _build_daily_oil_context
add_event_defaults = _add_event_defaults


if __name__ == "__main__":
    main()
