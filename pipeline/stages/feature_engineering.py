"""Stage 2 encodes selected feature groups into a training table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pipeline.features import feature_expressions

IDENTIFIER_COLUMNS = ["date", "store_id", "product_id", "id", "sales_count"]


def _daily_mean_features(
    source: pl.LazyFrame,
    keys: list[str],
    prefix: str,
) -> pl.LazyFrame:
    value = f"_{prefix}_daily_mean"
    lag = f"_{prefix}_lag_1"
    daily = (
        source.group_by("date", *keys)
        .agg(pl.col("sales_count").mean().alias(value))
        .sort(*keys, "date")
        .with_columns(pl.col(value).shift(1).over(keys).alias(lag))
        .with_columns(
            pl.col(lag)
            .rolling_mean(window_size=7, min_samples=1)
            .over(keys)
            .cast(pl.Float32)
            .alias(f"feature_{prefix}_mean_7"),
            pl.col(lag)
            .rolling_mean(window_size=28, min_samples=1)
            .over(keys)
            .cast(pl.Float32)
            .alias(f"feature_{prefix}_mean_28"),
        )
    )
    return daily.select(
        "date",
        *keys,
        f"feature_{prefix}_mean_7",
        f"feature_{prefix}_mean_28",
    )


def _add_hierarchy_features(training: pl.LazyFrame) -> pl.LazyFrame:
    product = _daily_mean_features(training, ["product_id"], "product")
    store = _daily_mean_features(training, ["store_id"], "store")
    store_family = _daily_mean_features(
        training, ["store_id", "product_family"], "store_family"
    )
    return (
        training.join(product, on=["date", "product_id"], how="left", validate="m:1")
        .join(store, on=["date", "store_id"], how="left", validate="m:1")
        .join(
            store_family,
            on=["date", "store_id", "product_family"],
            how="left",
            validate="m:1",
        )
    )


def build_training_features(
    merged_path: Path,
    output_path: Path,
    groups: tuple[str, ...],
    compression: str = "zstd",
) -> dict[str, Any]:
    source = pl.scan_parquet(merged_path)
    training = source.filter(pl.col("dataset_split") == "train")
    if "hierarchy" in groups:
        training = _add_hierarchy_features(training)
    hierarchy_columns = (
        [
            "feature_product_mean_7",
            "feature_product_mean_28",
            "feature_store_mean_7",
            "feature_store_mean_28",
            "feature_store_family_mean_7",
            "feature_store_family_mean_28",
        ]
        if "hierarchy" in groups
        else []
    )
    training = training.sort(["store_id", "product_id", "date"]).select(
        *IDENTIFIER_COLUMNS,
        *hierarchy_columns,
        *feature_expressions(groups),
    )
    training.sink_parquet(output_path, compression=compression)
    written = pl.scan_parquet(output_path)
    schema = written.collect_schema()
    feature_columns = [name for name in schema.names() if name.startswith("feature_")]
    stats = (
        written.select(
            pl.len().alias("rows"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
            pl.col("sales_count").null_count().alias("null_targets"),
        )
        .collect()
        .row(0, named=True)
    )
    if int(stats["null_targets"]):
        raise ValueError("Stage 2 output contains null training targets")
    return {
        **stats,
        "feature_columns": feature_columns,
        "feature_groups": list(groups),
    }
