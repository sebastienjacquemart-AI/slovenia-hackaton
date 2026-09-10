"""Stage 2 encodes selected feature groups into a training table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pipeline.features import feature_expressions


IDENTIFIER_COLUMNS = ["date", "store_id", "product_id", "id", "sales_count"]


def build_training_features(
    merged_path: Path,
    output_path: Path,
    groups: tuple[str, ...],
    compression: str = "zstd",
) -> dict[str, Any]:
    source = pl.scan_parquet(merged_path)
    training = (
        source.filter(pl.col("dataset_split") == "train")
        .sort(["store_id", "product_id", "date"])
        .select(*IDENTIFIER_COLUMNS, *feature_expressions(groups))
    )
    training.sink_parquet(output_path, compression=compression)
    written = pl.scan_parquet(output_path)
    schema = written.collect_schema()
    feature_columns = [name for name in schema.names() if name.startswith("feature_")]
    stats = written.select(
        pl.len().alias("rows"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("sales_count").null_count().alias("null_targets"),
    ).collect().row(0, named=True)
    if int(stats["null_targets"]):
        raise ValueError("Stage 2 output contains null training targets")
    return {
        **stats,
        "feature_columns": feature_columns,
        "feature_groups": list(groups),
    }
