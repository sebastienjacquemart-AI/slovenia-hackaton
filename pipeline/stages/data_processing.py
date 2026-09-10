"""Stage 1 reads, validates, and merges the challenge inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.data_processing.builder import build_dataset, write_parquet
from pipeline.data_processing.config import EXPECTED_TEST_ROWS, EXPECTED_TRAIN_ROWS
from pipeline.data_processing.sources import scan_sources
from pipeline.data_processing.validation import (
    validate_context_sources,
    validate_dimension,
    validate_panel,
    validate_sample_ids,
)

INPUT_FILES = (
    "sales_history.csv",
    "test.csv",
    "item_catalogue.csv",
    "store_info.csv",
    "events_and_holidays.csv",
    "index_oil.csv",
    "sample_submission.csv",
)


def source_paths(data_dir: Path, context_dir: Path) -> list[Path]:
    return [data_dir / filename for filename in INPUT_FILES] + [
        context_dir / "store_traffic.csv"
    ]


def build_merged_data(
    data_dir: Path, context_dir: Path, output_path: Path, compression: str = "zstd"
) -> dict[str, Any]:
    sources = scan_sources(data_dir, context_dir)
    history_stats = validate_panel(
        sources["history"],
        "sales_history.csv",
        EXPECTED_TRAIN_ROWS,
        require_sales=True,
    )
    test_stats = validate_panel(
        sources["test"], "test.csv", EXPECTED_TEST_ROWS, require_sales=False
    )
    validate_dimension(sources["items"], "item_catalogue.csv", "product_id", 500)
    validate_dimension(sources["stores"], "store_info.csv", "store_id", 20)
    validate_context_sources(sources["items"], sources["events"])
    validate_sample_ids(sources["test"], sources["sample"])

    dataset, unmatched_events = build_dataset(sources, history_stats, test_stats)
    metadata = write_parquet(dataset, output_path, compression)
    metadata["unmatched_events"] = unmatched_events.height
    return metadata
