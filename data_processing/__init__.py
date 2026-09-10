"""Components and command-line entry point for the forecasting dataset."""

from .cli import (
    add_event_defaults,
    build_daily_oil_context,
    build_dataset,
    build_event_context,
    parse_args,
    parse_date,
    scan_sources,
    scalar,
    validate_context_sources,
    validate_dimension,
    validate_panel,
    validate_sample_ids,
    write_parquet,
)

__all__ = [
    "add_event_defaults", "build_daily_oil_context", "build_dataset",
    "build_event_context", "parse_args", "parse_date", "scan_sources",
    "scalar", "validate_context_sources", "validate_dimension",
    "validate_panel", "validate_sample_ids", "write_parquet",
]
