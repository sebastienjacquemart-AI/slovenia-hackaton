"""Stage 1 data-processing components."""

from .builder import build_dataset, write_parquet
from .context import add_event_defaults, build_daily_oil_context, build_event_context
from .sources import parse_date, scan_sources
from .validation import (
    scalar,
    validate_context_sources,
    validate_dimension,
    validate_panel,
    validate_sample_ids,
)

__all__ = [
    "add_event_defaults",
    "build_daily_oil_context",
    "build_dataset",
    "build_event_context",
    "parse_date",
    "scalar",
    "scan_sources",
    "validate_context_sources",
    "validate_dimension",
    "validate_panel",
    "validate_sample_ids",
    "write_parquet",
]
