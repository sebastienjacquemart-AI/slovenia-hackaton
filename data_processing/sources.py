from __future__ import annotations

from pathlib import Path
import polars as pl
from .config import EVENT_SCHEMA, HISTORY_SCHEMA, ITEM_SCHEMA, OIL_SCHEMA, STORE_SCHEMA, TEST_SCHEMA

def parse_date(column: str = "date") -> pl.Expr:
    return pl.col(column).str.strptime(pl.Date, "%Y-%m-%d", strict=True)

def scan_sources(data_dir: Path) -> dict[str, pl.LazyFrame]:
    files = {"history": "sales_history.csv", "test": "test.csv", "items": "item_catalogue.csv", "stores": "store_info.csv", "events": "events_and_holidays.csv", "oil": "index_oil.csv", "sample": "sample_submission.csv"}
    paths = {name: data_dir / filename for name, filename in files.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required data files: " + ", ".join(missing))
    return {
        "history": pl.scan_csv(paths["history"], schema=HISTORY_SCHEMA).with_columns(parse_date()),
        "test": pl.scan_csv(paths["test"], schema=TEST_SCHEMA).with_columns(parse_date()),
        "items": pl.scan_csv(paths["items"], schema=ITEM_SCHEMA),
        "stores": pl.scan_csv(paths["stores"], schema=STORE_SCHEMA),
        "events": pl.scan_csv(paths["events"], schema=EVENT_SCHEMA).with_columns(parse_date()),
        "oil": pl.scan_csv(paths["oil"], schema=OIL_SCHEMA).with_columns(parse_date()),
        "sample": pl.scan_csv(paths["sample"], schema_overrides={"id": pl.Int64}),
    }
