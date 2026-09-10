from __future__ import annotations

import polars as pl

KEY_COLUMNS = ["date", "product_id", "store_id"]
EXPECTED_TRAIN_ROWS = 1_800_000
EXPECTED_TEST_ROWS = 280_000
EXPECTED_OUTPUT_ROWS = EXPECTED_TRAIN_ROWS + EXPECTED_TEST_ROWS

HISTORY_SCHEMA = {"id": pl.Int64, "date": pl.String, "store_id": pl.Int16, "product_id": pl.Int32, "sales_count": pl.Float64, "promotion": pl.Boolean}
TEST_SCHEMA = {"id": pl.Int64, "date": pl.String, "store_id": pl.Int16, "product_id": pl.Int32, "promotion": pl.Boolean}
ITEM_SCHEMA = {"product_id": pl.Int32, "product_family": pl.String, "product_class": pl.Int32, "is_perishable": pl.Int8}
STORE_SCHEMA = {"store_id": pl.Int16, "city": pl.String, "department": pl.String, "store_type": pl.String, "store_cluster": pl.Int16}
EVENT_SCHEMA = {"date": pl.String, "event_type": pl.String, "scope": pl.String, "location": pl.String, "is_transferred": pl.Boolean}
OIL_SCHEMA = {"date": pl.String, "oil_price": pl.Float64}
EVENT_LIST_COLUMNS = ["event_types", "event_scopes", "event_locations"]
EVENT_BOOLEAN_COLUMNS = ["is_event_day", "is_effective_event_day", "is_effective_holiday", "is_1d_before_holiday", "is_2d_before_holiday", "has_holiday", "has_transfer", "has_additional", "has_special_event", "has_national_event", "has_regional_event", "has_local_event", "has_effective_national_event", "has_effective_regional_event", "has_effective_local_event", "has_transferred_event"]
OUTPUT_COLUMNS = ["date", "product_id", "store_id", "id", "dataset_split", "sales_count", "is_likely_stockout", "promotion", "product_family", "product_class", "is_perishable", "city", "department", "store_type", "store_cluster", "year", "month", "day", "day_of_week", "week_of_year", "is_weekend", "oil_price", "oil_price_source_missing", "event_count", "effective_event_count", "is_event_day", "is_effective_event_day", "event_types", "event_scopes", "event_locations", "has_holiday", "is_effective_holiday", "days_until_effective_holiday", "is_1d_before_holiday", "is_2d_before_holiday", "days_since_effective_holiday", "has_transfer", "has_additional", "has_special_event", "has_national_event", "has_regional_event", "has_local_event", "has_effective_national_event", "has_effective_regional_event", "has_effective_local_event", "has_transferred_event"]
