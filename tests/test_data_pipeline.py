from __future__ import annotations

import unittest
from datetime import date

import polars as pl

from pipeline.data_processing import (
    build_daily_oil_context,
    build_event_context,
    validate_context_sources,
)


class EventContextTests(unittest.TestCase):
    def test_events_follow_geographic_scope(self) -> None:
        stores = pl.DataFrame(
            {
                "store_id": [1, 2],
                "city": ["Alpha", "Beta"],
                "department": ["North", "South"],
            },
            schema={"store_id": pl.Int16, "city": pl.String, "department": pl.String},
        )
        events = pl.DataFrame(
            {
                "date": [date(2026, 1, 2)] * 4,
                "event_type": ["Holiday", "Event", "Additional", "Holiday"],
                "scope": ["National", "Regional", "Local", "Local"],
                "location": ["Country", "North", "Beta", "Elsewhere"],
                "is_transferred": [False, False, False, False],
            }
        )

        context, unmatched = build_event_context(events, stores)

        self.assertEqual(unmatched.height, 1)
        counts = dict(context.select("store_id", "event_count").iter_rows())
        self.assertEqual(counts, {1: 2, 2: 2})
        store_one = context.filter(pl.col("store_id") == 1).row(0, named=True)
        store_two = context.filter(pl.col("store_id") == 2).row(0, named=True)
        self.assertTrue(store_one["has_national_event"])
        self.assertTrue(store_one["has_regional_event"])
        self.assertFalse(store_one["has_local_event"])
        self.assertTrue(store_two["has_national_event"])
        self.assertTrue(store_two["has_local_event"])

    def test_transferred_holidays_and_store_aware_distances(self) -> None:
        stores = pl.DataFrame(
            {
                "store_id": [1, 2],
                "city": ["Alpha", "Beta"],
                "department": ["North", "South"],
            },
            schema={"store_id": pl.Int16, "city": pl.String, "department": pl.String},
        )
        events = pl.DataFrame(
            {
                "date": [
                    date(2026, 1, 3),
                    date(2026, 1, 4),
                    date(2026, 1, 6),
                    date(2026, 1, 5),
                ],
                "event_type": ["Holiday", "Transfer", "Holiday", "Holiday"],
                "scope": ["National", "National", "Local", "Local"],
                "location": ["Country", "Country", "Alpha", "Elsewhere"],
                "is_transferred": [True, False, False, False],
            }
        )

        context, unmatched = build_event_context(
            events, stores, date(2026, 1, 2), date(2026, 1, 7)
        )

        self.assertEqual(unmatched.height, 1)
        transferred_away = context.filter(
            (pl.col("date") == date(2026, 1, 3)) & (pl.col("store_id") == 1)
        ).row(0, named=True)
        self.assertTrue(transferred_away["has_holiday"])
        self.assertTrue(transferred_away["is_event_day"])
        self.assertFalse(transferred_away["is_effective_holiday"])
        self.assertFalse(transferred_away["is_effective_event_day"])
        self.assertEqual(transferred_away["event_count"], 1)
        self.assertEqual(transferred_away["effective_event_count"], 0)
        self.assertEqual(transferred_away["days_until_effective_holiday"], 1)
        self.assertTrue(transferred_away["is_1d_before_holiday"])

        transferred_to = context.filter(
            (pl.col("date") == date(2026, 1, 4)) & (pl.col("store_id") == 1)
        ).row(0, named=True)
        self.assertTrue(transferred_to["is_effective_holiday"])
        self.assertEqual(transferred_to["days_until_effective_holiday"], 0)
        self.assertEqual(transferred_to["days_since_effective_holiday"], 0)

        before_local = context.filter(pl.col("date") == date(2026, 1, 5)).sort(
            "store_id"
        )
        self.assertEqual(
            before_local["days_until_effective_holiday"].to_list(), [1, -1]
        )
        self.assertEqual(before_local["is_1d_before_holiday"].to_list(), [True, False])


class OilContextTests(unittest.TestCase):
    def test_oil_prices_only_fill_forward(self) -> None:
        oil = pl.DataFrame(
            {
                "date": [date(2026, 1, 2), date(2026, 1, 4)],
                "oil_price": [50.0, 52.0],
            }
        )

        result = build_daily_oil_context(oil, date(2026, 1, 1), date(2026, 1, 5))

        self.assertEqual(
            result.get_column("oil_price").to_list(),
            [None, 50.0, 50.0, 52.0, 52.0],
        )
        self.assertEqual(
            result.get_column("oil_price_source_missing").to_list(),
            [True, False, True, False, True],
        )


class SourceValidationTests(unittest.TestCase):
    def test_unknown_event_scope_is_rejected(self) -> None:
        items = pl.DataFrame(
            {
                "product_family": ["Family"],
                "product_class": [1],
                "is_perishable": [0],
            }
        ).lazy()
        events = pl.DataFrame(
            {
                "date": [date(2026, 1, 1)],
                "event_type": ["Holiday"],
                "scope": ["Unknown"],
                "location": ["Place"],
                "is_transferred": [False],
            }
        ).lazy()

        with self.assertRaisesRegex(ValueError, "unknown scope"):
            validate_context_sources(items, events)


if __name__ == "__main__":
    unittest.main()
