from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from pipeline.cache import run_cached_stage
from pipeline.features import FEATURE_GROUPS, parse_feature_groups
from pipeline.models import MODELS, get_model
from pipeline.stages.feature_engineering import build_training_features
from pipeline.stages.model_training import train_and_evaluate


class CacheTests(unittest.TestCase):
    def test_input_and_code_changes_invalidate_a_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            code = root / "stage.py"
            output = root / "output.txt"
            manifest = root / "manifest.json"
            source.write_text("one")
            code.write_text("version = 1")
            builds = 0

            def build(path: Path) -> dict[str, int]:
                nonlocal builds
                builds += 1
                path.write_text(f"build {builds}")
                return {"builds": builds}

            first_hit, _ = run_cached_stage(
                stage="test",
                output_path=output,
                manifest_path=manifest,
                inputs=[source],
                code_files=[code],
                parameters={"option": 1},
                build=build,
            )
            second_hit, _ = run_cached_stage(
                stage="test",
                output_path=output,
                manifest_path=manifest,
                inputs=[source],
                code_files=[code],
                parameters={"option": 1},
                build=build,
            )
            code.write_text("version = 2")
            third_hit, _ = run_cached_stage(
                stage="test",
                output_path=output,
                manifest_path=manifest,
                inputs=[source],
                code_files=[code],
                parameters={"option": 1},
                build=build,
            )

            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertFalse(third_hit)
            self.assertEqual(builds, 2)


class FeatureStageTests(unittest.TestCase):
    def test_feature_groups_are_selectable_and_lags_do_not_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged = root / "merged.parquet"
            output = root / "features.parquet"
            rows = 35
            pl.DataFrame(
                {
                    "date": [date(2026, 1, 1) + timedelta(days=index) for index in range(rows)],
                    "store_id": [1] * rows,
                    "product_id": [10] * rows,
                    "id": list(range(rows)),
                    "sales_count": [float(index) for index in range(rows)],
                    "dataset_split": ["train"] * rows,
                    "promotion": [False] * rows,
                    "year": [2026] * rows,
                    "month": [1] * rows,
                    "day": [(index % 28) + 1 for index in range(rows)],
                    "day_of_week": [(index % 7) + 1 for index in range(rows)],
                    "week_of_year": [(index // 7) + 1 for index in range(rows)],
                    "is_weekend": [False] * rows,
                }
            ).write_parquet(merged)

            metadata = build_training_features(
                merged, output, ("calendar", "sales_history")
            )
            result = pl.read_parquet(output)

            self.assertEqual(metadata["feature_groups"], ["calendar", "sales_history"])
            self.assertNotIn("feature_promotion", result.columns)
            self.assertIsNone(result["feature_sales_lag_1"][0])
            self.assertEqual(result["feature_sales_lag_1"][1], 0.0)
            self.assertEqual(result["feature_sales_lag_7"][8], 1.0)

    def test_feature_group_parser_and_model_registry(self) -> None:
        self.assertEqual(parse_feature_groups("all"), tuple(FEATURE_GROUPS))
        self.assertEqual(
            parse_feature_groups("calendar,promotion,calendar"),
            ("calendar", "promotion"),
        )
        self.assertEqual(set(MODELS), {"historical_quantile", "lightgbm"})
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            get_model("missing")


class ModelStageTests(unittest.TestCase):
    def _write_feature_fixture(self, path: Path, rows: int = 40) -> None:
        pl.DataFrame(
            {
                "date": [
                    date(2026, 1, 1) + timedelta(days=index) for index in range(rows)
                ],
                "store_id": [1] * rows,
                "product_id": [10] * rows,
                "id": list(range(rows)),
                "sales_count": [float(index % 9) for index in range(rows)],
                "feature_day": list(range(rows)),
                "feature_cycle": [index % 7 for index in range(rows)],
            }
        ).write_parquet(path)

    def test_historical_model_writes_chronological_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.parquet"
            report = root / "report.json"
            self._write_feature_fixture(features)

            result = train_and_evaluate(
                features, report, "historical_quantile", holdout_days=3, seed=42
            )

            self.assertTrue(report.is_file())
            self.assertEqual(result["training_rows"], 37)
            self.assertEqual(result["validation_rows"], 3)
            self.assertLess(result["training_end"], result["validation_start"])
            self.assertIn("mean_pinball", result)

    def test_lightgbm_model_is_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.parquet"
            report = root / "report.json"
            self._write_feature_fixture(features)

            result = train_and_evaluate(
                features, report, "lightgbm", holdout_days=5, seed=42
            )

            self.assertEqual(result["model"], "lightgbm")
            self.assertEqual(result["feature_count"], 2)
            self.assertGreaterEqual(result["mean_pinball"], 0)


if __name__ == "__main__":
    unittest.main()
