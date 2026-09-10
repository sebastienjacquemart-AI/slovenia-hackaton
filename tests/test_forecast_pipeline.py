from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from pipeline.cache import run_cached_stage
from pipeline.features import FEATURE_GROUPS, parse_feature_groups
from pipeline.forecast import (
    ForecastExplanations,
    SalesPanel,
    recursive_forecast_with_shap,
    write_shap_values,
)
from pipeline.models import MODELS, get_model
from pipeline.stages.feature_engineering import build_training_features
from pipeline.stages.model_training import train_and_evaluate
from pipeline.stockouts import likely_stockout_flags


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
                    "date": [
                        date(2026, 1, 1) + timedelta(days=index)
                        for index in range(rows)
                    ],
                    "store_id": [1] * rows,
                    "product_id": [10] * rows,
                    "id": list(range(rows)),
                    "sales_count": [float(index) for index in range(rows)],
                    "dataset_split": ["train"] * rows,
                    "is_likely_stockout": [False] * rows,
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

    def test_stockout_flag_uses_only_prior_sales(self) -> None:
        sales = pl.Series([100, 100, 100, 0, 0, 110]).to_numpy().reshape(-1, 1)

        flags = likely_stockout_flags(sales)

        self.assertEqual(flags[:, 0].tolist(), [False, False, False, True, True, False])


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

    def test_shap_values_follow_predictions_when_quantiles_are_sorted(self) -> None:
        class FakeModel:
            def __init__(self, predictions: list[float]) -> None:
                self.predictions = np.asarray(predictions, dtype=np.float32)

            def predict(
                self, features: np.ndarray, pred_contrib: bool = False
            ) -> np.ndarray:
                if not pred_contrib:
                    return self.predictions
                contributions = np.zeros(
                    (len(features), features.shape[1] + 1), dtype=np.float32
                )
                contributions[:, -1] = self.predictions
                return contributions

        models = {
            0.25: FakeModel([3.0, -1.0]),
            0.50: FakeModel([1.0, 4.0]),
            0.75: FakeModel([2.0, 2.0]),
            0.95: FakeModel([4.0, 3.0]),
        }
        history = np.ones((35, 2), dtype=np.float32)

        forecasts, explanations = recursive_forecast_with_shap(
            models, history, horizon=1
        )

        np.testing.assert_array_equal(
            forecasts[0], [[1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 3.0, 4.0]]
        )
        np.testing.assert_allclose(
            explanations.source_quantiles[0],
            [[0.50, 0.75, 0.25, 0.95], [0.25, 0.75, 0.95, 0.50]],
        )
        reconstructed = explanations.base_values + explanations.shap_values.sum(axis=-1)
        np.testing.assert_allclose(reconstructed, explanations.raw_predictions)

    def test_shap_sidecar_is_aligned_to_test_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.csv"
            output_path = root / "predictions.shap.parquet"
            pd.DataFrame(
                {
                    "id": ["second", "first"],
                    "date": ["2026-01-01", "2026-01-01"],
                    "store_id": [2, 1],
                    "product_id": [20, 10],
                }
            ).to_csv(test_path, index=False)
            panel = SalesPanel(
                dates=pd.DatetimeIndex(["2025-12-31"]),
                pairs=pd.MultiIndex.from_tuples(
                    [(1, 10), (2, 20)], names=["store_id", "product_id"]
                ),
                values=np.zeros((1, 2), dtype=np.float32),
            )
            shap_values = np.ones((1, 2, 4, 2), dtype=np.float32)
            base_values = np.zeros((1, 2, 4), dtype=np.float32)
            raw_predictions = shap_values.sum(axis=-1)
            forecasts = raw_predictions.copy()
            explanations = ForecastExplanations(
                raw_predictions=raw_predictions,
                shap_values=shap_values,
                base_values=base_values,
                source_quantiles=np.broadcast_to(
                    np.asarray([0.25, 0.50, 0.75, 0.95], dtype=np.float32),
                    (1, 2, 4),
                ),
            )

            write_shap_values(
                panel,
                forecasts,
                explanations,
                ("first_feature", "second_feature"),
                test_path,
                output_path,
            )

            result = pl.read_parquet(output_path)
            self.assertEqual(result.height, 8)
            self.assertEqual(result["id"].head(4).to_list(), ["second"] * 4)
            self.assertIn("shap_first_feature", result.columns)
            self.assertIn("shap_second_feature", result.columns)


if __name__ == "__main__":
    unittest.main()
