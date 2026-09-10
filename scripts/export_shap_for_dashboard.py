"""Trim a pipeline.forecast SHAP sidecar (`*.shap.parquet`) into a small CSV
the dashboard can load in-browser.

The raw sidecar has one row per (id, quantile) and one `shap_<feature>`
column per model feature — too wide/heavy to fetch and parse client-side
with no backend and no parquet reader. SHAP contributions are additive by
construction, so we lose nothing by summing them into a handful of
domain-meaningful buckets that match the feature groups already used
throughout this project (see gustavo_context/CONTEXT.md) — the bucket
values still sum exactly to prediction - base_value, both on the model's
own training scale (e.g. log1p). That is NOT the same scale as the
submitted forecast (sales units, after inverse-transform and
postprocessing) — display_base/display_prediction carry that separately,
for anything shown to a reviewer as "the forecast."

Usage:
    uv run python scripts/export_shap_for_dashboard.py [--shap-parquet PATH] [--output PATH]

Defaults to the newest `*.shap.parquet` under pipeline/predictions/, and
writes `<same name>.dashboard.csv` beside it.
"""

import argparse
from pathlib import Path

import polars as pl

BUCKETS = {
    "sales_trend": [
        "shap_sales_lag_1", "shap_sales_lag_7", "shap_sales_lag_14", "shap_sales_lag_28",
        "shap_sales_mean_7", "shap_sales_std_7", "shap_sales_mean_28", "shap_sales_std_28",
        "shap_likely_stockout_lag_1",
    ],
    "promotion": ["shap_promotion"],
    "calendar": [
        "shap_year", "shap_month", "shap_day", "shap_day_of_week",
        "shap_week_of_year", "shap_is_weekend",
    ],
    "holiday_event": [
        "shap_event_count", "shap_effective_event_count", "shap_is_event_day",
        "shap_is_effective_event_day", "shap_has_holiday", "shap_is_effective_holiday",
        "shap_days_until_effective_holiday", "shap_days_since_effective_holiday",
        "shap_has_transfer", "shap_has_additional", "shap_has_special_event",
        "shap_has_national_event", "shap_has_regional_event", "shap_has_local_event",
        "shap_has_effective_national_event", "shap_has_effective_regional_event",
        "shap_has_effective_local_event", "shap_has_transferred_event",
    ],
    "product": ["shap_product_family", "shap_product_class", "shap_is_perishable"],
    "store": [
        "shap_store_id", "shap_city", "shap_department", "shap_store_type", "shap_store_cluster",
    ],
    "oil_price": ["shap_oil_price", "shap_oil_price_source_missing"],
    "traffic": [
        "shap_traffic_lag_1", "shap_traffic_mean_7", "shap_traffic_mean_28",
        "shap_traffic_ratio_7_28", "shap_traffic_source_missing",
    ],
}


def find_latest_shap_parquet(predictions_dir: Path) -> Path:
    candidates = sorted(predictions_dir.glob("*.shap.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No *.shap.parquet found in {predictions_dir}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shap-parquet", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    shap_path = args.shap_parquet or find_latest_shap_parquet(Path("pipeline/predictions"))
    schema = pl.scan_parquet(shap_path).collect_schema().names()
    bucketed = {col for cols in BUCKETS.values() for col in cols}
    missing = [col for col in bucketed if col not in schema]
    if missing:
        raise ValueError(f"Columns in BUCKETS not found in {shap_path}: {missing}")
    # A shap_<feature> column outside every bucket would silently break the
    # additive property (bucket sums must equal prediction - base_value) —
    # the pipeline's feature set has changed under this script before.
    uncovered = [c for c in schema if c.startswith("shap_") and c not in bucketed]
    if uncovered:
        raise ValueError(f"shap_ columns in {shap_path} not covered by any bucket: {uncovered}")

    # base_value/raw_model_prediction are on the model's own training scale
    # (e.g. log1p) — that's the scale the shap_<feature> columns are
    # additive in. submitted_prediction is on sales-unit scale, after the
    # inverse transform plus postprocessing (negative-value clipping,
    # quantile-crossing correction), so it does NOT sum with base_value —
    # a real run showed a >900-unit gap, not floating-point noise. Export
    # both: base_value/prediction stay in model scale for the additive
    # breakdown, display_base/display_prediction are the sales-unit numbers
    # for anything shown to a reviewer as "the forecast."
    transforms = pl.scan_parquet(shap_path).select("target_transform").unique().collect()["target_transform"].to_list()
    if len(transforms) != 1:
        raise ValueError(f"Expected a single target_transform in {shap_path}, found {transforms}")
    transform = transforms[0]
    if transform == "log1p":
        display_base_expr = pl.col("base_value").exp() - 1
    elif transform in ("none", "identity"):
        display_base_expr = pl.col("base_value")
    else:
        raise ValueError(f"Unhandled target_transform {transform!r} — add inverse-transform handling before trusting display_base")

    # Round to 2dp — this is a visualization, not a re-derivation of the model,
    # and unrounded float32 text roughly doubles the file size for no benefit.
    bucket_exprs = [
        pl.sum_horizontal([pl.col(c) for c in cols]).round(2).alias(f"contrib_{name}")
        for name, cols in BUCKETS.items()
    ]
    out = (
        pl.scan_parquet(shap_path)
        .select(
            pl.col("id"),
            pl.col("submitted_quantile").alias("quantile"),
            pl.col("base_value").round(2),
            pl.col("raw_model_prediction").round(2).alias("prediction"),
            display_base_expr.round(2).alias("display_base"),
            pl.col("submitted_prediction").round(2).alias("display_prediction"),
            *bucket_exprs,
        )
        .collect()
    )

    output_path = args.output or shap_path.with_suffix("").with_suffix(".dashboard.csv")
    out.write_csv(output_path)
    print(f"Wrote {out.height:,} rows to {output_path}")


if __name__ == "__main__":
    main()
