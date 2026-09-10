"""Command-line orchestration for the three forecasting stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cache import run_cached_stage
from .features import FEATURE_GROUPS, parse_feature_groups
from .models import MODELS
from .stages.data_processing import build_merged_data, source_paths
from .stages.feature_engineering import build_training_features
from .stages.model_training import train_and_evaluate


PACKAGE_DIR = Path(__file__).resolve().parent
CHALLENGE_DIR = PACKAGE_DIR.parent
DATA_PROCESSING_DIR = CHALLENGE_DIR / "data_processing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cache-aware forecasting training pipeline."
    )
    parser.add_argument(
        "stage", nargs="?", choices=("all", "data", "features", "train"), default="all"
    )
    parser.add_argument("--data-dir", type=Path, default=CHALLENGE_DIR / "data")
    parser.add_argument(
        "--cache-dir", type=Path, default=CHALLENGE_DIR / ".cache" / "pipeline"
    )
    parser.add_argument(
        "--features",
        default="all",
        help=f"Comma-separated groups or all. Available: {', '.join(FEATURE_GROUPS)}",
    )
    parser.add_argument("--model", choices=tuple(MODELS), default="lightgbm")
    parser.add_argument("--holdout-days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Ignore matching stage caches")
    return parser.parse_args()


def _display(stage: str, hit: bool, metadata: dict[str, Any]) -> None:
    state = "cache hit" if hit else "built"
    print(f"{stage}: {state}")
    print(json.dumps(metadata, indent=2, sort_keys=True, default=str))


def run_pipeline(args: argparse.Namespace) -> None:
    groups = parse_feature_groups(args.features)
    cache_dir = args.cache_dir.resolve()
    merged_path = cache_dir / "stage1_merged.parquet"
    features_path = cache_dir / "stage2_training_features.parquet"
    report_path = cache_dir / f"stage3_{args.model}_report.json"
    cache_code = PACKAGE_DIR / "cache.py"

    hit, metadata = run_cached_stage(
        stage="stage1_data_processing",
        output_path=merged_path,
        manifest_path=cache_dir / "stage1_manifest.json",
        inputs=source_paths(args.data_dir),
        code_files=[
            cache_code,
            DATA_PROCESSING_DIR / "cli.py",
            DATA_PROCESSING_DIR / "config.py",
            DATA_PROCESSING_DIR / "context.py",
            DATA_PROCESSING_DIR / "sources.py",
            DATA_PROCESSING_DIR / "validation.py",
            PACKAGE_DIR / "stages" / "data_processing.py",
        ],
        parameters={"compression": "zstd"},
        build=lambda output: build_merged_data(args.data_dir, output),
        force=args.force,
    )
    _display("Stage 1 data processing", hit, metadata)

    if args.stage in ("all", "features", "train"):
        hit, metadata = run_cached_stage(
            stage="stage2_feature_engineering",
            output_path=features_path,
            manifest_path=cache_dir / "stage2_manifest.json",
            inputs=[merged_path],
            code_files=[
                cache_code,
                PACKAGE_DIR / "features.py",
                PACKAGE_DIR / "stages" / "feature_engineering.py",
            ],
            parameters={"compression": "zstd", "feature_groups": groups},
            build=lambda output: build_training_features(merged_path, output, groups),
            force=args.force,
        )
        _display("Stage 2 feature engineering", hit, metadata)
    if args.stage in ("all", "train"):
        hit, metadata = run_cached_stage(
            stage="stage3_model_training",
            output_path=report_path,
            manifest_path=cache_dir / f"stage3_{args.model}_manifest.json",
            inputs=[features_path],
            code_files=[
                cache_code,
                PACKAGE_DIR / "models.py",
                PACKAGE_DIR / "stages" / "model_training.py",
            ],
            parameters={
                "feature_groups": groups,
                "holdout_days": args.holdout_days,
                "model": args.model,
                "seed": args.seed,
            },
            build=lambda output: train_and_evaluate(
                features_path, output, args.model, args.holdout_days, args.seed
            ),
            force=args.force,
        )
        _display("Stage 3 model training", hit, metadata)


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
