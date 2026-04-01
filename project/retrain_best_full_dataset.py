from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.data_loader import load_dataset
from src.evaluation import build_training_pipeline, derive_resampling_summary
from src.models import get_model_specs
from src.utils import ensure_dir, load_config, save_json, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the best discovered model on the full dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper_style_config.yaml"),
        help="Benchmark config file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cic_unsw",
        help="Dataset key in config to retrain on full data.",
    )
    parser.add_argument(
        "--best-summary",
        type=Path,
        default=None,
        help="Optional path to best_model_summary.json. Defaults to the benchmark output for the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("full_retrain_outputs"),
        help="Directory to save the full-dataset retrained model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model_name instead of using the best summary.",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default=None,
        help="Override sampling_strategy instead of using the best summary.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override alert threshold instead of reusing the saved best threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parent
    config = load_config(config_path)
    random_seed = int(config["random_seed"])
    set_random_seed(random_seed)

    if args.dataset not in config["datasets"]:
        raise KeyError(f"Dataset '{args.dataset}' not found in config.")

    dataset_cfg = dict(config["datasets"][args.dataset])
    dataset_cfg["max_rows"] = None
    dataset_cfg["sample_frac"] = None

    default_best_summary = (
        project_root
        / config["output_root"]
        / args.dataset
        / "best_model_summary.json"
    )
    best_summary_path = args.best_summary.resolve() if args.best_summary else default_best_summary
    if not best_summary_path.exists():
        raise FileNotFoundError(f"Best summary not found: {best_summary_path}")

    best_summary = load_config(best_summary_path)
    model_name = args.model_name or best_summary["model_name"]
    sampling_strategy = args.sampling_strategy or best_summary["sampling_strategy"]
    threshold = float(args.threshold) if args.threshold is not None else float(best_summary["threshold"])

    bundle = load_dataset(args.dataset, dataset_cfg, random_seed)
    model_specs = get_model_specs(random_seed)
    if model_name not in model_specs:
        raise KeyError(f"Model '{model_name}' is not available in current environment.")

    pipeline = build_training_pipeline(
        model_spec=model_specs[model_name],
        strategy_name=sampling_strategy,
        X_template=bundle.features,
        y_train=bundle.target,
        random_seed=random_seed,
    )

    pipeline.fit(bundle.features, bundle.target)
    resampling_summary = derive_resampling_summary(pipeline, bundle.features, bundle.target)

    output_dir = ensure_dir((project_root / args.output_dir / args.dataset).resolve())
    safe_name = f"{model_name}__{sampling_strategy}".replace("/", "_")
    model_path = output_dir / f"{safe_name}__full_dataset.joblib"

    metadata = dict(best_summary)
    metadata.update(
        {
            "dataset_name": args.dataset,
            "rows_trained_full_dataset": int(bundle.features.shape[0]),
            "full_dataset_retrain": True,
        }
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": threshold,
            "metadata": metadata,
        },
        model_path,
    )

    save_json(metadata, output_dir / f"{safe_name}__full_dataset_summary.json")
    save_json(bundle.stats, output_dir / "dataset_stats_full.json")
    if resampling_summary is not None:
        save_json(resampling_summary, output_dir / "resampling_summary_full.json")

    print(f"Dataset: {args.dataset}")
    print(f"Model: {model_name}")
    print(f"Strategy: {sampling_strategy}")
    print(f"Threshold reused from benchmark: {threshold:.6f}")
    print(f"Rows used for full retrain: {bundle.features.shape[0]}")
    print(f"Saved full retrain bundle to: {model_path}")


if __name__ == "__main__":
    main()
