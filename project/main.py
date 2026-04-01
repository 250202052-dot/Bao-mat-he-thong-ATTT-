from __future__ import annotations

import argparse
from pathlib import Path

from src.runner import run_benchmark_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run imbalanced IDS benchmark project.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper_style_config.yaml"),
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark_project(args.config.resolve())


if __name__ == "__main__":
    main()
