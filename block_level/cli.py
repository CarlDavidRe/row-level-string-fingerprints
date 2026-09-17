from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import SweepRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m block_level",
        description="Run a block-level string-fingerprint parameter sweep.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="path to a JSON experiment configuration",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the config and its input paths without running the sweep",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.load(args.config)
    if args.validate_only:
        config.validate_inputs()
        points = sum(1 for _ in config.sweep.experiments())
        print(f"Valid configuration: {points} sweep configuration(s)")
        return 0
    summary_path = SweepRunner(config).run()
    print(f"Sweep complete: {summary_path}")
    return 0
