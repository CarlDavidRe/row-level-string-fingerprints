from __future__ import annotations

import argparse
import sys
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    configs = ExperimentConfig.load_many(args.config)
    for config in configs:
        config.validate_inputs()
    if args.validate_only:
        for config in configs:
            points = sum(1 for _ in config.sweep.experiments())
            print(f"{config.workload_name}: {points} sweep configuration(s)")
        print(f"Valid configuration: {len(configs)} workload(s)")
        return 0
    for index, config in enumerate(configs, 1):
        print(f"\n=== Workload {index}/{len(configs)}: {config.workload_name} ===")
        summary_path = SweepRunner(config).run()
        print(f"Sweep complete: {summary_path}")
    return 0
