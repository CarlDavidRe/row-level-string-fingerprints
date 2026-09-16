#!/usr/bin/env python3
"""Write a per-fingerprint table of skipped blocks from a sweep directory."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "configuration",
    "point_id",
    "fingerprint_width",
    "ngram_size",
    "min_block_frequency",
    "max_block_frequency",
    "metadata_file",
    "blocks_skipped",
    "skipped_partitions",
    "query_count",
]


def load_skipped_blocks(manifest_path: Path) -> dict[str, int]:
    """Return benchmark block-skip totals keyed by metadata filename."""
    summary_path = manifest_path.parent / "block_skipping_results" / "metrics_summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open(newline="") as handle:
        return {
            row["result_file"]: int(row["total_num_skipped_blocks"])
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path, help="Directory containing experiment_manifest.json files")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <sweep_dir>/skipped_blocks_by_config.csv)",
    )
    args = parser.parse_args()

    output_path = args.output or args.sweep_dir / "skipped_blocks_by_config.csv"
    rows = []
    for manifest_path in sorted(args.sweep_dir.rglob("experiment_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        benchmark_totals = load_skipped_blocks(manifest_path)
        configuration = manifest_path.parent.relative_to(args.sweep_dir)
        for point in manifest["points"]:
            metrics = point["metrics"]
            parameters = point["parameters"]
            # Point ids such as ``block-infix-v003`` correspond to the benchmark
            # result file ``block_infix_fingerprint_v003.json``.
            result_file = point["id"].replace("block-infix-", "block_infix_fingerprint_") + ".json"
            rows.append({
                "configuration": str(configuration),
                "point_id": point["id"],
                "fingerprint_width": parameters["fingerprint_width"],
                "ngram_size": parameters["ngram_size"],
                "min_block_frequency": parameters["min_block_frequency"],
                "max_block_frequency": parameters["max_block_frequency"],
                "metadata_file": result_file,
                "blocks_skipped": benchmark_totals.get(result_file, metrics["total_skipped_partitions"]),
                "skipped_partitions": metrics["total_skipped_partitions"],
                "query_count": metrics["query_count"],
            })

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
