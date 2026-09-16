#!/usr/bin/env python3
"""Write a per-fingerprint table of zero-bit queries and feature-group sizes."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "configuration",
    "ngram_size",
    "min_block_frequency",
    "max_block_frequency",
    "fingerprint_width",
    "average_ngrams_per_selected_bit",
    "query_count",
    "zero_bit_query_count",
]


def average_ngrams_per_selected_bit(manifest_path: Path, fingerprint_width: int) -> float:
    """Return the mean feature-group size for one configuration and bit width."""
    diagnostics_path = manifest_path.parent / "block_skipping_results" / "selected_ngram_diagnostics.csv"
    if not diagnostics_path.exists():
        return float("nan")

    alias_counts = []
    with diagnostics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["fingerprint_width"]) == fingerprint_width:
                alias_counts.append(int(row["alias_count"]))
    if not alias_counts:
        return float("nan")
    return sum(alias_counts) / len(alias_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path, help="Directory containing experiment_manifest.json files")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <sweep_dir>/zero_bit_queries_by_config.csv)",
    )
    args = parser.parse_args()

    output_path = args.output or args.sweep_dir / "zero_bit_queries_by_config.csv"
    rows = []
    for manifest_path in sorted(args.sweep_dir.rglob("experiment_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        configuration = str(manifest_path.parent.relative_to(args.sweep_dir))
        for point in manifest["points"]:
            parameters = point["parameters"]
            metrics = point["metrics"]
            fingerprint_width = parameters["fingerprint_width"]
            rows.append({
                "configuration": configuration,
                "ngram_size": parameters["ngram_size"],
                "min_block_frequency": parameters["min_block_frequency"],
                "max_block_frequency": parameters["max_block_frequency"],
                "fingerprint_width": fingerprint_width,
                "average_ngrams_per_selected_bit": average_ngrams_per_selected_bit(
                    manifest_path, fingerprint_width
                ),
                "query_count": metrics["query_count"],
                "zero_bit_query_count": metrics["zero_bit_query_count"],
            })

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
