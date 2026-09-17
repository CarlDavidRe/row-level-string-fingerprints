#!/usr/bin/env python3
"""Write zero-bit query and selected-alias counts for each fingerprint."""

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
    "selected_aliases",
    "query_count",
    "zero_bit_query_count",
]


def selected_aliases(metadata_path: Path) -> int:
    """Count the n-gram aliases represented by one fingerprint version."""
    metadata = json.loads(metadata_path.read_text())
    return sum(
        len(group)
        for target in metadata["targets"].values()
        for group in target.get(
            "feature_groups", [[feature] for feature in target.get("features", [])]
        )
    )


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
            metadata_file = (
                point["id"].replace("block-infix-", "block_infix_fingerprint_")
                + ".json"
            )
            rows.append({
                "configuration": configuration,
                "ngram_size": parameters["ngram_size"],
                "min_block_frequency": parameters["min_block_frequency"],
                "max_block_frequency": parameters["max_block_frequency"],
                "fingerprint_width": fingerprint_width,
                "selected_aliases": selected_aliases(
                    manifest_path.parent / "block_infix_fingerprints" / metadata_file
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
