"""Read legacy query rows and structured block-skipping JSON lines."""

from __future__ import annotations

import csv
import json
from pathlib import Path


QUERY_FORMAT = "block-skipping-query-v1"
PREDICATE_MODES = frozenset({"individual_needle", "all_needles", "actual_predicate"})


def parse_query_file(path: Path) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON query in {path}:{line_number}: {error}") from error
                if not isinstance(payload, dict) or payload.get("format") != QUERY_FORMAT:
                    raise ValueError(
                        f"unsupported structured query in {path}:{line_number}"
                    )
                mode = payload.get("predicate_mode")
                predicate = payload.get("predicate")
                needles = payload.get("lookup_needles")
                if mode not in PREDICATE_MODES or not isinstance(predicate, str) or not predicate:
                    raise ValueError(f"invalid predicate in {path}:{line_number}")
                if not isinstance(needles, list) or not all(
                    isinstance(needle, str) for needle in needles
                ):
                    raise ValueError(f"invalid lookup_needles in {path}:{line_number}")
                # The full SQL predicate is authoritative for ground truth. Only
                # all/individual modes guarantee that every lookup needle is required.
                lookup_needles = tuple(needles) if mode != "actual_predicate" else ()
                selectivity = payload.get("selectivity")
                rows.append((line_number, {
                    "predicate_value": (
                        lookup_needles[0] if len(lookup_needles) == 1 else predicate
                    ),
                    "predicate_mode": mode,
                    "predicate_sql": predicate,
                    "lookup_needles": lookup_needles,
                    "expected_matching_rows": (
                        str(payload["row_count"]) if payload.get("row_count") is not None else ""
                    ),
                    "expected_matching_percent": (
                        f"{float(selectivity) * 100:.8f}%"
                        if selectivity is not None else ""
                    ),
                }))
            else:
                legacy = next(csv.reader([line]))
                if not legacy:
                    continue
                rows.append((line_number, {
                    "predicate_value": legacy[0],
                    "predicate_mode": "individual_needle",
                    "predicate_sql": None,
                    "lookup_needles": (legacy[0],),
                    "expected_matching_rows": legacy[1] if len(legacy) > 1 else "",
                    "expected_matching_percent": legacy[2] if len(legacy) > 2 else "",
                }))
    return rows
