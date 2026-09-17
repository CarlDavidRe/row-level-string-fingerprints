from __future__ import annotations

import csv
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import duckdb

from .config import FingerprintConfig


BLOCK_VALUE_SEPARATOR = chr(0)


def quote_identifier(identifier: str) -> str:
    return chr(34) + identifier.replace(chr(34), chr(34) * 2) + chr(34)


@dataclass(frozen=True)
class FingerprintProbe:
    candidate_partition_ids: frozenset[int]
    width: int = 0
    ones: int = 0


@dataclass(frozen=True)
class FingerprintVersion:
    metadata_version: int
    merge_step: str
    metadata_file: str
    metadata_size_bytes: int
    mean_matrix_rows: float
    ngram_size: int | None
    probe: Callable[[str, str, str], FingerprintProbe]


class NGramGenerator:
    """Generate canonical character n-grams using the original notebook rules."""

    def __init__(self, n: int, ascii_only: bool = True):
        if n <= 0:
            raise ValueError("n must be positive for n-gram generation")
        self.n = n
        self.ascii_only = ascii_only

    def generate(self, text: str) -> list[str]:
        normalized = self.strip_accents(text).lower()
        if len(normalized) < self.n:
            return []
        ngrams = [normalized[index:index + self.n] for index in range(len(normalized) - self.n + 1)]
        return [gram for gram in ngrams if gram.isascii()] if self.ascii_only else ngrams

    @staticmethod
    def strip_accents(text: str) -> str:
        return "".join(
            character
            for character in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(character)
        )


def binary_entropy(successes: int, population: int) -> float:
    if successes <= 0 or successes >= population:
        return 0.0
    probability = successes / population
    return -(
        probability * math.log2(probability)
        + (1 - probability) * math.log2(1 - probability)
    )


class FeatureSelector:
    """Select block n-gram groups for one fingerprint configuration."""

    def __init__(self, config: FingerprintConfig):
        self.config = config

    def candidates(
        self, partition_grams: dict[int, frozenset[str]]
    ) -> tuple[int, dict[str, int]]:
        partition_ids = sorted(partition_grams)
        position = {partition_id: index for index, partition_id in enumerate(partition_ids)}
        presence: dict[str, int] = {}
        for partition_id, grams in partition_grams.items():
            bit = 1 << position[partition_id]
            for gram in grams:
                presence[gram] = presence.get(gram, 0) | bit
        block_count = len(partition_ids)
        if not block_count:
            return 0, {}
        maximum_count = max(1, math.floor(self.config.max_block_frequency * block_count))
        minimum_count = 1
        if self.config.feature_selection_method in {
            "fingerprint_distribution_entropy",
            "fingerprint_distribution_entropy_equivalence_classes",
            "fingerprint_internal_entropy_equivalence_classes",
        }:
            minimum_count = max(1, math.ceil(self.config.min_block_frequency * block_count))
        return block_count, {
            gram: mask
            for gram, mask in sorted(presence.items())
            if mask.bit_count() <= maximum_count
            and (
                self.config.feature_selection_method.endswith("_hamming_clusters")
                or minimum_count <= mask.bit_count()
            )
        }

    @staticmethod
    def local_split(
        block_count: int,
        candidates: dict[str, int],
        limit: int,
        include_zero_score_features: bool = False,
    ) -> list[str]:
        selected: list[str] = []
        groups = [(1 << block_count) - 1]
        while candidates and len(selected) < limit:
            best_feature = None
            best_group_index = -1
            best_score = 0.0
            for gram, gram_mask in candidates.items():
                for group_index, group_mask in enumerate(groups):
                    size = group_mask.bit_count()
                    score = binary_entropy((gram_mask & group_mask).bit_count(), size) * size
                    if score > best_score or (
                        score == best_score
                        and score > 0
                        and (best_feature is None or gram < best_feature)
                    ):
                        best_feature = gram
                        best_group_index = group_index
                        best_score = score
            if best_feature is None:
                if not include_zero_score_features:
                    break
                best_feature = min(candidates)
                best_group_index = 0
            selected.append(best_feature)
            feature_mask = candidates.pop(best_feature)
            parent = groups.pop(best_group_index)
            absent, present = parent & ~feature_mask, parent & feature_mask
            groups.extend(group for group in (absent, present) if group)
        return selected

    @staticmethod
    def distribution_entropy(
        block_count: int,
        candidates: dict[str, int],
        limit: int,
        include_zero_score_features: bool = False,
    ) -> list[str]:
        selected: list[str] = []
        groups = [(1 << block_count) - 1]
        while candidates and len(selected) < limit:
            best_feature = None
            best_score = 0.0
            for gram, gram_mask in candidates.items():
                score = math.fsum(
                    group_mask.bit_count()
                    * binary_entropy(
                        (gram_mask & group_mask).bit_count(), group_mask.bit_count()
                    )
                    for group_mask in groups
                )
                if score > best_score or (
                    score == best_score
                    and score > 0
                    and (best_feature is None or gram < best_feature)
                ):
                    best_feature = gram
                    best_score = score
            if best_feature is None:
                if not include_zero_score_features:
                    break
                best_feature = min(candidates)
            selected.append(best_feature)
            feature_mask = candidates.pop(best_feature)
            next_groups = []
            for group_mask in groups:
                absent, present = group_mask & ~feature_mask, group_mask & feature_mask
                next_groups.extend(group for group in (absent, present) if group)
            groups = next_groups
        return selected

    @staticmethod
    def internal_entropy(
        block_count: int, candidates: dict[str, int], limit: int
    ) -> list[str]:
        selected: list[str] = []
        ones_by_block = [0] * block_count
        while candidates and len(selected) < limit:
            next_width = len(selected) + 1
            blocks_by_ones: dict[int, int] = {}
            for block_index, ones in enumerate(ones_by_block):
                blocks_by_ones[ones] = blocks_by_ones.get(ones, 0) | (1 << block_index)
            best_feature = None
            best_score = -1.0
            for gram, gram_mask in candidates.items():
                if not selected:
                    score = block_count * binary_entropy(gram_mask.bit_count(), block_count)
                else:
                    score = math.fsum(
                        (group_mask.bit_count() - (gram_mask & group_mask).bit_count())
                        * binary_entropy(ones, next_width)
                        + (gram_mask & group_mask).bit_count()
                        * binary_entropy(ones + 1, next_width)
                        for ones, group_mask in blocks_by_ones.items()
                    )
                if score > best_score or (
                    score == best_score and (best_feature is None or gram < best_feature)
                ):
                    best_feature = gram
                    best_score = score
            if best_feature is None:
                break
            selected.append(best_feature)
            feature_mask = candidates.pop(best_feature)
            for block_index in range(block_count):
                ones_by_block[block_index] += (feature_mask >> block_index) & 1
        return selected

    @staticmethod
    def collapse_equivalent(
        candidates: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
        aliases_by_mask: dict[int, list[str]] = {}
        for feature, presence_mask in candidates.items():
            aliases_by_mask.setdefault(presence_mask, []).append(feature)
        representatives = {
            aliases[0]: presence_mask for presence_mask, aliases in aliases_by_mask.items()
        }
        aliases = {values[0]: tuple(values) for values in aliases_by_mask.values()}
        return representatives, aliases

    @staticmethod
    def hamming_clusters(
        block_count: int,
        candidates: dict[str, int],
        cluster_count: int,
        max_iterations: int,
    ) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
        items = sorted(candidates.items())
        cluster_count = min(cluster_count, len(items))
        if not cluster_count:
            return {}, {}
        centers = [items[index * len(items) // cluster_count][1] for index in range(cluster_count)]
        assignments: list[int] = []
        for _ in range(max_iterations):
            next_assignments = [
                min(
                    range(cluster_count),
                    key=lambda index: ((mask ^ centers[index]).bit_count(), index),
                )
                for _, mask in items
            ]
            members: list[list[tuple[str, int]]] = [[] for _ in range(cluster_count)]
            for item, cluster_index in zip(items, next_assignments):
                members[cluster_index].append(item)
            next_centers = []
            for cluster_index, cluster_members in enumerate(members):
                if not cluster_members:
                    next_centers.append(centers[cluster_index])
                    continue
                one_counts = [0] * block_count
                for _, mask in cluster_members:
                    for bit_index in range(block_count):
                        one_counts[bit_index] += (mask >> bit_index) & 1
                center = 0
                previous = centers[cluster_index]
                for bit_index, count in enumerate(one_counts):
                    if 2 * count > len(cluster_members) or (
                        2 * count == len(cluster_members) and (previous >> bit_index) & 1
                    ):
                        center |= 1 << bit_index
                next_centers.append(center)
            if next_assignments == assignments and next_centers == centers:
                break
            assignments, centers = next_assignments, next_centers
        members = [[] for _ in range(cluster_count)]
        for item, cluster_index in zip(items, assignments):
            members[cluster_index].append(item)
        representatives: dict[str, int] = {}
        aliases_by_representative: dict[str, tuple[str, ...]] = {}
        for cluster_members in members:
            if not cluster_members:
                continue
            aliases = tuple(gram for gram, _ in cluster_members)
            cluster_mask = 0
            for _, mask in cluster_members:
                cluster_mask |= mask
            representatives[aliases[0]] = cluster_mask
            aliases_by_representative[aliases[0]] = aliases
        return representatives, aliases_by_representative

    def select(
        self, partition_grams: dict[int, frozenset[str]], limit: int
    ) -> list[tuple[str, ...]]:
        block_count, candidates = self.candidates(partition_grams)
        if not block_count:
            return []
        method = self.config.feature_selection_method
        if method == "local_split_entropy":
            return [(feature,) for feature in self.local_split(block_count, candidates, limit)]
        if method == "fingerprint_distribution_entropy":
            return [
                (feature,)
                for feature in self.distribution_entropy(block_count, candidates, limit)
            ]
        if method.endswith("_equivalence_classes"):
            representatives, aliases = self.collapse_equivalent(candidates)
            selector = self.internal_entropy if method.startswith("fingerprint_internal") else (
                self.local_split if method.startswith("local_split") else self.distribution_entropy
            )
            selected = selector(block_count, representatives, limit)
            return [aliases[feature] for feature in selected]
        if method.endswith("_hamming_clusters"):
            representatives, aliases = self.hamming_clusters(
                block_count,
                candidates,
                self.config.hamming_cluster_count,
                self.config.hamming_cluster_max_iterations,
            )
            selector = self.local_split if method.startswith("local_split") else self.distribution_entropy
            selected = selector(
                block_count, representatives, limit, include_zero_score_features=True
            )
            return [aliases[feature] for feature in selected]
        raise ValueError(f"unknown feature-selection method: {method!r}")


class FingerprintBuilder:
    """Build all nested fingerprint widths for one sweep point."""

    DIAGNOSTIC_COLUMNS = [
        "metadata_version", "metadata_file", "fingerprint_width",
        "feature_selection_method", "hamming_cluster_count", "ngram_size",
        "min_block_frequency", "max_block_frequency", "table_name", "column_name",
        "bit_index", "ngram", "ngrams", "alias_count", "block_presence_count",
        "total_block_count", "block_frequency",
    ]

    def __init__(
        self,
        config: FingerprintConfig,
        block_size_rows: int,
        fingerprint_dir: Path,
        output_dir: Path,
    ):
        self.config = config
        self.block_size_rows = block_size_rows
        self.fingerprint_dir = fingerprint_dir
        self.output_dir = output_dir
        self.ngrams = NGramGenerator(config.ngram_size, config.ascii_only)
        self.selector = FeatureSelector(config)

    def iter_ngrams(self, text: str):
        for gram in self.ngrams.generate(text):
            if BLOCK_VALUE_SEPARATOR not in gram:
                yield gram

    def concatenate_partition_ngrams(
        self, con: duckdb.DuckDBPyConnection, table: str, column: str
    ) -> dict[int, frozenset[str]]:
        cursor = con.execute(
            f"SELECT partition_id, CAST({quote_identifier(column)} AS VARCHAR) "
            f"FROM {quote_identifier(table)} ORDER BY partition_id, rowid"
        )
        partition_grams: dict[int, frozenset[str]] = {}
        active_partition: int | None = None
        values: list[str] = []

        def finish_partition() -> None:
            if active_partition is not None:
                partition_grams[active_partition] = frozenset(
                    self.iter_ngrams(BLOCK_VALUE_SEPARATOR.join(values))
                )

        while True:
            batch = cursor.fetchmany(50_000)
            if not batch:
                break
            for raw_partition, value in batch:
                partition_id = int(raw_partition)
                if active_partition is None:
                    active_partition = partition_id
                elif partition_id != active_partition:
                    finish_partition()
                    active_partition = partition_id
                    values = []
                if value is not None:
                    values.append(str(value))
        finish_partition()
        return partition_grams

    @staticmethod
    def encode_block(grams: frozenset[str], groups: tuple[tuple[str, ...], ...]) -> int:
        return sum(
            1 << index
            for index, aliases in enumerate(groups)
            if any(feature in grams for feature in aliases)
        )

    def encode_query(self, predicate: str, feature_to_bit: dict[str, int]) -> int:
        query_mask = 0
        for feature in frozenset(self.ngrams.generate(predicate)):
            if feature in feature_to_bit:
                query_mask |= 1 << feature_to_bit[feature]
        return query_mask

    @staticmethod
    def payload_size(width: int, profiles: dict[tuple[str, str], dict]) -> int:
        return math.ceil(width / 8) * sum(
            len(profile["block_masks"]) for profile in profiles.values()
        )

    def write_metadata(
        self, path: Path, width: int, profiles: dict[tuple[str, str], dict]
    ) -> None:
        hex_digits = max(1, math.ceil(width / 4))
        diagnostics_enabled = self.config.feature_selection_method != (
            "fingerprint_internal_entropy_equivalence_classes"
        )
        payload = {
            "schema_version": 1,
            "representation": "one_concatenated_infix_mask_per_block",
            "block_size_rows": self.block_size_rows,
            "fingerprint_width": width,
            "fingerprint_bytes_per_block": math.ceil(width / 8),
            "fingerprint_payload_size_bytes": self.payload_size(width, profiles),
            "ngram_size": self.config.ngram_size,
            "feature_selection_method": self.config.feature_selection_method,
            "hamming_cluster_count": self.config.hamming_cluster_count,
            "hamming_cluster_max_iterations": self.config.hamming_cluster_max_iterations,
            "min_block_frequency": self.config.min_block_frequency,
            "max_block_frequency": self.config.max_block_frequency,
            "ascii_only": self.config.ascii_only,
            "normalization": "NFKD-strip-accents-lower",
            "targets": {
                f"{table}.{column}": {
                    "features": list(profile["features"]),
                    "feature_groups": [list(aliases) for aliases in profile["feature_groups"]],
                    **({"feature_diagnostics": [
                        {
                            "bit_index": bit_index,
                            "ngram": feature,
                            "ngrams": list(profile["feature_groups"][bit_index]),
                            "alias_count": len(profile["feature_groups"][bit_index]),
                            "block_presence_count": profile["feature_block_counts"][bit_index],
                            "total_block_count": profile["block_count"],
                            "block_frequency": (
                                profile["feature_block_counts"][bit_index] / profile["block_count"]
                                if profile["block_count"] else 0.0
                            ),
                        }
                        for bit_index, feature in enumerate(profile["features"])
                    ]} if diagnostics_enabled else {}),
                    "blocks": [
                        {
                            "partition_id": partition_id,
                            "mask_hex": format(mask, f"0{hex_digits}x"),
                        }
                        for partition_id, mask in sorted(profile["block_masks"].items())
                    ],
                }
                for (table, column), profile in sorted(profiles.items())
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def build(
        self, con: duckdb.DuckDBPyConnection, targets: set[tuple[str, str]]
    ) -> list[FingerprintVersion]:
        max_width = max(self.config.widths)
        full_profiles: dict[tuple[str, str], dict] = {}
        diagnostics_enabled = self.config.feature_selection_method != (
            "fingerprint_internal_entropy_equivalence_classes"
        )
        for target_index, (table, column) in enumerate(sorted(targets), 1):
            print(f"Building block n-grams [{target_index}/{len(targets)}]: {table}.{column}")
            partition_grams = self.concatenate_partition_ngrams(con, table, column)
            groups = tuple(self.selector.select(partition_grams, max_width))
            features = tuple(aliases[0] for aliases in groups)
            full_profiles[(table, column)] = {
                "features": features,
                "feature_groups": groups,
                "feature_block_counts": (() if not diagnostics_enabled else tuple(
                    sum(any(feature in grams for feature in aliases) for grams in partition_grams.values())
                    for aliases in groups
                )),
                "block_count": len(partition_grams),
                "block_masks": {
                    partition_id: self.encode_block(grams, groups)
                    for partition_id, grams in partition_grams.items()
                },
            }
            print(
                f"  {len(partition_grams):,} blocks; selected {len(features):,}/{max_width} "
                f"features with {self.config.feature_selection_method}"
            )

        selected_count = max(
            (len(profile["features"]) for profile in full_profiles.values()), default=0
        )
        saturated = all(
            len(profile["features"]) < max_width for profile in full_profiles.values()
        )
        version_widths = self.config.widths
        if saturated:
            last_index = next(
                index for index, width in enumerate(self.config.widths) if width >= selected_count
            )
            version_widths = self.config.widths[:last_index + 1]
            print(f"  selection saturated at {selected_count} features; stopping at {version_widths[-1]} bits")

        versions: list[FingerprintVersion] = []
        diagnostic_rows: list[dict] = []
        for version_id, width in enumerate(version_widths):
            width_mask = (1 << width) - 1
            profiles = {
                target: {
                    "features": profile["features"][:width],
                    "feature_groups": profile["feature_groups"][:width],
                    "feature_to_bit": {
                        feature: bit_index
                        for bit_index, aliases in enumerate(profile["feature_groups"][:width])
                        for feature in aliases
                    },
                    "feature_block_counts": profile["feature_block_counts"][:width],
                    "block_count": profile["block_count"],
                    "block_masks": {
                        partition_id: mask & width_mask
                        for partition_id, mask in profile["block_masks"].items()
                    },
                }
                for target, profile in full_profiles.items()
            }
            metadata_path = self.fingerprint_dir / f"block_infix_fingerprint_v{version_id:03d}.json"
            self.write_metadata(metadata_path, width, profiles)
            if diagnostics_enabled:
                for (table, column), profile in sorted(profiles.items()):
                    for bit_index, feature in enumerate(profile["features"]):
                        presence_count = profile["feature_block_counts"][bit_index]
                        diagnostic_rows.append({
                            "metadata_version": version_id,
                            "metadata_file": metadata_path.name,
                            "fingerprint_width": width,
                            "feature_selection_method": self.config.feature_selection_method,
                            "hamming_cluster_count": self.config.hamming_cluster_count,
                            "ngram_size": self.config.ngram_size,
                            "min_block_frequency": self.config.min_block_frequency,
                            "max_block_frequency": self.config.max_block_frequency,
                            "table_name": table,
                            "column_name": column,
                            "bit_index": bit_index,
                            "ngram": feature,
                            "ngrams": json.dumps(profile["feature_groups"][bit_index], ensure_ascii=True),
                            "alias_count": len(profile["feature_groups"][bit_index]),
                            "block_presence_count": presence_count,
                            "total_block_count": profile["block_count"],
                            "block_frequency": presence_count / profile["block_count"] if profile["block_count"] else 0.0,
                        })

            def make_probe(version_profiles, fingerprint_width):
                def probe(table: str, column: str, predicate: str) -> FingerprintProbe:
                    profile = version_profiles[(table, column)]
                    query_mask = self.encode_query(str(predicate or ""), profile["feature_to_bit"])
                    candidates = frozenset(
                        partition_id
                        for partition_id, block_mask in profile["block_masks"].items()
                        if (block_mask & query_mask) == query_mask
                    )
                    return FingerprintProbe(candidates, fingerprint_width, query_mask.bit_count())
                return probe

            versions.append(FingerprintVersion(
                metadata_version=version_id,
                merge_step=(
                    f"concatenated_block_infix_{self.config.feature_selection_method}_{width}bit"
                ),
                metadata_file=metadata_path.name,
                metadata_size_bytes=self.payload_size(width, profiles),
                mean_matrix_rows=1.0,
                ngram_size=self.config.ngram_size,
                probe=make_probe(profiles, width),
            ))

        diagnostics_path = self.output_dir / "selected_ngram_diagnostics.csv"
        if diagnostics_enabled:
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.DIAGNOSTIC_COLUMNS)
                writer.writeheader()
                writer.writerows(diagnostic_rows)
        elif diagnostics_path.exists():
            diagnostics_path.unlink()
        return versions
