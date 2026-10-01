from __future__ import annotations

import csv
import json
import math
import unicodedata
from array import array
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import duckdb
import numpy as np

from .config import FingerprintConfig
from .progress import log_progress
from .storage import feature_mapping_path, write_partition_matrices


BLOCK_VALUE_SEPARATOR = chr(0)
SUBBLOCK_JOINT_ENTROPY_METHOD = (
    "fingerprint_subblock_joint_entropy_equivalence_classes"
)
INTERNAL_ENTROPY_METHOD = "fingerprint_internal_entropy_equivalence_classes"
SCOPE_CONFIGURABLE_METHODS = frozenset({
    INTERNAL_ENTROPY_METHOD,
    SUBBLOCK_JOINT_ENTROPY_METHOD,
})


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
    feature_mapping_file: str | None = None


@dataclass(frozen=True)
class ExtractedNGrams:
    partition_subblocks: dict[int, tuple[frozenset[str], ...]]
    selection_units: dict[int, frozenset[str]] | tuple[frozenset[str], ...]
    frequency_unit: str


class FingerprintDataCache:
    """Bounded cache for extraction work shared by compatible sweep points."""

    def __init__(self) -> None:
        self.signature: tuple[int, bool, int | None] | None = None
        self.targets: dict[tuple[str, str], ExtractedNGrams] = {}

    def get(
        self,
        signature: tuple[int, bool, int | None],
        target: tuple[str, str],
    ) -> ExtractedNGrams | None:
        if signature != self.signature:
            self.signature = signature
            self.targets.clear()
            return None
        return self.targets.get(target)

    def put(
        self,
        signature: tuple[int, bool, int | None],
        target: tuple[str, str],
        extracted: ExtractedNGrams,
    ) -> None:
        if signature != self.signature:
            self.signature = signature
            self.targets.clear()
        self.targets[target] = extracted


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


def prefer_feature_on_tie(
    gram: str, frequency: int, best_feature: str | None, best_frequency: int
) -> bool:
    """Prefer the more common n-gram, then the lexicographically smaller one."""
    return (
        best_feature is None
        or frequency > best_frequency
        or (frequency == best_frequency and gram < best_feature)
    )


class FeatureSelector:
    """Select block n-gram groups for one fingerprint configuration."""

    def __init__(self, config: FingerprintConfig):
        self.config = config
        self._candidate_postings: dict[str, array[int]] = {}

    def candidates(
        self,
        partition_grams: dict[int, frozenset[str]] | tuple[frozenset[str], ...],
    ) -> tuple[int, dict[str, int]]:
        units = (
            (partition_grams[partition_id] for partition_id in sorted(partition_grams))
            if isinstance(partition_grams, dict)
            else iter(partition_grams)
        )
        postings: dict[str, list[int]] = {}
        started_at = perf_counter()
        next_report = 250_000
        for unit_index, grams in enumerate(units):
            for gram in grams:
                postings.setdefault(gram, []).append(unit_index)
            processed = unit_index + 1
            if processed >= next_report:
                log_progress(
                    f"Candidate collection: {processed:,}/{len(partition_grams):,} "
                    f"units; {len(postings):,} distinct n-grams",
                    started_at,
                )
                next_report += 250_000
        block_count = len(partition_grams)
        if not block_count:
            self._candidate_postings = {}
            return 0, {}
        maximum_count = max(1, math.floor(self.config.max_block_frequency * block_count))
        minimum_count = 1
        if self.config.feature_selection_method in {
            "fingerprint_distribution_entropy",
            "fingerprint_distribution_entropy_equivalence_classes",
            "fingerprint_internal_entropy_equivalence_classes",
            SUBBLOCK_JOINT_ENTROPY_METHOD,
        }:
            minimum_count = max(1, math.ceil(self.config.min_block_frequency * block_count))
        filtered: dict[str, int] = {}
        retained_postings: dict[str, array[int]] = {}
        byte_count = math.ceil(block_count / 8)
        postings_typecode = "I" if block_count <= 0xFFFFFFFF else "Q"
        candidate_names = sorted(postings)
        pack_report_every = max(1, math.ceil(len(candidate_names) / 10))
        for candidate_index, gram in enumerate(candidate_names, 1):
            unit_indexes = postings.pop(gram)
            occurrence_count = len(unit_indexes)
            if occurrence_count <= maximum_count and (
                self.config.feature_selection_method.endswith("_hamming_clusters")
                or minimum_count <= occurrence_count
            ):
                packed = bytearray(byte_count)
                for unit_index in unit_indexes:
                    packed[unit_index >> 3] |= 1 << (unit_index & 7)
                filtered[gram] = int.from_bytes(packed, "little")
                retained_postings[gram] = array(postings_typecode, unit_indexes)
            if block_count >= 250_000 and (
                candidate_index % pack_report_every == 0
                or candidate_index == len(candidate_names)
            ):
                log_progress(
                    f"Bitset packing: {candidate_index:,}/{len(candidate_names):,} "
                    f"n-grams; {len(filtered):,} retained",
                    started_at,
                )
        if block_count >= 250_000:
            log_progress(
                f"Candidate collection complete: {len(filtered):,}/{len(candidate_names):,} "
                "n-grams passed frequency filters",
                started_at,
            )
        self._candidate_postings = retained_postings
        return block_count, filtered

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
            best_frequency = -1
            for gram, gram_mask in candidates.items():
                frequency = gram_mask.bit_count()
                for group_index, group_mask in enumerate(groups):
                    size = group_mask.bit_count()
                    score = binary_entropy((gram_mask & group_mask).bit_count(), size) * size
                    if score > best_score or (
                        score == best_score
                        and score > 0
                        and prefer_feature_on_tie(
                            gram, frequency, best_feature, best_frequency
                        )
                    ):
                        best_feature = gram
                        best_group_index = group_index
                        best_score = score
                        best_frequency = frequency
            if best_feature is None:
                if not include_zero_score_features:
                    break
                best_feature = min(
                    candidates, key=lambda gram: (-candidates[gram].bit_count(), gram)
                )
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
            best_frequency = -1
            for gram, gram_mask in candidates.items():
                frequency = gram_mask.bit_count()
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
                    and prefer_feature_on_tie(
                        gram, frequency, best_feature, best_frequency
                    )
                ):
                    best_feature = gram
                    best_score = score
                    best_frequency = frequency
            if best_feature is None:
                if not include_zero_score_features:
                    break
                best_feature = min(
                    candidates, key=lambda gram: (-candidates[gram].bit_count(), gram)
                )
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
        block_count: int,
        candidates: dict[str, int],
        limit: int,
        postings: dict[str, array[int]] | None = None,
    ) -> list[str]:
        if postings:
            return FeatureSelector.native_internal_entropy(
                block_count, candidates, limit, postings
            )
        selected: list[str] = []
        blocks_by_ones = {0: (1 << block_count) - 1}
        progress_step = 1_000 if limit >= 1_000 else 0
        started_at = perf_counter()
        while candidates and len(selected) < limit:
            next_width = len(selected) + 1
            best_feature = None
            best_score = -1.0
            best_frequency = -1
            for gram, gram_mask in candidates.items():
                frequency = gram_mask.bit_count()
                if not selected:
                    score = block_count * binary_entropy(frequency, block_count)
                else:
                    score_terms = []
                    for ones, group_mask in blocks_by_ones.items():
                        group_count = group_mask.bit_count()
                        present_count = (gram_mask & group_mask).bit_count()
                        score_terms.append(
                            (group_count - present_count)
                            * binary_entropy(ones, next_width)
                            + present_count
                            * binary_entropy(ones + 1, next_width)
                        )
                    score = math.fsum(score_terms)
                if score > best_score or (
                    score == best_score
                    and prefer_feature_on_tie(
                        gram, frequency, best_feature, best_frequency
                    )
                ):
                    best_feature = gram
                    best_score = score
                    best_frequency = frequency
            if best_feature is None:
                break
            selected.append(best_feature)
            feature_mask = candidates.pop(best_feature)
            next_blocks_by_ones: dict[int, int] = {}
            for ones, group_mask in blocks_by_ones.items():
                present = group_mask & feature_mask
                absent = group_mask & ~feature_mask
                if absent:
                    next_blocks_by_ones[ones] = (
                        next_blocks_by_ones.get(ones, 0) | absent
                    )
                if present:
                    next_blocks_by_ones[ones + 1] = (
                        next_blocks_by_ones.get(ones + 1, 0) | present
                    )
            blocks_by_ones = next_blocks_by_ones
            if progress_step and (
                len(selected) % progress_step == 0 or len(selected) == limit
            ):
                log_progress(
                    f"Feature selection: {len(selected):,}/{limit:,} selected; "
                    f"{len(candidates):,} candidates remain",
                    started_at,
                )
        return selected

    @staticmethod
    def native_internal_entropy(
        block_count: int,
        candidates: dict[str, int],
        limit: int,
        postings: dict[str, array[int]],
    ) -> list[str]:
        """Score sparse candidate postings in NumPy while preserving exact ties."""
        names = tuple(candidates)
        masks = tuple(candidates.values())
        lengths = np.fromiter(
            (len(postings[name]) for name in names),
            dtype=np.int64,
            count=len(names),
        )
        offsets = np.empty(len(names) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        index_dtype = np.uint32 if block_count <= 0xFFFFFFFF else np.uint64
        unit_indexes = np.empty(int(offsets[-1]), dtype=index_dtype)
        for candidate_index, name in enumerate(names):
            start, stop = int(offsets[candidate_index]), int(offsets[candidate_index + 1])
            unit_indexes[start:stop] = np.frombuffer(
                postings[name], dtype=index_dtype
            )

        count_dtype = np.uint16 if limit <= np.iinfo(np.uint16).max else np.uint32
        ones_by_unit = np.zeros(block_count, dtype=count_dtype)
        active = np.ones(len(names), dtype=np.bool_)
        blocks_by_ones = {0: (1 << block_count) - 1}
        selected: list[str] = []
        progress_step = 100 if limit >= 100 else 0
        started_at = perf_counter()

        while active.any() and len(selected) < limit:
            next_width = len(selected) + 1
            if not selected:
                scores = np.fromiter(
                    (
                        block_count * binary_entropy(int(count), block_count)
                        for count in lengths
                    ),
                    dtype=np.float64,
                    count=len(names),
                )
            else:
                entropies = np.fromiter(
                    (
                        binary_entropy(ones, next_width)
                        for ones in range(next_width + 1)
                    ),
                    dtype=np.float64,
                    count=next_width + 1,
                )
                deltas = entropies[1:] - entropies[:-1]
                occurrence_weights = deltas[ones_by_unit[unit_indexes]]
                scores = np.add.reduceat(occurrence_weights, offsets[:-1])
            scores[~active] = -np.inf
            approximate_best = float(scores.max())
            tolerance = np.finfo(np.float64).eps * max(1, block_count) * 32
            contender_indexes = np.flatnonzero(
                active & (scores >= approximate_best - tolerance)
            )

            best_index = -1
            best_score = -1.0
            best_frequency = -1
            for raw_index in contender_indexes:
                candidate_index = int(raw_index)
                candidate_frequency = int(lengths[candidate_index])
                gram_mask = masks[candidate_index]
                if not selected:
                    exact_score = block_count * binary_entropy(
                        candidate_frequency, block_count
                    )
                else:
                    exact_score = math.fsum(
                        (
                            group_mask.bit_count()
                            - (gram_mask & group_mask).bit_count()
                        )
                        * binary_entropy(ones, next_width)
                        + (gram_mask & group_mask).bit_count()
                        * binary_entropy(ones + 1, next_width)
                        for ones, group_mask in blocks_by_ones.items()
                    )
                if exact_score > best_score or (
                    exact_score == best_score
                    and prefer_feature_on_tie(
                        names[candidate_index], candidate_frequency,
                        names[best_index] if best_index >= 0 else None, best_frequency,
                    )
                ):
                    best_index = candidate_index
                    best_score = exact_score
                    best_frequency = candidate_frequency
            if best_index < 0:
                break

            active[best_index] = False
            selected.append(names[best_index])
            feature_mask = masks[best_index]
            next_blocks_by_ones: dict[int, int] = {}
            for ones, group_mask in blocks_by_ones.items():
                present = group_mask & feature_mask
                absent = group_mask & ~feature_mask
                if absent:
                    next_blocks_by_ones[ones] = (
                        next_blocks_by_ones.get(ones, 0) | absent
                    )
                if present:
                    next_blocks_by_ones[ones + 1] = (
                        next_blocks_by_ones.get(ones + 1, 0) | present
                    )
            blocks_by_ones = next_blocks_by_ones
            start, stop = int(offsets[best_index]), int(offsets[best_index + 1])
            ones_by_unit[unit_indexes[start:stop]] += 1

            if progress_step and (
                len(selected) % progress_step == 0 or len(selected) == limit
            ):
                log_progress(
                    f"Native feature selection: {len(selected):,}/{limit:,} "
                    f"selected; {int(active.sum()):,} candidates remain",
                    started_at,
                )
        return selected

    @staticmethod
    def within_block_joint_entropy(
        block_sizes: tuple[int, ...], candidates: dict[str, int], limit: int
    ) -> list[str]:
        """Maximize entropy of feature occurrence vectors across sub-blocks."""
        selected: list[str] = []
        block_ranges: list[tuple[int, int]] = []
        offset = 0
        for block_size in block_sizes:
            block_ranges.append((offset, (1 << block_size) - 1))
            offset += block_size
        pattern_counts: list[dict[int, int]] = [{} for _ in block_sizes]
        count_log_count_sums = [0.0] * len(block_sizes)
        progress_step = 1_000 if limit >= 1_000 else 0
        started_at = perf_counter()

        while candidates and len(selected) < limit:
            next_width = len(selected) + 1
            best_feature = None
            best_score = -1.0
            best_frequency = -1
            for gram, gram_mask in candidates.items():
                frequency = gram_mask.bit_count()
                if not selected:
                    present_blocks = sum(
                        bool((gram_mask >> block_offset) & local_mask)
                        for block_offset, local_mask in block_ranges
                    )
                    score = len(block_sizes) * binary_entropy(
                        present_blocks, len(block_sizes)
                    )
                else:
                    entropies = []
                    for block_index, (block_offset, local_mask) in enumerate(block_ranges):
                        pattern = (gram_mask >> block_offset) & local_mask
                        current_count = pattern_counts[block_index].get(pattern, 0)
                        next_count = current_count + 1
                        next_count_log_count_sum = count_log_count_sums[block_index]
                        if current_count > 1:
                            next_count_log_count_sum -= (
                                current_count * math.log2(current_count)
                            )
                        if next_count > 1:
                            next_count_log_count_sum += next_count * math.log2(next_count)
                        entropies.append(
                            math.log2(next_width)
                            - next_count_log_count_sum / next_width
                        )
                    score = math.fsum(entropies)
                if score > best_score or (
                    score == best_score
                    and prefer_feature_on_tie(
                        gram, frequency, best_feature, best_frequency
                    )
                ):
                    best_feature = gram
                    best_score = score
                    best_frequency = frequency
            if best_feature is None:
                break
            selected.append(best_feature)
            feature_mask = candidates.pop(best_feature)
            for block_index, (block_offset, local_mask) in enumerate(block_ranges):
                pattern = (feature_mask >> block_offset) & local_mask
                current_count = pattern_counts[block_index].get(pattern, 0)
                next_count = current_count + 1
                if current_count > 1:
                    count_log_count_sums[block_index] -= (
                        current_count * math.log2(current_count)
                    )
                if next_count > 1:
                    count_log_count_sums[block_index] += next_count * math.log2(next_count)
                pattern_counts[block_index][pattern] = next_count
            if progress_step and (
                len(selected) % progress_step == 0 or len(selected) == limit
            ):
                log_progress(
                    f"Feature selection: {len(selected):,}/{limit:,} selected; "
                    f"{len(candidates):,} candidates remain",
                    started_at,
                )
        return selected

    @staticmethod
    def collapse_equivalent(
        candidates: dict[str, int],
    ) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
        aliases_by_mask: dict[int, list[str]] = {}
        for feature in sorted(candidates):
            aliases_by_mask.setdefault(candidates[feature], []).append(feature)
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
        self,
        partition_grams: dict[int, frozenset[str]] | tuple[frozenset[str], ...],
        limit: int,
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
            if method == SUBBLOCK_JOINT_ENTROPY_METHOD:
                raise ValueError(
                    f"{method} requires physical block boundaries"
                )
            if method.startswith("fingerprint_internal"):
                selected = self.internal_entropy(
                    block_count,
                    representatives,
                    limit,
                    {
                        feature: self._candidate_postings[feature]
                        for feature in representatives
                    },
                )
            else:
                selector = (
                    self.local_split
                    if method.startswith("local_split")
                    else self.distribution_entropy
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

    def select_block_local_joint_entropy(
        self,
        partition_subblocks: dict[int, tuple[frozenset[str], ...]],
        limit: int,
    ) -> dict[int, tuple[tuple[str, ...], ...]]:
        units = tuple(
            grams
            for partition_id in sorted(partition_subblocks)
            for grams in partition_subblocks[partition_id]
        )
        _, global_candidates = self.candidates(units)
        global_representatives, global_aliases = self.collapse_equivalent(
            global_candidates
        )
        selected_by_block: dict[int, tuple[tuple[str, ...], ...]] = {}
        offset = 0
        partition_ids = sorted(partition_subblocks)
        started_at = perf_counter()
        for block_index, partition_id in enumerate(partition_ids, 1):
            block_size = len(partition_subblocks[partition_id])
            local_mask = (1 << block_size) - 1
            local_candidates = {
                feature: (presence_mask >> offset) & local_mask
                for feature, presence_mask in global_representatives.items()
            }
            selected = self.within_block_joint_entropy(
                (block_size,), local_candidates, limit
            )
            selected_by_block[partition_id] = tuple(
                global_aliases[feature] for feature in selected
            )
            offset += block_size
            log_progress(
                f"Local feature selection: block {block_index:,}/{len(partition_ids):,} "
                f"selected {len(selected):,} features",
                started_at,
            )
        return selected_by_block

    def select_block_local_internal_entropy(
        self,
        partition_subblocks: dict[int, tuple[frozenset[str], ...]],
        limit: int,
    ) -> dict[int, tuple[tuple[str, ...], ...]]:
        """Select independently using each physical block's sub-blocks as units."""
        selected_by_block: dict[int, tuple[tuple[str, ...], ...]] = {}
        partition_ids = sorted(partition_subblocks)
        started_at = perf_counter()
        for block_index, partition_id in enumerate(partition_ids, 1):
            subblocks = partition_subblocks[partition_id]
            subblock_count, candidates = self.candidates(subblocks)
            representatives, aliases = self.collapse_equivalent(candidates)
            selected = self.internal_entropy(
                subblock_count,
                representatives,
                limit,
                {
                    feature: self._candidate_postings[feature]
                    for feature in representatives
                },
            )
            selected_by_block[partition_id] = tuple(
                aliases[feature] for feature in selected
            )
            log_progress(
                f"Local feature selection: block {block_index:,}/{len(partition_ids):,} "
                f"selected {len(selected):,} features",
                started_at,
            )
        return selected_by_block

    def select_global_joint_entropy(
        self,
        partition_subblocks: dict[int, tuple[frozenset[str], ...]],
        limit: int,
    ) -> list[tuple[str, ...]]:
        units = tuple(
            grams
            for partition_id in sorted(partition_subblocks)
            for grams in partition_subblocks[partition_id]
        )
        _, candidates = self.candidates(units)
        representatives, aliases = self.collapse_equivalent(candidates)
        block_sizes = tuple(
            len(partition_subblocks[partition_id])
            for partition_id in sorted(partition_subblocks)
        )
        selected = (
            self.internal_entropy(
                len(block_sizes),
                representatives,
                limit,
                {
                    feature: self._candidate_postings[feature]
                    for feature in representatives
                },
            )
            if block_sizes and all(block_size == 1 for block_size in block_sizes)
            else self.within_block_joint_entropy(block_sizes, representatives, limit)
        )
        return [aliases[feature] for feature in selected]

    def select_global_subblock_internal_entropy(
        self,
        partition_subblocks: dict[int, tuple[frozenset[str], ...]],
        limit: int,
    ) -> list[tuple[str, ...]]:
        """Treat every sub-block as an independent internal-entropy unit."""
        units = tuple(
            grams
            for partition_id in sorted(partition_subblocks)
            for grams in partition_subblocks[partition_id]
        )
        subblock_count, candidates = self.candidates(units)
        representatives, aliases = self.collapse_equivalent(candidates)
        selected = self.internal_entropy(
            subblock_count,
            representatives,
            limit,
            {
                feature: self._candidate_postings[feature]
                for feature in representatives
            },
        )
        return [aliases[feature] for feature in selected]


class FingerprintBuilder:
    """Build all nested fingerprint widths for one sweep point."""

    DIAGNOSTIC_COLUMNS = [
        "metadata_version", "metadata_file", "fingerprint_width",
        "feature_selection_method", "hamming_cluster_count", "ngram_size",
        "min_block_frequency", "max_block_frequency", "subblock_size_rows",
        "frequency_unit", "table_name", "column_name",
        "bit_index", "ngram", "ngrams", "alias_count", "block_presence_count",
        "total_block_count", "block_frequency",
    ]

    def __init__(
        self,
        config: FingerprintConfig,
        block_size_rows: int,
        fingerprint_dir: Path,
        output_dir: Path,
        data_cache: FingerprintDataCache | None = None,
    ):
        self.config = config
        self.block_size_rows = block_size_rows
        self.fingerprint_dir = fingerprint_dir
        self.output_dir = output_dir
        self.data_cache = data_cache
        self.ngrams = NGramGenerator(config.ngram_size, config.ascii_only)
        self.selector = FeatureSelector(config)
        if (
            config.subblock_size_rows is not None
            and config.subblock_size_rows > block_size_rows
        ):
            raise ValueError(
                "subblock_size_rows must not exceed block_size_rows; "
                f"got {config.subblock_size_rows} for block_size_rows={block_size_rows}"
            )

    @property
    def uses_subblock_matrix(self) -> bool:
        return self.config.subblock_size_rows is not None

    @property
    def uses_block_local_mapping(self) -> bool:
        return (
            self.config.feature_selection_method in SCOPE_CONFIGURABLE_METHODS
            and self.config.feature_selection_scope == "local"
        )

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
        rows_read = 0
        next_report = 500_000
        started_at = perf_counter()

        def finish_partition() -> None:
            if active_partition is not None:
                partition_grams[active_partition] = frozenset(
                    self.iter_ngrams(BLOCK_VALUE_SEPARATOR.join(values))
                )

        while True:
            batch = cursor.fetchmany(50_000)
            if not batch:
                break
            rows_read += len(batch)
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
            if rows_read >= next_report:
                log_progress(
                    f"Database scan: {rows_read:,} rows read; "
                    f"{len(partition_grams):,} blocks completed",
                    started_at,
                )
                next_report += 500_000
        finish_partition()
        log_progress(
            f"Database scan complete: {rows_read:,} rows, "
            f"{len(partition_grams):,} blocks",
            started_at,
        )
        return partition_grams

    def partition_subblock_ngrams(
        self, con: duckdb.DuckDBPyConnection, table: str, column: str
    ) -> dict[int, tuple[frozenset[str], ...]]:
        """Split every physical block into consecutive, row-aligned sub-blocks."""
        if self.config.subblock_size_rows is None:
            raise ValueError("subblock_size_rows is required for sub-block fingerprints")
        cursor = con.execute(
            f"SELECT partition_id, CAST({quote_identifier(column)} AS VARCHAR) "
            f"FROM {quote_identifier(table)} ORDER BY partition_id, rowid"
        )
        result: dict[int, tuple[frozenset[str], ...]] = {}
        active_partition: int | None = None
        rows_in_subblock = 0
        values: list[str] = []
        subblocks: list[frozenset[str]] = []
        rows_read = 0
        subblocks_built = 0
        next_report = 500_000
        started_at = perf_counter()

        def finish_subblock() -> None:
            nonlocal rows_in_subblock, values, subblocks_built
            if rows_in_subblock:
                subblocks.append(frozenset(
                    self.iter_ngrams(BLOCK_VALUE_SEPARATOR.join(values))
                ))
                subblocks_built += 1
                rows_in_subblock = 0
                values = []

        def finish_partition() -> None:
            nonlocal subblocks
            if active_partition is not None:
                finish_subblock()
                result[active_partition] = tuple(subblocks)
                subblocks = []

        while True:
            batch = cursor.fetchmany(50_000)
            if not batch:
                break
            rows_read += len(batch)
            for raw_partition, value in batch:
                partition_id = int(raw_partition)
                if active_partition is None:
                    active_partition = partition_id
                elif partition_id != active_partition:
                    finish_partition()
                    active_partition = partition_id
                if value is not None:
                    values.append(str(value))
                rows_in_subblock += 1
                if rows_in_subblock == self.config.subblock_size_rows:
                    finish_subblock()
            if rows_read >= next_report:
                log_progress(
                    f"Database scan: {rows_read:,} rows read; "
                    f"{len(result):,} blocks and {subblocks_built:,} sub-blocks completed",
                    started_at,
                )
                next_report += 500_000
        finish_partition()
        log_progress(
            f"Database scan complete: {rows_read:,} rows, {len(result):,} blocks, "
            f"{subblocks_built:,} sub-blocks",
            started_at,
        )
        return result

    @staticmethod
    def flatten_subblocks(
        partition_subblocks: dict[int, tuple[frozenset[str], ...]]
    ) -> tuple[frozenset[str], ...]:
        return tuple(
            grams
            for partition_id in sorted(partition_subblocks)
            for grams in partition_subblocks[partition_id]
        )

    def extract_ngrams(
        self,
        con: duckdb.DuckDBPyConnection,
        table: str,
        column: str,
    ) -> ExtractedNGrams:
        signature = (
            self.config.ngram_size,
            self.config.ascii_only,
            self.config.subblock_size_rows,
        )
        target = (table, column)
        cached = (
            self.data_cache.get(signature, target)
            if self.data_cache is not None
            else None
        )
        if cached is not None:
            unit_label = (
                "sub-blocks" if cached.frequency_unit == "subblock" else "blocks"
            )
            log_progress(
                f"Reusing cached n-grams for {table}.{column}: "
                f"{len(cached.selection_units):,} {unit_label}"
            )
            return cached

        if self.uses_subblock_matrix:
            partition_subblocks = self.partition_subblock_ngrams(con, table, column)
            selection_units = self.flatten_subblocks(partition_subblocks)
            frequency_unit = "subblock"
        else:
            partition_grams = self.concatenate_partition_ngrams(con, table, column)
            partition_subblocks = {
                partition_id: (grams,)
                for partition_id, grams in partition_grams.items()
            }
            selection_units = partition_grams
            frequency_unit = "block"
        extracted = ExtractedNGrams(
            partition_subblocks=partition_subblocks,
            selection_units=selection_units,
            frequency_unit=frequency_unit,
        )
        if self.data_cache is not None:
            self.data_cache.put(signature, target, extracted)
        return extracted

    @staticmethod
    def encode_block(grams: frozenset[str], groups: tuple[tuple[str, ...], ...]) -> int:
        return sum(
            1 << index
            for index, aliases in enumerate(groups)
            if any(feature in grams for feature in aliases)
        )

    @staticmethod
    def encode_grams(grams: frozenset[str], feature_to_bit: dict[str, int]) -> int:
        mask = 0
        for feature in grams:
            bit_index = feature_to_bit.get(feature)
            if bit_index is not None:
                mask |= 1 << bit_index
        return mask

    def encode_query(self, predicate: str, feature_to_bit: dict[str, int]) -> int:
        query_mask = 0
        for feature in frozenset(self.ngrams.generate(predicate)):
            if feature in feature_to_bit:
                query_mask |= 1 << feature_to_bit[feature]
        return query_mask

    @staticmethod
    def payload_size(width: int, profiles: dict[tuple[str, str], dict]) -> int:
        return math.ceil(width / 8) * sum(
            sum(len(rows) for rows in profile["block_rows"].values())
            for profile in profiles.values()
        )

    def write_metadata(
        self, path: Path, width: int, profiles: dict[tuple[str, str], dict]
    ) -> int:
        if self.uses_subblock_matrix:
            return write_partition_matrices(
                path, width, profiles, local_mapping=self.uses_block_local_mapping
            )
        hex_digits = max(1, math.ceil(width / 4))
        diagnostics_enabled = (
            not self.uses_block_local_mapping
            and self.config.feature_selection_method != INTERNAL_ENTROPY_METHOD
        )
        representation = (
            "deduplicated_subblock_infix_mask_matrix_per_block"
            if self.uses_subblock_matrix
            else "one_concatenated_infix_mask_per_block"
        )
        metadata_targets = {}
        for (table, column), profile in sorted(profiles.items()):
            blocks = []
            for partition_id, rows in sorted(profile["block_rows"].items()):
                block = {
                    "partition_id": partition_id,
                    **({
                        "matrix_rows_hex": [
                            format(mask, f"0{hex_digits}x") for mask in rows
                        ],
                        "matrix_row_count": len(rows),
                    } if self.uses_subblock_matrix else {
                        "mask_hex": format(rows[0], f"0{hex_digits}x"),
                    }),
                }
                if self.uses_block_local_mapping:
                    groups = profile["block_feature_groups"][partition_id]
                    block["features"] = [aliases[0] for aliases in groups]
                    block["feature_groups"] = [list(aliases) for aliases in groups]
                blocks.append(block)
            target_payload = {
                "feature_scope": (
                    "block_local" if self.uses_block_local_mapping else "target_shared"
                ),
                "features": list(profile["features"]),
                "feature_groups": [
                    list(aliases) for aliases in profile["feature_groups"]
                ],
                "blocks": blocks,
            }
            if diagnostics_enabled:
                target_payload["feature_diagnostics"] = [
                    {
                        "bit_index": bit_index,
                        "ngram": feature,
                        "ngrams": list(profile["feature_groups"][bit_index]),
                        "alias_count": len(profile["feature_groups"][bit_index]),
                        "frequency_unit": profile["frequency_unit"],
                        "presence_count": profile["feature_unit_counts"][bit_index],
                        "total_unit_count": profile["unit_count"],
                        "frequency": (
                            profile["feature_unit_counts"][bit_index] / profile["unit_count"]
                            if profile["unit_count"] else 0.0
                        ),
                        "block_presence_count": profile["feature_unit_counts"][bit_index],
                        "total_block_count": profile["unit_count"],
                        "block_frequency": (
                            profile["feature_unit_counts"][bit_index] / profile["unit_count"]
                            if profile["unit_count"] else 0.0
                        ),
                    }
                    for bit_index, feature in enumerate(profile["features"])
                ]
            metadata_targets[f"{table}.{column}"] = target_payload
        payload = {
            "schema_version": 1,
            "representation": representation,
            "block_size_rows": self.block_size_rows,
            "subblock_size_rows": self.config.subblock_size_rows,
            "subblocks_per_full_block": (
                math.ceil(self.block_size_rows / self.config.subblock_size_rows)
                if self.config.subblock_size_rows is not None else 1
            ),
            "fingerprint_width": width,
            "fingerprint_bytes_per_block": math.ceil(width / 8),
            "fingerprint_bytes_per_matrix_row": math.ceil(width / 8),
            "matrix_row_count": sum(
                len(rows)
                for profile in profiles.values()
                for rows in profile["block_rows"].values()
            ),
            "fingerprint_payload_size_bytes": self.payload_size(width, profiles),
            "ngram_size": self.config.ngram_size,
            "feature_selection_method": self.config.feature_selection_method,
            "feature_scope": (
                "block_local" if self.uses_block_local_mapping else "target_shared"
            ),
            "hamming_cluster_count": self.config.hamming_cluster_count,
            "hamming_cluster_max_iterations": self.config.hamming_cluster_max_iterations,
            "min_block_frequency": self.config.min_block_frequency,
            "max_block_frequency": self.config.max_block_frequency,
            "ascii_only": self.config.ascii_only,
            "normalization": "NFKD-strip-accents-lower",
            "targets": metadata_targets,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return path.stat().st_size

    def build(
        self, con: duckdb.DuckDBPyConnection, targets: set[tuple[str, str]]
    ) -> list[FingerprintVersion]:
        build_started_at = perf_counter()
        max_width = max(self.config.widths)
        full_profiles: dict[tuple[str, str], dict] = {}
        diagnostics_enabled = (
            not self.uses_block_local_mapping
            and self.config.feature_selection_method != INTERNAL_ENTROPY_METHOD
        )
        for target_index, (table, column) in enumerate(sorted(targets), 1):
            target_started_at = perf_counter()
            log_progress(
                f"Building block n-grams [{target_index}/{len(targets)}]: {table}.{column}"
            )
            extracted = self.extract_ngrams(con, table, column)
            partition_subblocks = extracted.partition_subblocks
            selection_units = extracted.selection_units
            frequency_unit = extracted.frequency_unit
            block_rows: dict[int, tuple[int, ...]] = {}
            if self.uses_block_local_mapping:
                log_progress(
                    f"Selecting up to {max_width:,} local features for "
                    f"{len(partition_subblocks):,} blocks",
                    target_started_at,
                )
                block_feature_groups = (
                    self.selector.select_block_local_internal_entropy(
                        partition_subblocks, max_width
                    )
                    if self.config.feature_selection_method == INTERNAL_ENTROPY_METHOD
                    else self.selector.select_block_local_joint_entropy(
                        partition_subblocks, max_width
                    )
                )
                for block_index, (partition_id, subblocks) in enumerate(
                    partition_subblocks.items(), 1
                ):
                    groups = block_feature_groups[partition_id]
                    feature_to_bit = {
                        feature: bit_index
                        for bit_index, aliases in enumerate(groups)
                        for feature in aliases
                    }
                    block_rows[partition_id] = tuple(sorted({
                        self.encode_grams(grams, feature_to_bit)
                        for grams in subblocks
                    }))
                    if (
                        block_index == 1
                        or block_index % 10 == 0
                        or block_index == len(partition_subblocks)
                    ):
                        log_progress(
                            f"Encoded local block {block_index:,}/{len(partition_subblocks):,} "
                            f"({len(groups):,} features, "
                            f"{len(block_rows[partition_id]):,} rows)",
                            target_started_at,
                        )
                features: tuple[str, ...] = ()
                groups: tuple[tuple[str, ...], ...] = ()
                feature_unit_counts: tuple[int, ...] = ()
                selected_counts = [
                    len(groups) for groups in block_feature_groups.values()
                ]
            else:
                unit_label = "sub-blocks" if frequency_unit == "subblock" else "blocks"
                log_progress(
                    f"Selecting up to {max_width:,} shared features from "
                    f"{len(selection_units):,} {unit_label}",
                    target_started_at,
                )
                groups = tuple(
                    (
                        self.selector.select_global_subblock_internal_entropy(
                            partition_subblocks, max_width
                        )
                        if self.config.feature_selection_method == INTERNAL_ENTROPY_METHOD
                        else self.selector.select_global_joint_entropy(
                            partition_subblocks, max_width
                        )
                    )
                    if self.uses_subblock_matrix
                    else self.selector.select(selection_units, max_width)
                )
                features = tuple(aliases[0] for aliases in groups)
                feature_to_bit = {
                    feature: bit_index
                    for bit_index, aliases in enumerate(groups)
                    for feature in aliases
                }
                mutable_feature_unit_counts = [0] * len(groups)
                for partition_id, subblocks in partition_subblocks.items():
                    masks = []
                    for grams in subblocks:
                        mask = self.encode_grams(grams, feature_to_bit)
                        masks.append(mask)
                        remaining = mask
                        while remaining:
                            lowest_bit = remaining & -remaining
                            mutable_feature_unit_counts[
                                lowest_bit.bit_length() - 1
                            ] += 1
                            remaining ^= lowest_bit
                    block_rows[partition_id] = tuple(sorted(set(masks)))
                feature_unit_counts = tuple(mutable_feature_unit_counts)
                block_feature_groups = {}
                selected_counts = [len(features)]
            full_profiles[(table, column)] = {
                "features": features,
                "feature_groups": groups,
                "block_feature_groups": block_feature_groups,
                "feature_unit_counts": feature_unit_counts,
                "unit_count": len(selection_units),
                "frequency_unit": frequency_unit,
                "block_rows": block_rows,
                "selected_count": max(selected_counts, default=0),
            }
            source_row_count = sum(len(rows) for rows in partition_subblocks.values())
            stored_row_count = sum(
                len(rows)
                for rows in full_profiles[(table, column)]["block_rows"].values()
            )
            selection_summary = (
                f"{min(selected_counts, default=0):,}-"
                f"{max(selected_counts, default=0):,}/{max_width} features per block"
                if self.uses_block_local_mapping else
                f"{max(selected_counts, default=0):,}/{max_width} shared features"
            )
            print(
                f"  {len(partition_subblocks):,} blocks; selected {selection_summary} "
                f"with {self.config.feature_selection_method}"
            )
            if self.uses_subblock_matrix:
                print(
                    f"  {source_row_count:,} sub-block rows -> {stored_row_count:,} "
                    "distinct full-width matrix rows"
                )
            log_progress(f"Finished target {table}.{column}", target_started_at)

        selected_count = max(
            (profile["selected_count"] for profile in full_profiles.values()), default=0
        )
        saturated = all(
            profile["selected_count"] < max_width for profile in full_profiles.values()
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
            version_started_at = perf_counter()
            log_progress(
                f"Preparing metadata version {version_id + 1}/{len(version_widths)} "
                f"({width:,} bits)"
            )
            width_mask = (1 << width) - 1
            profiles = {}
            for target, profile in full_profiles.items():
                version_profile = {
                    "features": profile["features"][:width],
                    "feature_groups": profile["feature_groups"][:width],
                    "block_feature_groups": {
                        partition_id: groups[:width]
                        for partition_id, groups in profile["block_feature_groups"].items()
                    },
                    "feature_unit_counts": profile["feature_unit_counts"][:width],
                    "unit_count": profile["unit_count"],
                    "frequency_unit": profile["frequency_unit"],
                    "block_rows": {
                        partition_id: tuple(sorted({
                            mask & width_mask for mask in rows
                        }))
                        for partition_id, rows in profile["block_rows"].items()
                    },
                }
                if self.uses_block_local_mapping:
                    version_profile["block_feature_to_bit"] = {
                        partition_id: {
                            feature: bit_index
                            for bit_index, aliases in enumerate(groups)
                            for feature in aliases
                        }
                        for partition_id, groups in version_profile[
                            "block_feature_groups"
                        ].items()
                    }
                else:
                    version_profile["feature_to_bit"] = {
                        feature: bit_index
                        for bit_index, aliases in enumerate(
                            version_profile["feature_groups"]
                        )
                        for feature in aliases
                    }
                profiles[target] = version_profile
            metadata_path = self.fingerprint_dir / (
                f"partition_metadata_v{version_id:03d}.parquet"
                if self.uses_subblock_matrix else
                f"block_infix_fingerprint_v{version_id:03d}.json"
            )
            metadata_size_bytes = self.write_metadata(metadata_path, width, profiles)
            log_progress(
                f"Wrote {metadata_path} "
                f"({metadata_size_bytes / 2**20:,.2f} MiB on disk)",
                version_started_at,
            )
            if diagnostics_enabled:
                for (table, column), profile in sorted(profiles.items()):
                    for bit_index, feature in enumerate(profile["features"]):
                        presence_count = profile["feature_unit_counts"][bit_index]
                        diagnostic_rows.append({
                            "metadata_version": version_id,
                            "metadata_file": metadata_path.name,
                            "fingerprint_width": width,
                            "feature_selection_method": self.config.feature_selection_method,
                            "hamming_cluster_count": self.config.hamming_cluster_count,
                            "ngram_size": self.config.ngram_size,
                            "min_block_frequency": self.config.min_block_frequency,
                            "max_block_frequency": self.config.max_block_frequency,
                            "subblock_size_rows": self.config.subblock_size_rows or "",
                            "frequency_unit": profile["frequency_unit"],
                            "table_name": table,
                            "column_name": column,
                            "bit_index": bit_index,
                            "ngram": feature,
                            "ngrams": json.dumps(profile["feature_groups"][bit_index], ensure_ascii=True),
                            "alias_count": len(profile["feature_groups"][bit_index]),
                            "block_presence_count": presence_count,
                            "total_block_count": profile["unit_count"],
                            "block_frequency": presence_count / profile["unit_count"] if profile["unit_count"] else 0.0,
                        })

            def make_probe(version_profiles, fingerprint_width):
                def probe(table: str, column: str, predicate: str) -> FingerprintProbe:
                    profile = version_profiles[(table, column)]
                    query_text = str(predicate or "")
                    query_ones = []
                    candidates = set()
                    for partition_id, matrix_rows in profile["block_rows"].items():
                        feature_to_bit = (
                            profile["block_feature_to_bit"][partition_id]
                            if self.uses_block_local_mapping
                            else profile["feature_to_bit"]
                        )
                        query_mask = self.encode_query(query_text, feature_to_bit)
                        query_ones.append(query_mask.bit_count())
                        if any(
                            (block_mask & query_mask) == query_mask
                            for block_mask in matrix_rows
                        ):
                            candidates.add(partition_id)
                    return FingerprintProbe(
                        frozenset(candidates),
                        fingerprint_width,
                        max(query_ones, default=0),
                    )
                return probe

            total_matrix_rows = sum(
                len(rows)
                for profile in profiles.values()
                for rows in profile["block_rows"].values()
            )
            total_blocks = sum(
                len(profile["block_rows"]) for profile in profiles.values()
            )
            versions.append(FingerprintVersion(
                metadata_version=version_id,
                merge_step=(
                    (
                        f"subblock_matrix_{self.config.subblock_size_rows}row_"
                        if self.uses_subblock_matrix else "concatenated_block_infix_"
                    )
                    + f"{self.config.feature_selection_method}_{width}bit"
                ),
                metadata_file=metadata_path.name,
                metadata_size_bytes=metadata_size_bytes,
                mean_matrix_rows=(
                    total_matrix_rows / total_blocks if total_blocks else 0.0
                ),
                ngram_size=self.config.ngram_size,
                probe=make_probe(profiles, width),
                feature_mapping_file=(
                    feature_mapping_path(metadata_path).name
                    if self.uses_subblock_matrix else None
                ),
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
        log_progress(
            f"Fingerprint build complete: {len(versions)} metadata version(s)",
            build_started_at,
        )
        return versions
