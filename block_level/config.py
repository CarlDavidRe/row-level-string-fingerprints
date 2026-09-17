from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


FEATURE_SELECTION_METHODS = frozenset({
    "local_split_entropy",
    "local_split_entropy_equivalence_classes",
    "fingerprint_distribution_entropy",
    "fingerprint_distribution_entropy_equivalence_classes",
    "fingerprint_internal_entropy_equivalence_classes",
    "local_split_entropy_hamming_clusters",
    "fingerprint_distribution_entropy_hamming_clusters",
})


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return tuple(value)


@dataclass(frozen=True)
class FingerprintConfig:
    widths: tuple[int, ...]
    ngram_size: int
    feature_selection_method: str
    min_block_frequency: float = 0.0
    max_block_frequency: float = 1.0
    ascii_only: bool = True
    hamming_cluster_count: int = 154
    hamming_cluster_max_iterations: int = 10

    def __post_init__(self) -> None:
        if not self.widths or tuple(sorted(set(self.widths))) != self.widths:
            raise ValueError("fingerprint widths must be non-empty, unique, and increasing")
        if any(width <= 0 for width in self.widths):
            raise ValueError("fingerprint widths must be positive")
        if self.ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        if self.feature_selection_method not in FEATURE_SELECTION_METHODS:
            raise ValueError(
                f"unknown feature-selection method: {self.feature_selection_method!r}"
            )
        if not 0 <= self.min_block_frequency <= self.max_block_frequency <= 1:
            raise ValueError("block frequencies must satisfy 0 <= min <= max <= 1")
        if self.hamming_cluster_count <= 0:
            raise ValueError("hamming_cluster_count must be positive")
        if self.hamming_cluster_max_iterations <= 0:
            raise ValueError("hamming_cluster_max_iterations must be positive")


@dataclass(frozen=True)
class SweepConfig:
    widths_by_method: Mapping[str, tuple[int, ...]]
    feature_selection_methods: tuple[str, ...]
    ngram_sizes: tuple[int, ...]
    min_block_frequencies_by_method: Mapping[str, tuple[float, ...]]
    max_block_frequencies: tuple[float, ...] = (1.0,)
    hamming_cluster_counts: tuple[int, ...] = (154,)
    ascii_only: bool = True
    hamming_cluster_max_iterations: int = 10

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SweepConfig:
        methods = tuple(str(value) for value in _sequence(
            raw.get("feature_selection_methods"), "sweep.feature_selection_methods"
        ))
        widths_raw = raw.get("widths_by_method")
        minimums_raw = raw.get("min_block_frequencies_by_method")
        if not isinstance(widths_raw, Mapping):
            raise ValueError("sweep.widths_by_method must be an object")
        if not isinstance(minimums_raw, Mapping):
            raise ValueError("sweep.min_block_frequencies_by_method must be an object")
        return cls(
            widths_by_method={
                str(method): tuple(int(value) for value in _sequence(widths, f"widths for {method}"))
                for method, widths in widths_raw.items()
            },
            feature_selection_methods=methods,
            ngram_sizes=tuple(int(value) for value in _sequence(
                raw.get("ngram_sizes"), "sweep.ngram_sizes"
            )),
            min_block_frequencies_by_method={
                str(method): tuple(float(value) for value in _sequence(values, f"minimums for {method}"))
                for method, values in minimums_raw.items()
            },
            max_block_frequencies=tuple(float(value) for value in _sequence(
                raw.get("max_block_frequencies", [1.0]), "sweep.max_block_frequencies"
            )),
            hamming_cluster_counts=tuple(int(value) for value in _sequence(
                raw.get("hamming_cluster_counts", [154]), "sweep.hamming_cluster_counts"
            )),
            ascii_only=bool(raw.get("ascii_only", True)),
            hamming_cluster_max_iterations=int(raw.get("hamming_cluster_max_iterations", 10)),
        )

    def __post_init__(self) -> None:
        unknown = set(self.feature_selection_methods) - FEATURE_SELECTION_METHODS
        if unknown:
            raise ValueError(f"unknown feature-selection methods: {sorted(unknown)}")
        missing_widths = set(self.feature_selection_methods) - set(self.widths_by_method)
        missing_minimums = set(self.feature_selection_methods) - set(
            self.min_block_frequencies_by_method
        )
        if missing_widths or missing_minimums:
            raise ValueError(
                f"sweep methods missing widths={sorted(missing_widths)} or "
                f"minimum frequencies={sorted(missing_minimums)}"
            )
        if any(size <= 0 for size in self.ngram_sizes):
            raise ValueError("ngram sizes must be positive")
        if any(not 0 < value <= 1 for value in self.max_block_frequencies):
            raise ValueError("maximum block frequencies must be in (0, 1]")
        if any(count <= 0 for count in self.hamming_cluster_counts):
            raise ValueError("Hamming cluster counts must be positive")
        for method in self.feature_selection_methods:
            widths = self.widths_by_method[method]
            if tuple(sorted(set(widths))) != tuple(widths) or any(value <= 0 for value in widths):
                raise ValueError(f"widths for {method} must be positive, unique, and increasing")
            if any(not 0 <= value <= 1 for value in self.min_block_frequencies_by_method[method]):
                raise ValueError(f"minimum block frequencies for {method} must be in [0, 1]")

    def experiments(self) -> Iterator[FingerprintConfig]:
        for method in self.feature_selection_methods:
            maximums = (1.0,) if method.endswith("_hamming_clusters") else self.max_block_frequencies
            cluster_counts = (
                self.hamming_cluster_counts if method.endswith("_hamming_clusters") else (154,)
            )
            for ngram_size in self.ngram_sizes:
                for minimum in self.min_block_frequencies_by_method[method]:
                    for maximum in maximums:
                        for cluster_count in cluster_counts:
                            if minimum <= maximum:
                                yield FingerprintConfig(
                                    widths=tuple(self.widths_by_method[method]),
                                    ngram_size=ngram_size,
                                    feature_selection_method=method,
                                    min_block_frequency=minimum,
                                    max_block_frequency=maximum,
                                    ascii_only=self.ascii_only,
                                    hamming_cluster_count=cluster_count,
                                    hamming_cluster_max_iterations=self.hamming_cluster_max_iterations,
                                )


@dataclass(frozen=True)
class ExperimentConfig:
    database_path: Path
    query_source_dir: Path
    output_dir: Path
    sweep: SweepConfig
    block_size_rows: int = 16_384
    workload_name: str = "ceb_imdb"
    targets: frozenset[tuple[str, str]] | None = None
    query_limit: int | None = None
    partition_database: bool = False
    source_path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        source_path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {source_path}: {error}") from error
        if not isinstance(raw, Mapping):
            raise ValueError("experiment config must contain a JSON object")
        base_dir = source_path.parent

        def resolve(value: str | Path) -> Path:
            candidate = Path(value).expanduser()
            return (base_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

        if "database_path" not in raw:
            raise ValueError("database_path is required")
        database_path = resolve(str(raw["database_path"]))
        output_value = raw.get("output_dir")
        if output_value is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            output_dir = base_dir / f"ceb_imdb_block_skipping_sweep_{run_id}"
        else:
            output_dir = resolve(str(output_value))
        query_source = resolve(str(raw.get("query_source_dir", database_path.parent)))
        targets_raw = raw.get("targets")
        targets: frozenset[tuple[str, str]] | None = None
        if targets_raw is not None:
            parsed = set()
            for target in _sequence(targets_raw, "targets"):
                if isinstance(target, str):
                    table, separator, column = target.rpartition(".")
                    if not separator:
                        raise ValueError(f"target must be table.column: {target!r}")
                elif isinstance(target, (list, tuple)) and len(target) == 2:
                    table, column = map(str, target)
                else:
                    raise ValueError("each target must be 'table.column' or [table, column]")
                parsed.add((table, column))
            targets = frozenset(parsed)
        limit_raw = raw.get("query_limit")
        return cls(
            database_path=database_path,
            query_source_dir=query_source,
            output_dir=output_dir,
            sweep=SweepConfig.from_mapping(raw.get("sweep", {})),
            block_size_rows=int(raw.get("block_size_rows", 16_384)),
            workload_name=str(raw.get("workload_name", "ceb_imdb")),
            targets=targets,
            query_limit=None if limit_raw is None else int(limit_raw),
            partition_database=bool(raw.get("partition_database", False)),
            source_path=source_path,
        )

    def __post_init__(self) -> None:
        if self.block_size_rows <= 0:
            raise ValueError("block_size_rows must be positive")
        if self.query_limit is not None and self.query_limit <= 0:
            raise ValueError("query_limit must be positive or null")
        if not self.workload_name:
            raise ValueError("workload_name must not be empty")

    def validate_inputs(self) -> None:
        if not self.database_path.is_file():
            raise FileNotFoundError(f"DuckDB database not found: {self.database_path}")
        if not self.query_source_dir.is_dir():
            raise FileNotFoundError(f"query source directory not found: {self.query_source_dir}")
