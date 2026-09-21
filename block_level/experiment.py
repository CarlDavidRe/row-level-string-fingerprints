from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import duckdb

from .config import ExperimentConfig, FingerprintConfig
from .fingerprint import (
    SCOPE_CONFIGURABLE_METHODS,
    FingerprintBuilder,
    FingerprintVersion,
    NGramGenerator,
    quote_identifier,
)


RESULT_COLUMNS = [
    "query_id", "workload_file", "workload_row_number", "table_name", "column_name",
    "predicate_value", "expected_matching_rows", "expected_matching_percent",
    "query_ngram_count", "query_fingerprint_width", "query_fingerprint_ones",
    "metadata_version", "merge_step", "metadata_file", "partition_file_count",
    "metadata_total_size_bytes", "ground_truth_partition_count",
    "metadata_partition_count", "pruned_partition_count",
    "false_positive_partition_count", "false_negative_partition_count",
]
SKIPPED_COLUMNS = ["workload_file", "table_name", "column_name", "reason", "query_count"]
METRICS_COLUMNS = [
    "result_file", "workload", "data_structure", "block_size_rows", "ngram_size",
    "query_mode", "configured_query_count", "partition_count",
    "average_metadata_size_in_bytes", "mean_unneccesary_block_read_ratio",
    "mean_false_block_reads_factor", "mean_absolute_values", "included_query_count",
    "skipped_query_count", "total_query_count", "total_num_skipped_blocks",
]
SWEEP_COLUMNS = [
    "feature_selection_method", "feature_selection_scope", "ngram_size", "min_block_frequency",
    "max_block_frequency", "hamming_cluster_count", "subblock_size_rows",
    "fingerprint_width",
    "metadata_size_bytes", "mean_unnecessary_block_read_ratio", "zero_bit_query_count",
    "query_count", "total_candidate_partitions", "total_pruned_partitions",
    "total_ground_truth_partitions", "false_positive_partition_count",
    "false_negative_partition_count", "run_directory",
]


def write_csv(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]):
    return "" if not values else math.fsum(values) / len(values)


def query_target(path: Path) -> tuple[str, str]:
    suffix = "_queries.txt"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected query filename: {path.name}")
    table, separator, column = path.name[:-len(suffix)].rpartition(".")
    if not separator or not table or not column:
        raise ValueError(f"expected <table>.<column>{suffix}: {path.name}")
    return table, column


class DatabasePartitioner:
    """Add deterministic consecutive partition IDs to every DuckDB base table."""

    @staticmethod
    def partition(database_path: Path, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        with duckdb.connect(str(database_path)) as con:
            tables = [row[0] for row in con.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                     AND table_name <> 'partition_metadata'
                   ORDER BY table_name"""
            ).fetchall()]
            if not tables:
                raise ValueError("no base tables found in the database")
            for index, table_name in enumerate(tables, 1):
                table = quote_identifier(table_name)
                has_partition_id = bool(con.execute(
                    """SELECT count(*) FROM information_schema.columns
                       WHERE table_schema='main' AND table_name=? AND column_name='partition_id'""",
                    [table_name],
                ).fetchone()[0])
                source_columns = "* EXCLUDE (partition_id)" if has_partition_id else "*"
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table} AS
                    WITH ordered AS (
                        SELECT {source_columns}, rowid AS __row_number
                        FROM {table} ORDER BY rowid
                    )
                    SELECT * EXCLUDE (__row_number),
                           CAST(FLOOR(__row_number / {block_size}) AS INTEGER) AS partition_id
                    FROM ordered
                """)
                rows, blocks = con.execute(
                    f"SELECT count(*), count(DISTINCT partition_id) FROM {table}"
                ).fetchone()
                print(f"[{index}/{len(tables)}] {table_name}: {rows:,} rows, {blocks:,} blocks")


class WorkloadRepository:
    def __init__(self, config: ExperimentConfig, query_dir: Path):
        self.config = config
        self.query_dir = query_dir

    def copy_queries(self) -> list[Path]:
        sources = sorted(self.config.query_source_dir.glob("*_queries.txt"))
        if self.config.targets is not None:
            sources = [path for path in sources if query_target(path) in self.config.targets]
        if not sources:
            raise FileNotFoundError(
                f"no matching *_queries.txt files in {self.config.query_source_dir}"
            )
        self.query_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for source in sources:
            destination = self.query_dir / source.name
            shutil.copy2(source, destination)
            copied.append(destination)
        return copied

    @staticmethod
    def target_exists(con, table: str, column: str) -> bool:
        return bool(con.execute(
            """SELECT count(*) FROM information_schema.columns
               WHERE table_schema='main' AND table_name=? AND column_name=?""",
            [table, column],
        ).fetchone()[0])

    def load(self, con, query_files: list[Path]) -> tuple[list[dict], list[dict]]:
        queries: list[dict] = []
        skipped: list[dict] = []
        for path in query_files:
            table, column = query_target(path)
            with path.open(newline="", encoding="utf-8") as handle:
                source_rows = [
                    (number, row)
                    for number, row in enumerate(csv.reader(handle), 1)
                    if row
                ]
            if not self.target_exists(con, table, column):
                skipped.append({
                    "workload_file": path.name,
                    "table_name": table,
                    "column_name": column,
                    "reason": "target_not_in_database",
                    "query_count": len(source_rows),
                })
                continue
            for row_number, row in source_rows:
                if self.config.query_limit is not None and len(queries) >= self.config.query_limit:
                    return queries, skipped
                queries.append({
                    "query_id": len(queries),
                    "workload_file": path.name,
                    "workload_row_number": row_number,
                    "table_name": table,
                    "column_name": column,
                    "predicate_value": row[0],
                    "expected_matching_rows": row[1] if len(row) > 1 else "",
                    "expected_matching_percent": row[2] if len(row) > 2 else "",
                })
        if not queries:
            raise ValueError("no evaluable workload queries")
        return queries, skipped


@dataclass
class EvaluationResult:
    queries: list[dict]
    skipped: list[dict]
    ground_truths: dict[int, frozenset[int]]
    versions: list[FingerprintVersion]
    partition_ids: dict[tuple[str, str], frozenset[int]]
    matches_by_version: dict[int, dict[int, frozenset[int]]]
    rows: list[dict]


class FingerprintEvaluator:
    def __init__(
        self,
        config: ExperimentConfig,
        fingerprint_config: FingerprintConfig,
        query_files: list[Path],
        run_dir: Path,
    ):
        self.config = config
        self.fingerprint_config = fingerprint_config
        self.query_files = query_files
        self.run_dir = run_dir
        self.output_dir = run_dir / "block_skipping_results"
        self.fingerprint_dir = run_dir / "block_infix_fingerprints"
        self.ngrams = NGramGenerator(
            fingerprint_config.ngram_size, fingerprint_config.ascii_only
        )

    @staticmethod
    def escaped_like(value: str) -> str:
        escape = chr(92)
        return "%" + value.replace(escape, escape * 2).replace(
            "%", escape + "%"
        ).replace("_", escape + "_") + "%"

    def ground_truth(self, con, query: dict) -> frozenset[int]:
        rows = con.execute(
            f"SELECT DISTINCT partition_id FROM {quote_identifier(query['table_name'])} "
            f"WHERE CAST({quote_identifier(query['column_name'])} AS VARCHAR) "
            "ILIKE ? ESCAPE '\\' ORDER BY partition_id",
            [self.escaped_like(query["predicate_value"])],
        ).fetchall()
        return frozenset(int(row[0]) for row in rows)

    @staticmethod
    def assert_no_false_negatives(
        label: str,
        queries: list[dict],
        ground_truths: dict[int, frozenset[int]],
        matches: dict[int, frozenset[int]],
    ) -> None:
        failures = []
        for query in queries:
            query_id = query["query_id"]
            missing = ground_truths[query_id] - matches[query_id]
            if missing:
                failures.append((query_id, query["predicate_value"], sorted(missing)))
        if failures:
            sample = "; ".join(
                f"query_id={query_id} predicate={predicate!r} missed={missing[:20]}"
                for query_id, predicate, missing in failures[:10]
            )
            raise RuntimeError(
                f"FATAL: fingerprint pruning is incorrect for {label}; "
                f"{len(failures)} queries have false negatives. {sample}"
            )

    def evaluate(self) -> EvaluationResult:
        workload = WorkloadRepository(self.config, self.config.output_dir / "queries")
        with duckdb.connect(str(self.config.database_path), read_only=True) as con:
            queries, skipped = workload.load(con, self.query_files)
            targets = {(query["table_name"], query["column_name"]) for query in queries}
            partition_ids = {
                target: frozenset(
                    int(row[0])
                    for row in con.execute(
                        f"SELECT DISTINCT partition_id FROM {quote_identifier(target[0])} "
                        "ORDER BY partition_id"
                    ).fetchall()
                )
                for target in targets
            }
            ground_truths = {
                query["query_id"]: self.ground_truth(con, query) for query in queries
            }
            builder = FingerprintBuilder(
                self.fingerprint_config,
                self.config.block_size_rows,
                self.fingerprint_dir,
                self.output_dir,
            )
            versions = builder.build(con, targets)
            if not versions:
                raise ValueError("the fingerprint builder returned no versions")

            matches_by_version: dict[int, dict[int, frozenset[int]]] = {}
            result_rows: list[dict] = []
            for version in sorted(versions, key=lambda item: item.metadata_version):
                matches: dict[int, frozenset[int]] = {}
                probes = {}
                for query in queries:
                    query_id = query["query_id"]
                    target = (query["table_name"], query["column_name"])
                    probe = version.probe(*target, query["predicate_value"])
                    candidates = frozenset(int(value) for value in probe.candidate_partition_ids)
                    if not candidates <= partition_ids[target]:
                        invalid = sorted(candidates - partition_ids[target])
                        raise ValueError(
                            f"fingerprint version {version.metadata_version} returned invalid "
                            f"partition IDs for {target}: {invalid}"
                        )
                    matches[query_id] = candidates
                    probes[query_id] = probe
                self.assert_no_false_negatives(
                    f"version {version.metadata_version}", queries, ground_truths, matches
                )
                matches_by_version[version.metadata_version] = matches
                for query in queries:
                    query_id = query["query_id"]
                    target = (query["table_name"], query["column_name"])
                    truth = ground_truths[query_id]
                    candidates = matches[query_id]
                    probe = probes[query_id]
                    result_rows.append({
                        **query,
                        "query_ngram_count": len(frozenset(self.ngrams.generate(query["predicate_value"]))),
                        "query_fingerprint_width": probe.width,
                        "query_fingerprint_ones": probe.ones,
                        "metadata_version": version.metadata_version,
                        "merge_step": version.merge_step,
                        "metadata_file": version.metadata_file,
                        "partition_file_count": len(partition_ids[target]),
                        "metadata_total_size_bytes": version.metadata_size_bytes,
                        "ground_truth_partition_count": len(truth),
                        "metadata_partition_count": len(candidates),
                        "pruned_partition_count": len(partition_ids[target]) - len(candidates),
                        "false_positive_partition_count": len(candidates - truth),
                        "false_negative_partition_count": len(truth - candidates),
                    })
                print(f"Evaluated version {version.metadata_version}: {len(queries)} queries")

        write_csv(self.output_dir / "results.csv", RESULT_COLUMNS, result_rows)
        write_csv(self.output_dir / "skipped_workloads.csv", SKIPPED_COLUMNS, skipped)
        return EvaluationResult(
            queries, skipped, ground_truths, versions, partition_ids,
            matches_by_version, result_rows,
        )


class ResultExporter:
    def __init__(
        self,
        config: ExperimentConfig,
        fingerprint_config: FingerprintConfig,
        run_dir: Path,
    ):
        self.config = config
        self.fingerprint_config = fingerprint_config
        self.run_dir = run_dir
        self.output_dir = run_dir / "block_skipping_results"
        self.fingerprint_dir = run_dir / "block_infix_fingerprints"

    def export(self, evaluation: EvaluationResult) -> None:
        versions_by_id = {
            version.metadata_version: version for version in evaluation.versions
        }
        metrics_rows, size_rows, ratio_rows, manifest_points = [], [], [], []
        for version_id in sorted(versions_by_id):
            version = versions_by_id[version_id]
            rows = [row for row in evaluation.rows if row["metadata_version"] == version_id]
            ratios: list[float] = []
            factors: list[float] = []
            absolutes: list[float] = []
            skipped_queries = 0
            for row in rows:
                total = len(evaluation.partition_ids[(row["table_name"], row["column_name"])])
                truth = row["ground_truth_partition_count"]
                false_positives = row["false_positive_partition_count"]
                skippable = total - truth
                if skippable == 0:
                    skipped_queries += 1
                    continue
                ratios.append(false_positives / skippable)
                if truth > 0:
                    factors.append(false_positives / truth)
                absolutes.append(false_positives * self.config.block_size_rows)
            represented_targets = {(row["table_name"], row["column_name"]) for row in rows}
            matrix_count = sum(len(evaluation.partition_ids[target]) for target in represented_targets)
            total_scanned = sum(row["metadata_partition_count"] for row in rows)
            total_truth = sum(row["ground_truth_partition_count"] for row in rows)
            total_possible = sum(row["partition_file_count"] for row in rows)
            total_skipped = total_possible - total_scanned
            metrics = {
                "metadata_size_bytes": version.metadata_size_bytes,
                "metadata_size_mib": version.metadata_size_bytes / 2**20,
                "partition_count": matrix_count,
                "query_count": len(rows),
                "total_skipped_partitions": total_skipped,
                "false_positive_partition_count": sum(row["false_positive_partition_count"] for row in rows),
                "false_negative_partition_count": sum(row["false_negative_partition_count"] for row in rows),
                "zero_bit_query_count": sum(row["query_fingerprint_ones"] == 0 for row in rows),
            }
            metrics_rows.append({
                "result_file": version.metadata_file,
                "workload": self.config.workload_name,
                "data_structure": "fingerprintMatrix",
                "block_size_rows": self.config.block_size_rows,
                "ngram_size": "" if version.ngram_size is None else version.ngram_size,
                "query_mode": "query_file",
                "configured_query_count": self.config.query_limit or "",
                "partition_count": matrix_count,
                "average_metadata_size_in_bytes": version.metadata_size_bytes / matrix_count if matrix_count else "",
                "mean_unneccesary_block_read_ratio": mean(ratios),
                "mean_false_block_reads_factor": mean(factors),
                "mean_absolute_values": mean(absolutes),
                "included_query_count": len(ratios),
                "skipped_query_count": skipped_queries,
                "total_query_count": len(rows),
                "total_num_skipped_blocks": total_skipped,
            })
            size_rows.append({
                "metadata_version": version_id,
                "merge_step": version.merge_step,
                "metadata_file": version.metadata_file,
                "metadata_file_size_bytes": version.metadata_size_bytes,
                "metadata_file_size_mib": f"{version.metadata_size_bytes / 2**20:.6f}",
                "total_skipped_partitions": total_skipped,
                "total_scanned_partitions": total_scanned,
                "total_ground_truth_partitions": total_truth,
                "query_count": len(rows),
            })
            ratio_rows.append({
                "metadata_version": version_id,
                "merge_step": version.merge_step,
                "metadata_file": version.metadata_file,
                "metadata_file_size_bytes": version.metadata_size_bytes,
                "metadata_file_size_kb": f"{version.metadata_size_bytes / 1024:.8f}",
                "mean_metadata_file_size_kb_per_partition": (
                    f"{version.metadata_size_bytes / 1024 / matrix_count:.8f}" if matrix_count else ""
                ),
                "mean_matrix_rows": f"{version.mean_matrix_rows:.8f}",
                "partition_count": matrix_count,
                "mean_unnecessary_block_read_ratio": "" if not ratios else f"{mean(ratios):.8f}",
                "ratio_query_count": len(ratios),
                "query_count": len(rows),
                "zero_bit_query_count": metrics["zero_bit_query_count"],
            })
            manifest_points.append({
                "id": f"block-infix-v{version_id:03d}",
                "title": version.merge_step,
                "technique": (
                    "subblock_infix_fingerprint_matrix"
                    if self.fingerprint_config.subblock_size_rows is not None
                    else "concatenated_block_infix_fingerprint"
                ),
                "parent_id": None,
                "parameters": {
                    "fingerprint_width": next((row["query_fingerprint_width"] for row in rows), 0),
                    "feature_selection_method": self.fingerprint_config.feature_selection_method,
                    "feature_selection_scope": self.fingerprint_config.feature_selection_scope,
                    "ngram_size": version.ngram_size,
                    "block_size_rows": self.config.block_size_rows,
                    "subblock_size_rows": self.fingerprint_config.subblock_size_rows,
                    "min_block_frequency": self.fingerprint_config.min_block_frequency,
                    "max_block_frequency": self.fingerprint_config.max_block_frequency,
                    "feature_candidates": (
                        "frequency_filtered_observed_subblock_ngrams"
                        if self.fingerprint_config.subblock_size_rows is not None
                        else "frequency_filtered_observed_block_ngrams"
                    ),
                },
                "metadata_dir": str(self.fingerprint_dir.resolve()),
                "results_path": str((self.output_dir / "results.csv").resolve()),
                "metrics": metrics,
            })

        baseline = next((
            float(row["mean_unnecessary_block_read_ratio"])
            for row in ratio_rows
            if row["metadata_version"] == 0 and row["mean_unnecessary_block_read_ratio"] != ""
        ), None)
        for row in ratio_rows:
            value = row["mean_unnecessary_block_read_ratio"]
            row["unnecessary_block_read_ratio_increase_from_v000"] = (
                "" if value == "" or baseline is None else f"{float(value) - baseline:.8f}"
            )
        write_csv(self.output_dir / "metrics_summary.csv", METRICS_COLUMNS, metrics_rows)
        write_csv(self.output_dir / "size_pruning_tradeoff_summary.csv", [
            "metadata_version", "merge_step", "metadata_file", "metadata_file_size_bytes",
            "metadata_file_size_mib", "total_skipped_partitions", "total_scanned_partitions",
            "total_ground_truth_partitions", "query_count",
        ], size_rows)
        write_csv(self.output_dir / "mean_rows_unnecessary_block_read_ratio_summary.csv", [
            "metadata_version", "merge_step", "metadata_file", "metadata_file_size_bytes",
            "metadata_file_size_kb", "mean_metadata_file_size_kb_per_partition",
            "mean_matrix_rows", "partition_count", "mean_unnecessary_block_read_ratio",
            "unnecessary_block_read_ratio_increase_from_v000", "ratio_query_count",
            "query_count", "zero_bit_query_count",
        ], ratio_rows)
        (self.run_dir / "experiment_manifest.json").write_text(
            json.dumps({"format_version": 1, "points": manifest_points}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.plot(evaluation, size_rows, ratio_rows)

    def plot(self, evaluation: EvaluationResult, size_rows: list[dict], ratio_rows: list[dict]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        sizes_mib = [float(row["metadata_file_size_mib"]) for row in size_rows]
        skipped = [row["total_skipped_partitions"] for row in size_rows]
        labels = [f"v{row['metadata_version']}" for row in size_rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sizes_mib, skipped, s=90, color="#2563eb", edgecolor="#172554")
        for label, x_value, y_value in zip(labels, sizes_mib, skipped):
            ax.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(7, 7))
        ax.set(xlabel="Metadata Size (MiB)", ylabel="Total Skipped Partitions Across Workload",
               title="Metadata Size vs. Partition Pruning")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(self.output_dir / "size_pruning_tradeoff.pdf"); plt.close(fig)

        plottable = [row for row in ratio_rows if row["mean_unnecessary_block_read_ratio"] != ""]
        fig, ax = plt.subplots(figsize=(8, 5))
        x_values = [float(row["mean_metadata_file_size_kb_per_partition"]) for row in plottable]
        y_values = [float(row["mean_unnecessary_block_read_ratio"]) for row in plottable]
        ax.scatter(x_values, y_values, s=90, color="#2563eb", edgecolor="#172554")
        for row, x_value, y_value in zip(plottable, x_values, y_values):
            ax.annotate(f"v{row['metadata_version']}", (x_value, y_value),
                        textcoords="offset points", xytext=(7, 7))
        ax.set(xlabel="Mean Metadata Size per Partition Matrix (kB)",
               ylabel="Mean Unnecessary Block Read Ratio",
               title="Metadata Size vs. Mean Unnecessary Block Read Ratio")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(self.output_dir / "mean_rows_unnecessary_block_read_ratio.pdf"); plt.close(fig)

        diagnostics_path = self.output_dir / "selected_ngram_diagnostics.csv"
        if not diagnostics_path.is_file():
            return
        csv_field_limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(csv_field_limit)
                break
            except OverflowError:
                csv_field_limit //= 10
        with diagnostics_path.open(newline="", encoding="utf-8") as handle:
            diagnostics = list(csv.DictReader(handle))
        versions = sorted(evaluation.versions, key=lambda version: version.metadata_version)
        panel_columns = min(3, max(1, len(versions)))
        panel_rows = math.ceil(len(versions) / panel_columns)
        fig, axes = plt.subplots(panel_rows, panel_columns, squeeze=False,
                                 figsize=(4.5 * panel_columns, 3.4 * panel_rows),
                                 sharex=True, sharey=True)
        for ax, version in zip(axes.flat, versions):
            frequencies = [
                float(row["block_frequency"])
                for row in diagnostics
                if int(row["metadata_version"]) == version.metadata_version
            ]
            width = next((
                row["query_fingerprint_width"] for row in evaluation.rows
                if row["metadata_version"] == version.metadata_version
            ), 0)
            if frequencies:
                ax.hist(frequencies, bins=[index / 20 for index in range(21)],
                        weights=[1 / len(frequencies)] * len(frequencies),
                        color="#2563eb", edgecolor="#172554", alpha=.85)
            else:
                ax.text(.5, .5, "No selected features", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(f"{width}-bit ({len(frequencies)} features)")
            ax.set_xlim(0, 1); ax.set_xticks([index / 10 for index in range(11)])
            ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        for ax in list(axes.flat)[len(versions):]:
            ax.set_visible(False)
        frequency_unit = (
            "Sub-block" if self.fingerprint_config.subblock_size_rows is not None
            else "Block"
        )
        fig.supxlabel(f"{frequency_unit} presence frequency")
        fig.supylabel("Share of selected n-grams")
        fig.suptitle(
            "Selected n-gram block-frequency distributions\n"
            f"{self.fingerprint_config.feature_selection_method}; "
            f"n={self.fingerprint_config.ngram_size}; frequency="
            f"[{self.fingerprint_config.min_block_frequency:g}, "
            f"{self.fingerprint_config.max_block_frequency:g}]"
        )
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(self.output_dir / "selected_ngram_block_frequency_histograms.pdf")
        plt.close(fig)


class SweepRunner:
    """Run every point described by an experiment configuration."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    @staticmethod
    def _frequency_label(value: float) -> str:
        return format(value, "g").replace(".", "p")

    def run_directory(self, fingerprint: FingerprintConfig) -> Path:
        path = (
            self.config.output_dir
            / fingerprint.feature_selection_method
        )
        if fingerprint.feature_selection_method in SCOPE_CONFIGURABLE_METHODS:
            path /= f"feature_scope_{fingerprint.feature_selection_scope}"
        path /= f"ngram_{fingerprint.ngram_size}"
        if fingerprint.feature_selection_method.endswith("_hamming_clusters"):
            path /= f"hamming_cluster_count_{fingerprint.hamming_cluster_count}"
        if fingerprint.subblock_size_rows is not None:
            path /= f"subblock_size_rows_{fingerprint.subblock_size_rows}"
        return (
            path
            / f"min_block_frequency_{self._frequency_label(fingerprint.min_block_frequency)}"
            / f"max_block_frequency_{self._frequency_label(fingerprint.max_block_frequency)}"
        )

    def run(self) -> Path:
        self.config.validate_inputs()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.partition_database:
            print(f"Partitioning database into {self.config.block_size_rows:,}-row blocks")
            DatabasePartitioner.partition(
                self.config.database_path, self.config.block_size_rows
            )
        workload = WorkloadRepository(self.config, self.config.output_dir / "queries")
        query_files = workload.copy_queries()
        print(f"Copied {len(query_files)} query files to {self.config.output_dir / 'queries'}")
        sweep_rows: list[dict] = []
        experiments = list(self.config.sweep.experiments())
        for index, fingerprint in enumerate(experiments, 1):
            run_dir = self.run_directory(fingerprint)
            print(
                f"\n=== [{index}/{len(experiments)}] {fingerprint.feature_selection_method}; "
                f"scope={fingerprint.feature_selection_scope}; "
                f"n={fingerprint.ngram_size}; frequency="
                f"[{fingerprint.min_block_frequency:g}, {fingerprint.max_block_frequency:g}]; "
                f"widths={fingerprint.widths}"
                + (
                    f"; subblock_size_rows={fingerprint.subblock_size_rows}"
                    if fingerprint.subblock_size_rows is not None else ""
                )
                + " ==="
            )
            evaluator = FingerprintEvaluator(
                self.config, fingerprint, query_files, run_dir
            )
            evaluation = evaluator.evaluate()
            ResultExporter(self.config, fingerprint, run_dir).export(evaluation)
            for version in evaluation.versions:
                rows = [
                    row for row in evaluation.rows
                    if row["metadata_version"] == version.metadata_version
                ]
                ratios = [
                    row["false_positive_partition_count"]
                    / (row["partition_file_count"] - row["ground_truth_partition_count"])
                    for row in rows
                    if row["partition_file_count"] > row["ground_truth_partition_count"]
                ]
                sweep_rows.append({
                    "feature_selection_method": fingerprint.feature_selection_method,
                    "feature_selection_scope": fingerprint.feature_selection_scope,
                    "ngram_size": fingerprint.ngram_size,
                    "min_block_frequency": fingerprint.min_block_frequency,
                    "max_block_frequency": fingerprint.max_block_frequency,
                    "hamming_cluster_count": fingerprint.hamming_cluster_count,
                    "subblock_size_rows": fingerprint.subblock_size_rows or "",
                    "fingerprint_width": rows[0]["query_fingerprint_width"],
                    "metadata_size_bytes": version.metadata_size_bytes,
                    "mean_unnecessary_block_read_ratio": mean(ratios),
                    "zero_bit_query_count": sum(row["query_fingerprint_ones"] == 0 for row in rows),
                    "query_count": len(rows),
                    "total_candidate_partitions": sum(row["metadata_partition_count"] for row in rows),
                    "total_pruned_partitions": sum(row["pruned_partition_count"] for row in rows),
                    "total_ground_truth_partitions": sum(row["ground_truth_partition_count"] for row in rows),
                    "false_positive_partition_count": sum(row["false_positive_partition_count"] for row in rows),
                    "false_negative_partition_count": sum(row["false_negative_partition_count"] for row in rows),
                    "run_directory": str(run_dir.resolve()),
                })
        summary_path = self.config.output_dir / "fingerprint_sweep_summary.csv"
        write_csv(summary_path, SWEEP_COLUMNS, sweep_rows)
        snapshot = {
            "database_path": str(self.config.database_path),
            "query_source_dir": str(self.config.query_source_dir),
            "output_dir": str(self.config.output_dir),
            "block_size_rows": self.config.block_size_rows,
            "workload_name": self.config.workload_name,
            "targets": None if self.config.targets is None else sorted(self.config.targets),
            "query_limit": self.config.query_limit,
            "partition_database": self.config.partition_database,
            "sweep": asdict(self.config.sweep),
        }
        (self.config.output_dir / "resolved_experiment_config.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote {len(sweep_rows)} sweep points to {summary_path}")
        return summary_path
