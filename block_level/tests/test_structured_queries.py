from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from block_level.config import ExperimentConfig, FingerprintConfig, SweepConfig
from block_level.experiment import FingerprintEvaluator, WorkloadRepository
from block_level.query import parse_query_file


def structured_query(predicate: str, needles: list[str], *, mode: str = "all_needles") -> str:
    return json.dumps({
        "format": "block-skipping-query-v1",
        "predicate_mode": mode,
        "predicate": predicate,
        "needles": needles,
        "lookup_needles": needles,
        "row_count": 1,
        "selectivity": 0.25,
    })


class StructuredQueryTest(unittest.TestCase):
    def test_json_lines_keep_exact_predicate_and_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.value_queries.txt"
            path.write_text(
                structured_query("contains(value, 'A')", ["A"])
                + "\nabc,2,50%\n",
                encoding="utf-8",
            )
            rows = parse_query_file(path)
            self.assertEqual([number for number, _ in rows], [1, 2])
            self.assertEqual(rows[0][1]["predicate_sql"], "contains(value, 'A')")
            self.assertEqual(rows[0][1]["lookup_needles"], ("A",))
            self.assertEqual(rows[0][1]["expected_matching_percent"], "25.00000000%")
            self.assertEqual(rows[1][1]["predicate_sql"], None)
            self.assertEqual(rows[1][1]["lookup_needles"], ("abc",))
            self.assertEqual(rows[1][1]["expected_matching_percent"], "50%")

    def test_ground_truth_executes_structured_sql_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "items.value_queries.txt"
            query_path.write_text(
                structured_query("contains(value, 'A')", ["A"]) + "\n",
                encoding="utf-8",
            )
            fingerprint = FingerprintConfig(
                widths=(1,), ngram_size=1,
                feature_selection_method="fingerprint_internal_entropy_equivalence_classes",
                subblock_size_rows=1, feature_selection_scope="local",
            )
            experiment = ExperimentConfig(
                database_path=root / "unused.duckdb",
                query_source_dir=root,
                output_dir=root / "output",
                sweep=SweepConfig(
                    widths_by_method={fingerprint.feature_selection_method: (1,)},
                    feature_selection_methods=(fingerprint.feature_selection_method,),
                    ngram_sizes=(1,),
                    min_block_frequencies_by_method={fingerprint.feature_selection_method: (0.0,)},
                    subblock_sizes_rows=(1,),
                    feature_selection_scope="local",
                ),
            )
            with duckdb.connect(":memory:") as connection:
                connection.execute("CREATE TABLE items(value VARCHAR, partition_id INTEGER)")
                connection.executemany("INSERT INTO items VALUES (?, ?)", [("A", 0), ("a", 1)])
                queries, skipped = WorkloadRepository(experiment, root / "copies").load(
                    connection, [query_path]
                )
                evaluator = FingerprintEvaluator(experiment, fingerprint, [query_path], root / "run")
                self.assertEqual(evaluator.ground_truth(connection, queries[0]), {0})
            self.assertEqual(skipped, [])

    def test_all_needles_probe_requires_one_matrix_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "test.duckdb"
            with duckdb.connect(str(database_path)) as connection:
                connection.execute("CREATE TABLE items(value VARCHAR, partition_id INTEGER)")
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    [("ab", 0), ("bc", 0), ("abc", 1)],
                )
            query_path = root / "items.value_queries.txt"
            query_path.write_text(
                structured_query(
                    "contains(value, 'ab') AND contains(value, 'bc')", ["ab", "bc"]
                ) + "\n"
                + structured_query("value IS NOT NULL", [], mode="actual_predicate") + "\n",
                encoding="utf-8",
            )
            method = "fingerprint_internal_entropy_equivalence_classes"
            fingerprint = FingerprintConfig(
                widths=(2,), ngram_size=2, feature_selection_method=method,
                subblock_size_rows=1, feature_selection_scope="local",
            )
            experiment = ExperimentConfig(
                database_path=database_path,
                query_source_dir=root,
                output_dir=root / "output",
                sweep=SweepConfig(
                    widths_by_method={method: (2,)},
                    feature_selection_methods=(method,),
                    ngram_sizes=(2,),
                    min_block_frequencies_by_method={method: (0.0,)},
                    subblock_sizes_rows=(1,),
                    feature_selection_scope="local",
                ),
                block_size_rows=2,
            )
            evaluation = FingerprintEvaluator(
                experiment, fingerprint, [query_path], root / "run"
            ).evaluate()
            self.assertEqual(evaluation.ground_truths[0], {1})
            self.assertEqual(evaluation.matches_by_version[0][0], {1})
            self.assertEqual(evaluation.ground_truths[1], {0, 1})
            self.assertEqual(evaluation.matches_by_version[0][1], {0, 1})
            self.assertEqual(evaluation.rows[0]["query_ngram_count"], 2)
            self.assertEqual(evaluation.rows[1]["query_fingerprint_ones"], 0)


if __name__ == "__main__":
    unittest.main()
