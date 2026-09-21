from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from block_level.config import ExperimentConfig, FingerprintConfig, SweepConfig
from block_level.experiment import FingerprintEvaluator
from block_level.fingerprint import FingerprintBuilder


class SubblockFingerprintTest(unittest.TestCase):
    def test_sweep_expands_each_subblock_size(self) -> None:
        method = "fingerprint_subblock_joint_entropy_equivalence_classes"
        sweep = SweepConfig.from_mapping({
            "feature_selection_methods": [method],
            "widths_by_method": {method: [8]},
            "ngram_sizes": [3],
            "min_block_frequencies_by_method": {method: [0.0]},
            "subblock_sizes_rows": [1, 256, 16384],
        })
        self.assertEqual(
            [point.subblock_size_rows for point in sweep.experiments()],
            [1, 256, 16384],
        )

    def test_legacy_variant_retains_single_mask_representation(self) -> None:
        config = FingerprintConfig(
            widths=(1,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_distribution_entropy_equivalence_classes"
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            builder = FingerprintBuilder(config, 2, root / "metadata", root / "results")
            with duckdb.connect(":memory:") as connection:
                connection.execute(
                    "CREATE TABLE items(value VARCHAR, partition_id INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    [("a", 0), ("b", 0), ("a", 1), ("a", 1)],
                )
                version = builder.build(connection, {("items", "value")})[0]

            metadata = json.loads(
                (root / "metadata" / version.metadata_file).read_text()
            )
            self.assertEqual(
                metadata["representation"], "one_concatenated_infix_mask_per_block"
            )
            self.assertTrue(all(
                "mask_hex" in block and "matrix_rows_hex" not in block
                for block in metadata["targets"]["items.value"]["blocks"]
            ))
            self.assertEqual(version.mean_matrix_rows, 1.0)

    def test_evaluator_has_no_false_negatives(self) -> None:
        fingerprint = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database_path = root / "test.duckdb"
            with duckdb.connect(str(database_path)) as connection:
                connection.execute(
                    "CREATE TABLE items(value VARCHAR, partition_id INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    [("aaa", 0), ("bbb", 0), ("ab", 1), ("bbb", 1)],
                )
            query_path = root / "items.value_queries.txt"
            query_path.write_text("a\nab\nbbb\n", encoding="utf-8")
            sweep = SweepConfig(
                widths_by_method={fingerprint.feature_selection_method: (2,)},
                feature_selection_methods=(fingerprint.feature_selection_method,),
                ngram_sizes=(1,),
                min_block_frequencies_by_method={
                    fingerprint.feature_selection_method: (0.0,)
                },
                subblock_sizes_rows=(1,),
            )
            experiment = ExperimentConfig(
                database_path=database_path,
                query_source_dir=root,
                output_dir=root / "output",
                sweep=sweep,
                block_size_rows=2,
            )
            evaluation = FingerprintEvaluator(
                experiment, fingerprint, [query_path], root / "run"
            ).evaluate()

            self.assertTrue(all(
                row["false_negative_partition_count"] == 0
                for row in evaluation.rows
            ))
            ab_query = next(
                query for query in evaluation.queries
                if query["predicate_value"] == "ab"
            )
            self.assertEqual(
                evaluation.matches_by_version[0][ab_query["query_id"]], {1}
            )

    def test_rows_are_deduplicated_again_after_width_truncation(self) -> None:
        config = FingerprintConfig(
            widths=(1, 2),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            builder = FingerprintBuilder(config, 2, root / "metadata", root / "results")
            with duckdb.connect(":memory:") as connection:
                connection.execute(
                    "CREATE TABLE items(value VARCHAR, partition_id INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    [("a", 0), ("ab", 0), ("b", 1), ("b", 1)],
                )
                versions = builder.build(connection, {("items", "value")})

            one_bit = json.loads(
                (root / "metadata" / versions[0].metadata_file).read_text()
            )
            two_bit = json.loads(
                (root / "metadata" / versions[1].metadata_file).read_text()
            )
            one_bit_counts = {
                block["partition_id"]: block["matrix_row_count"]
                for block in one_bit["targets"]["items.value"]["blocks"]
            }
            two_bit_counts = {
                block["partition_id"]: block["matrix_row_count"]
                for block in two_bit["targets"]["items.value"]["blocks"]
            }
            self.assertEqual(one_bit_counts, {0: 1, 1: 1})
            self.assertEqual(two_bit_counts, {0: 2, 1: 1})

    def test_matrix_probe_requires_one_row_to_contain_the_query(self) -> None:
        config = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            builder = FingerprintBuilder(config, 3, root / "metadata", root / "results")
            with duckdb.connect(":memory:") as connection:
                connection.execute(
                    "CREATE TABLE items(value VARCHAR, partition_id INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    [("aaa", 0), ("aaa", 0), ("bbb", 0), ("ab", 1), ("ab", 1)],
                )
                versions = builder.build(connection, {("items", "value")})

            self.assertEqual(len(versions), 1)
            probe = versions[0].probe
            self.assertEqual(probe("items", "value", "a").candidate_partition_ids, {0, 1})
            self.assertEqual(probe("items", "value", "ab").candidate_partition_ids, {1})
            self.assertEqual(versions[0].mean_matrix_rows, 1.5)
            self.assertEqual(versions[0].metadata_size_bytes, 3)

            metadata = json.loads(
                (root / "metadata" / versions[0].metadata_file).read_text()
            )
            self.assertEqual(
                metadata["representation"],
                "deduplicated_subblock_infix_mask_matrix_per_block",
            )
            blocks = metadata["targets"]["items.value"]["blocks"]
            self.assertEqual(
                {block["partition_id"]: block["matrix_row_count"] for block in blocks},
                {0: 2, 1: 1},
            )

    def test_subblocks_are_consecutive_and_count_null_rows(self) -> None:
        config = FingerprintConfig(
            widths=(1,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            builder = FingerprintBuilder(config, 4, root / "metadata", root / "results")
            with duckdb.connect(":memory:") as connection:
                connection.execute(
                    "CREATE TABLE items(value VARCHAR, partition_id INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    [("a", 0), (None, 0), ("b", 0), ("c", 0), ("d", 1)],
                )
                subblocks = builder.partition_subblock_ngrams(
                    connection, "items", "value"
                )

            self.assertEqual(subblocks[0], (frozenset({"a"}), frozenset({"b", "c"})))
            self.assertEqual(subblocks[1], (frozenset({"d"}),))


if __name__ == "__main__":
    unittest.main()
