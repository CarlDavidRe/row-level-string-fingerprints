from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from block_level.config import ExperimentConfig, FingerprintConfig, SweepConfig
from block_level.experiment import EvaluationInputs, FingerprintEvaluator
from block_level.fingerprint import (
    FeatureSelector,
    FingerprintBuilder,
    FingerprintDataCache,
)


class SubblockFingerprintTest(unittest.TestCase):
    def test_candidate_postings_pack_the_same_presence_masks(self) -> None:
        config = FingerprintConfig(
            widths=(3,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_internal_entropy_equivalence_classes"
            ),
        )
        selector = FeatureSelector(config)
        units = (
            frozenset({"a", "b"}),
            frozenset({"b"}),
            frozenset({"a", "c"}),
            frozenset({"c"}),
        )

        unit_count, candidates = selector.candidates(units)

        self.assertEqual(unit_count, 4)
        self.assertEqual(candidates, {"a": 0b0101, "b": 0b0011, "c": 0b1100})
        self.assertEqual(
            {name: list(postings) for name, postings in selector._candidate_postings.items()},
            {"a": [0, 2], "b": [0, 1], "c": [2, 3]},
        )

    def test_native_internal_entropy_matches_exact_bitset_selection(self) -> None:
        config = FingerprintConfig(
            widths=(5,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_internal_entropy_equivalence_classes"
            ),
        )
        selector = FeatureSelector(config)
        units = tuple(
            frozenset(
                feature
                for feature, divisor in (
                    ("a", 2), ("b", 3), ("c", 4), ("d", 5), ("e", 7)
                )
                if index % divisor in {0, 1}
            )
            for index in range(24)
        )
        unit_count, candidates = selector.candidates(units)
        representatives, _ = selector.collapse_equivalent(candidates)
        postings = {
            feature: selector._candidate_postings[feature]
            for feature in representatives
        }

        exact = selector.internal_entropy(
            unit_count, representatives.copy(), 5
        )
        native = selector.internal_entropy(
            unit_count, representatives.copy(), 5, postings
        )

        self.assertEqual(native, exact)

    def test_internal_entropy_without_subblocks_rejects_local_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires.*subblock_size_rows"):
            FingerprintConfig(
                widths=(2,),
                ngram_size=1,
                feature_selection_method=(
                    "fingerprint_internal_entropy_equivalence_classes"
                ),
                feature_selection_scope="local",
            )

    def test_mixed_sweep_expands_both_subblock_entropy_methods(self) -> None:
        internal = "fingerprint_internal_entropy_equivalence_classes"
        subblock = "fingerprint_subblock_joint_entropy_equivalence_classes"
        sweep = SweepConfig(
            widths_by_method={internal: (2,), subblock: (2,)},
            feature_selection_methods=(internal, subblock),
            ngram_sizes=(1,),
            min_block_frequencies_by_method={internal: (0.0,), subblock: (0.0,)},
            subblock_sizes_rows=(1,),
            feature_selection_scope="local",
        )

        points = list(sweep.experiments())
        self.assertEqual(
            [
                (
                    point.feature_selection_method,
                    point.feature_selection_scope,
                    point.subblock_size_rows,
                )
                for point in points
            ],
            [(internal, "local", 1), (subblock, "local", 1)],
        )

    def test_internal_entropy_sweep_without_subblocks_keeps_legacy_form(self) -> None:
        method = "fingerprint_internal_entropy_equivalence_classes"
        sweep = SweepConfig.from_mapping({
            "feature_selection_methods": [method],
            "widths_by_method": {method: [2]},
            "ngram_sizes": [1],
            "min_block_frequencies_by_method": {method: [0.0]},
            "feature_selection_scope": "local",
        })

        point = next(sweep.experiments())
        self.assertIsNone(point.subblock_size_rows)
        self.assertEqual(point.feature_selection_scope, "global")

    def test_internal_entropy_sweep_expands_multiple_scopes(self) -> None:
        method = "fingerprint_internal_entropy_equivalence_classes"
        sweep = SweepConfig.from_mapping({
            "feature_selection_methods": [method],
            "widths_by_method": {method: [2]},
            "ngram_sizes": [1],
            "min_block_frequencies_by_method": {method: [0.0]},
            "subblock_sizes_rows": [1, 8],
            "feature_selection_scopes": ["global", "local"],
        })

        self.assertEqual(
            [
                (point.subblock_size_rows, point.feature_selection_scope)
                for point in sweep.experiments()
            ],
            [(1, "global"), (1, "local"), (8, "global"), (8, "local")],
        )

    def test_global_subblock_internal_entropy_flattens_subblocks(self) -> None:
        method = "fingerprint_internal_entropy_equivalence_classes"
        selector = FeatureSelector(FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=method,
            subblock_size_rows=1,
            feature_selection_scope="global",
        ))
        partition_subblocks = {
            0: (frozenset({"a"}), frozenset({"b"})),
            1: (frozenset({"a"}), frozenset({"b"})),
        }

        selected = selector.select_global_subblock_internal_entropy(
            partition_subblocks, 2
        )

        self.assertEqual(selected, [("a",), ("b",)])

    def test_local_subblock_internal_entropy_builds_local_mappings(self) -> None:
        method = "fingerprint_internal_entropy_equivalence_classes"
        selector = FeatureSelector(FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=method,
            subblock_size_rows=1,
            feature_selection_scope="local",
        ))
        partition_subblocks = {
            0: (frozenset({"a"}), frozenset({"b"})),
            1: (frozenset({"c"}), frozenset({"d"})),
        }

        selected = selector.select_block_local_internal_entropy(
            partition_subblocks, 2
        )

        self.assertEqual(selected, {
            0: (("a",), ("b",)),
            1: (("c",), ("d",)),
        })

    def test_subblock_global_scope_uses_one_shared_mapping(self) -> None:
        config = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=1,
            feature_selection_scope="global",
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
                    [("a", 0), ("b", 0), ("a", 1), ("c", 1)],
                )
                version = builder.build(connection, {("items", "value")})[0]

            metadata = json.loads(
                (root / "metadata" / version.metadata_file).read_text()
            )
            target = metadata["targets"]["items.value"]
            self.assertEqual(target["feature_scope"], "target_shared")
            self.assertTrue(target["feature_groups"])
            self.assertTrue(all(
                "feature_groups" not in block for block in target["blocks"]
            ))

    def test_compatible_builders_reuse_extracted_ngrams(self) -> None:
        method = "fingerprint_internal_entropy_equivalence_classes"
        cache = FingerprintDataCache()
        global_config = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=method,
            subblock_size_rows=1,
            feature_selection_scope="global",
        )
        local_config = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=method,
            subblock_size_rows=1,
            feature_selection_scope="local",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with duckdb.connect(":memory:") as connection:
                connection.execute(
                    "CREATE TABLE items(value VARCHAR, partition_id INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO items VALUES (?, ?)",
                    [("a", 0), ("b", 0), ("c", 1), ("d", 1)],
                )
                global_builder = FingerprintBuilder(
                    global_config,
                    2,
                    root / "global_metadata",
                    root / "global_results",
                    cache,
                )
                global_builder.build(connection, {("items", "value")})

                local_builder = FingerprintBuilder(
                    local_config,
                    2,
                    root / "local_metadata",
                    root / "local_results",
                    cache,
                )
                with mock.patch.object(
                    local_builder,
                    "partition_subblock_ngrams",
                    side_effect=AssertionError("database scan should be cached"),
                ):
                    versions = local_builder.build(connection, {("items", "value")})

            self.assertEqual(len(versions), 1)

    def test_one_subblock_global_scope_matches_global_internal_entropy(self) -> None:
        subblock_config = FingerprintConfig(
            widths=(3,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=4,
            feature_selection_scope="global",
        )
        internal_config = FingerprintConfig(
            widths=(3,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_internal_entropy_equivalence_classes"
            ),
            feature_selection_scope="global",
        )
        partition_grams = {
            0: frozenset({"a", "b"}),
            1: frozenset({"b", "c"}),
            2: frozenset({"a", "c"}),
            3: frozenset({"a"}),
        }
        partition_subblocks = {
            partition_id: (grams,)
            for partition_id, grams in partition_grams.items()
        }

        subblock_groups = FeatureSelector(
            subblock_config
        ).select_global_joint_entropy(partition_subblocks, 3)
        internal_groups = FeatureSelector(internal_config).select(
            partition_grams, 3
        )

        self.assertEqual(subblock_groups, internal_groups)

    def test_sweep_expands_each_subblock_size(self) -> None:
        method = "fingerprint_subblock_joint_entropy_equivalence_classes"
        sweep = SweepConfig.from_mapping({
            "feature_selection_methods": [method],
            "widths_by_method": {method: [8]},
            "ngram_sizes": [3],
            "min_block_frequencies_by_method": {method: [0.0]},
            "subblock_sizes_rows": [1, 256, 16384],
            "feature_selection_scope": "local",
        })
        points = list(sweep.experiments())
        self.assertEqual(
            [point.subblock_size_rows for point in points],
            [1, 256, 16384],
        )
        self.assertTrue(all(
            point.feature_selection_scope == "local" for point in points
        ))

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

    def test_local_internal_entropy_with_subblocks_uses_local_matrix_mappings(self) -> None:
        config = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_internal_entropy_equivalence_classes"
            ),
            subblock_size_rows=1,
            feature_selection_scope="local",
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
                    [("a", 0), ("b", 0), ("c", 1), ("d", 1)],
                )
                version = builder.build(connection, {("items", "value")})[0]

            metadata = json.loads(
                (root / "metadata" / version.metadata_file).read_text()
            )
            self.assertEqual(
                metadata["representation"],
                "deduplicated_subblock_infix_mask_matrix_per_block",
            )
            self.assertEqual(metadata["subblock_size_rows"], 1)
            self.assertTrue(all(
                "matrix_rows_hex" in block
                for block in metadata["targets"]["items.value"]["blocks"]
            ))
            blocks = metadata["targets"]["items.value"]["blocks"]
            self.assertEqual(
                {
                    block["partition_id"]: block["feature_groups"]
                    for block in blocks
                },
                {0: [["a"], ["b"]], 1: [["c"], ["d"]]},
            )
            self.assertEqual(
                version.probe("items", "value", "abcd").candidate_partition_ids,
                frozenset(),
            )

    def test_evaluator_has_no_false_negatives(self) -> None:
        fingerprint = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=1,
            feature_selection_scope="local",
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
                    [
                        ("a", 0), ("b", 0), (None, 0), (None, 0),
                        ("ab", 1), (None, 1), (None, 1), (None, 1),
                    ],
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
                feature_selection_scope="local",
            )
            experiment = ExperimentConfig(
                database_path=database_path,
                query_source_dir=root,
                output_dir=root / "output",
                sweep=sweep,
                block_size_rows=4,
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

            cached_inputs = EvaluationInputs(
                evaluation.queries,
                evaluation.skipped,
                evaluation.ground_truths,
                evaluation.partition_ids,
            )
            cached_evaluator = FingerprintEvaluator(
                experiment,
                fingerprint,
                [query_path],
                root / "cached_run",
                cached_inputs,
            )
            with mock.patch.object(
                cached_evaluator,
                "prepare_inputs",
                side_effect=AssertionError("ground truth should be cached"),
            ):
                cached_evaluation = cached_evaluator.evaluate()
            self.assertEqual(
                cached_evaluation.ground_truths,
                evaluation.ground_truths,
            )

    def test_rows_are_deduplicated_again_after_width_truncation(self) -> None:
        config = FingerprintConfig(
            widths=(1, 2),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=1,
            feature_selection_scope="local",
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
                    [
                        ("a", 0), ("ab", 0), (None, 0), (None, 0),
                        (None, 1), (None, 1), (None, 1), (None, 1),
                    ],
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
            self.assertEqual(one_bit_counts, {0: 2, 1: 1})
            self.assertEqual(two_bit_counts, {0: 3, 1: 1})

    def test_matrix_probe_requires_one_row_to_contain_the_query(self) -> None:
        config = FingerprintConfig(
            widths=(2,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=1,
            feature_selection_scope="local",
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
                    [
                        ("a", 0), ("b", 0), (None, 0), (None, 0),
                        ("ab", 1), (None, 1), (None, 1), (None, 1),
                    ],
                )
                versions = builder.build(connection, {("items", "value")})

            self.assertEqual(len(versions), 1)
            probe = versions[0].probe
            self.assertEqual(probe("items", "value", "a").candidate_partition_ids, {0, 1})
            self.assertEqual(probe("items", "value", "ab").candidate_partition_ids, {1})
            self.assertEqual(versions[0].mean_matrix_rows, 2.5)
            self.assertEqual(versions[0].metadata_size_bytes, 5)

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
                {0: 3, 1: 2},
            )

    def test_subblocks_are_consecutive_and_count_null_rows(self) -> None:
        config = FingerprintConfig(
            widths=(1,),
            ngram_size=1,
            feature_selection_method=(
                "fingerprint_subblock_joint_entropy_equivalence_classes"
            ),
            subblock_size_rows=2,
            feature_selection_scope="local",
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
