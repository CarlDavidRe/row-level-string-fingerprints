from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from block_level.storage import (
    feature_mapping_path,
    feature_mapping_schema,
    matrix_schema,
    write_partition_matrices,
)


class PartitionMatrixStorageTest(unittest.TestCase):
    def test_wide_masks_and_alias_mappings_round_trip(self) -> None:
        profiles = {
            ("items", "value"): {
                "block_rows": {
                    3: (0, 1, 1 << 8, 1 << 12, (1 << 12) | 3),
                    7: (5,),
                },
                "block_feature_groups": {
                    3: (("a", "alias-a"), ("b",)),
                    7: (("c",),),
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            matrix_path = Path(directory) / "partition_metadata_v000.parquet"
            size = write_partition_matrices(
                matrix_path, 13, profiles, local_mapping=True
            )
            mapping_path = feature_mapping_path(matrix_path)
            self.assertEqual(
                size, matrix_path.stat().st_size + mapping_path.stat().st_size
            )
            self.assertEqual(pq.read_schema(matrix_path), matrix_schema())
            self.assertEqual(pq.read_schema(mapping_path), feature_mapping_schema())
            matrices = pq.read_table(matrix_path).to_pylist()
            mappings = pq.read_table(mapping_path).to_pylist()
            self.assertEqual(
                [(row["table_name"], row["column_name"], row["partition_id"]) for row in matrices],
                [(row["table_name"], row["column_name"], row["partition_id"]) for row in mappings],
            )
            self.assertEqual([row["partition_id"] for row in matrices], [3, 7])
            self.assertEqual(
                [
                    sum(bit << index for index, bit in enumerate(bits))
                    for bits in matrices[0]["metadata"]
                ],
                list(profiles[("items", "value")]["block_rows"][3]),
            )
            self.assertEqual(len(mappings[0]["feature_groups"]), 13)
            self.assertEqual(mappings[0]["feature_groups"][:2], [["a", "alias-a"], ["b"]])
            self.assertEqual(mappings[1]["feature_groups"][0], ["c"])
            self.assertTrue(all(group == [] for group in mappings[1]["feature_groups"][1:]))


if __name__ == "__main__":
    unittest.main()
