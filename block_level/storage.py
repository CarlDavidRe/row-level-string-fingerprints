"""Parquet storage for partition fingerprint matrices and their feature mappings.

The matrix schema matches the frozen partition-metadata schema used by the
matrix-compression project. Feature groups live in a keyed companion file,
since a fingerprint column can represent several equivalent n-grams.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def matrix_schema() -> pa.Schema:
    return pa.schema([
        pa.field("table_name", pa.string(), nullable=False),
        pa.field("column_name", pa.string(), nullable=False),
        pa.field("partition_id", pa.int64(), nullable=False),
        pa.field("metadata", pa.list_(pa.list_(pa.bool_())), nullable=False),
    ])


def feature_mapping_schema() -> pa.Schema:
    return pa.schema([
        pa.field("table_name", pa.string(), nullable=False),
        pa.field("column_name", pa.string(), nullable=False),
        pa.field("partition_id", pa.int64(), nullable=False),
        pa.field("feature_groups", pa.list_(pa.list_(pa.string())), nullable=False),
    ])


def feature_mapping_path(matrix_path: Path) -> Path:
    return matrix_path.with_name(
        matrix_path.name.replace("partition_metadata_", "feature_mappings_", 1)
    )


def bitset_matrix_array(rows: tuple[int, ...], width: int) -> pa.Array:
    """Pack row-major masks directly into Arrow's LSB-first Boolean bitmap."""
    if width < 0:
        raise ValueError("matrix width must be nonnegative")
    cell_count = len(rows) * width
    if cell_count > (1 << 31) - 1:
        raise ValueError("matrix exceeds Arrow's 32-bit list offsets")
    packed = bytearray((cell_count + 7) // 8)
    for row_index, mask in enumerate(rows):
        if mask < 0 or mask.bit_length() > width:
            raise ValueError("matrix row exceeds its declared width")
        if not width:
            continue
        byte_index, offset = divmod(row_index * width, 8)
        byte_count = (offset + width + 7) // 8
        chunk = (mask << offset).to_bytes(byte_count, "little")
        if offset:
            packed[byte_index] |= chunk[0]
            packed[byte_index + 1:byte_index + byte_count] = chunk[1:]
        else:
            packed[byte_index:byte_index + byte_count] = chunk
    values = pa.Array.from_buffers(
        pa.bool_(), cell_count, [None, pa.py_buffer(packed) if packed else None]
    )
    row_offsets = pa.array(
        range(0, cell_count + 1, width) if width else [0] * (len(rows) + 1),
        type=pa.int32(),
    )
    matrix_rows = pa.ListArray.from_arrays(row_offsets, values)
    return pa.ListArray.from_arrays(
        pa.array([0, len(rows)], type=pa.int32()), matrix_rows
    )


def write_partition_matrices(
    matrix_path: Path,
    width: int,
    profiles: dict[tuple[str, str], dict],
    *,
    local_mapping: bool,
) -> int:
    """Write aligned one-record-per-partition matrices and feature mappings.

    Returns the combined on-disk size of both files, including Parquet overhead.
    The matrix file is published last, after its mapping sidecar is complete.
    """
    mapping_path = feature_mapping_path(matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_matrix = matrix_path.with_name(matrix_path.name + ".tmp")
    temporary_mapping = mapping_path.with_name(mapping_path.name + ".tmp")
    try:
        with pq.ParquetWriter(temporary_matrix, matrix_schema()) as matrices, pq.ParquetWriter(
            temporary_mapping, feature_mapping_schema()
        ) as mappings:
            for (table, column), profile in sorted(profiles.items()):
                for partition_id, rows in sorted(profile["block_rows"].items()):
                    groups = (
                        profile["block_feature_groups"][partition_id]
                        if local_mapping else profile["feature_groups"]
                    )
                    if len(groups) > width:
                        raise ValueError("feature mapping exceeds matrix width")
                    keys = [
                        pa.array([table], type=pa.string()),
                        pa.array([column], type=pa.string()),
                        pa.array([partition_id], type=pa.int64()),
                    ]
                    matrices.write_table(pa.Table.from_arrays(
                        [*keys, bitset_matrix_array(rows, width)],
                        schema=matrix_schema(),
                    ))
                    mappings.write_table(pa.Table.from_arrays(
                        [*keys, pa.array(
                            [[list(group) for group in groups] + [[] for _ in range(width - len(groups))]],
                            type=pa.list_(pa.list_(pa.string())),
                        )],
                        schema=feature_mapping_schema(),
                    ))
        os.replace(temporary_mapping, mapping_path)
        os.replace(temporary_matrix, matrix_path)
    finally:
        temporary_matrix.unlink(missing_ok=True)
        temporary_mapping.unlink(missing_ok=True)
    return matrix_path.stat().st_size + mapping_path.stat().st_size
