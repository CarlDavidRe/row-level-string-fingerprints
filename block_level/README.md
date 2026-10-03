# Block-level fingerprint sweeps

This package is the scriptable form of `ceb_imdb_block_skipping_portable.ipynb`.
It separates configuration, feature selection/fingerprint construction, workload
evaluation, result export, and sweep orchestration. The original notebook is kept
in this directory as a reference.

Run the configured `job` and `tpch_sf10` sweep with:

```bash
.rowToBlock/bin/python -m block_level block_level/server.json
```

The CEB sweep uses the same parameters for `ceb_imdb` and `ceb_stack`:

```bash
.rowToBlock/bin/python -m block_level block_level/ceb.json
```

Prepare the CEB Stack DuckDB database, partition IDs, and generated query files
using the commands in the source project's README before running `ceb.json`.

For a fresh environment, install `block_level/requirements.txt` first.

The sole positional argument is a JSON experiment config. Relative database,
query, and output paths are resolved relative to that config file. When
`output_dir` is omitted, a timestamped `<workload_name>_block_skipping_sweep_*`
directory is created next to the config. A config may list multiple workloads
under `workloads`; shared sweep settings apply to each workload in order, and
each gets its own output directory and summary. Each workload entry needs
`workload_name` and `database_path`; `query_source_dir` defaults to the database
directory. A top-level `output_dir` places each workload in a named subdirectory.
Single-workload configs remain supported. Use `--validate-only` to validate the
configuration and input paths without executing the experiment:

```bash
.rowToBlock/bin/python -m block_level block_level/server.json --validate-only
```

`partition_database` defaults to `false`. Set it to `true` only when the source
DuckDB database does not already have the notebook's consecutive `partition_id`
columns; this option rewrites every base table in the database.

Every parameter combination gets one flat `config_NNNN/` directory containing
its fingerprint metadata, query-level results, summaries, plots, experiment
manifest, and a `parameters.json` snapshot of its settings. The sweep root
contains copied workload files, the resolved configuration, and
`fingerprint_sweep_summary.csv`.

Query files named `<table>.<column>_queries.txt` may contain either legacy
comma-separated needle rows or `block-skipping-query-v1` JSON lines. For JSON
lines, ground truth runs the supplied SQL `predicate` exactly. Fingerprint
probes use the `lookup_needles` required by `individual_needle` and
`all_needles` predicates; when there are no safe required needles (including
`actual_predicate`), all partitions remain candidates. Multi-needle probes
require all needle fingerprints to occur in the same stored matrix row.

## Sub-block fingerprint matrices

`fingerprint_subblock_joint_entropy_equivalence_classes` is the matrix-valued
variant of the block fingerprint. It splits each physical block into
consecutive, row-aligned sub-blocks. Every selected feature contributes a joint
occurrence vector with one zero/one value per sub-block. The empirical
distribution of these vectors is taken across selected features, and selection
greedily maximizes its categorical entropy. With one sub-block per physical
block, the vectors are scalar zero/one values and the objective is exactly the
internal-entropy objective.

The sub-block variant supports two feature-selection scopes:

- `local`: every physical block independently selects and stores its own feature
  groups. Queries are encoded separately with each block's mapping before the
  containment test.
- `global`: one feature mapping is shared by every block. For the sub-block
  variant, candidate scores sum the within-block joint entropies without mixing
  sub-block vectors across physical blocks.

`fingerprint_internal_entropy_equivalence_classes` also accepts an optional
`subblock_size_rows`. Without it, the method retains its original behavior: it
builds one fingerprint per physical block and globally maximizes the sum of
the fingerprints' binary internal entropies. With `subblock_size_rows`, every
sub-block is treated as an independent block for that same objective, and the
stored representation is a sub-block matrix. This is distinct from the joint
entropy method above: internal entropy balances the zero/one bits within each
sub-block fingerprint rather than maximizing the entropy of complete
occurrence vectors over sub-block positions.

The sub-block internal-entropy form supports both scopes. With `global`, all
sub-blocks across all physical blocks select one shared mapping. With `local`,
each physical block selects its own mapping using only its own sub-blocks,
including local candidate-frequency filtering and local equivalence classes.
The original form without a sub-block division remains global-only.

In a sweep, selecting `fingerprint_internal_entropy_equivalence_classes` and
providing `subblock_sizes_rows` enables this matrix form for every listed size.
Omit `subblock_sizes_rows` to run the original physical-block form.

Set `sweep.feature_selection_scope` to `local` or `global` for one scope. To
include both in one sweep, set `sweep.feature_selection_scopes` to
`["global", "local"]` instead.

Matrix sweep versions are written as two aligned Parquet files in the applicable
`config_NNNN/` directory: `partition_metadata_vNNN.parquet` and
`feature_mappings_vNNN.parquet`. The first has the same
`table_name, column_name, partition_id, metadata: list<list<bool>>` schema as
the matrix-compression project's frozen partition metadata. Each record is one
physical partition, and each nested row is one distinct sub-block mask. The
mapping file has the same keys in the same order and stores
`feature_groups: list<list<string>>`: outer position is the matrix column;
inner strings are equivalent n-grams that set that bit. Local scope stores
the selected mapping for each partition; global scope repeats its shared
mapping in every partition record. Unused columns are empty groups. Reported
matrix metadata size includes only `partition_metadata_vNNN.parquet`; the
feature mapping file is excluded. The original single-mask
(non-matrix) configurations still write JSON.

When entropy scores tie, selection prefers the n-gram present in more
selection units, then the lexicographically smaller n-gram. For local
sub-block selection, commonality is counted within the current physical
block; for global selection, it is counted across all sub-blocks. N-grams
with identical occurrence vectors share one feature group, whose
lexicographically smallest member is its representative.

Both matrix forms store one matrix row per distinct sub-block mask. During
joint-entropy selection every sub-block remains a coordinate of the joint
occurrence vectors; during internal-entropy selection it is an independent
entropy unit. Exact duplicate matrix rows are removed independently at every
fingerprint width afterward; this is a lossless storage and probing
optimization. A block is a candidate when the query mask is a subset of at
least one row in its matrix.

Configure the sub-block size in rows with `sweep.subblock_sizes_rows`. Each size
creates a separate sweep point and must be between 1 and `block_size_rows`,
inclusive. A full block has `ceil(block_size_rows / subblock_size_rows)` matrix
rows before duplicate removal. For this method the existing min/max
block-frequency cutoffs are applied to sub-block occurrence frequency.

```json
{
  "block_size_rows": 16384,
  "sweep": {
    "feature_selection_methods": [
      "fingerprint_subblock_joint_entropy_equivalence_classes"
    ],
    "widths_by_method": {
      "fingerprint_subblock_joint_entropy_equivalence_classes": [6000, 8000]
    },
    "ngram_sizes": [3],
    "min_block_frequencies_by_method": {
      "fingerprint_subblock_joint_entropy_equivalence_classes": [0.0]
    },
    "max_block_frequencies": [1.0],
    "feature_selection_scope": "local",
    "subblock_sizes_rows": [1, 256, 1024, 16384]
  }
}
```
