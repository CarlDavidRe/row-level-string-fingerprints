# Block-level fingerprint sweeps

This package is the scriptable form of `ceb_imdb_block_skipping_portable.ipynb`.
It separates configuration, feature selection/fingerprint construction, workload
evaluation, result export, and sweep orchestration. The original notebook is kept
in this directory as a reference.

Run the current notebook-equivalent sweep with:

```bash
.rowToBlock/bin/python -m block_level block_level/example_config.json
```

For a fresh environment, install `block_level/requirements.txt` first.

The sole positional argument is a JSON experiment config. Relative database,
query, and output paths are resolved relative to that config file. When
`output_dir` is omitted, a timestamped `ceb_imdb_block_skipping_sweep_*`
directory is created next to the config. Use `--validate-only` to validate the
configuration and input paths without executing the experiment:

```bash
.rowToBlock/bin/python -m block_level block_level/example_config.json --validate-only
```

`partition_database` defaults to `false`. Set it to `true` only when the source
DuckDB database does not already have the notebook's consecutive `partition_id`
columns; this option rewrites every base table in the database.

Every parameter combination gets its own directory with fingerprint metadata,
query-level results, summaries, plots, and an experiment manifest. The sweep
root contains copied workload files, the resolved configuration, and
`fingerprint_sweep_summary.csv`.

## Sub-block joint-entropy matrices

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

`fingerprint_internal_entropy_equivalence_classes` remains global-only. A local
scope is rejected for that method.

Set `sweep.feature_selection_scope` to `local` or `global` in the JSON
experiment configuration.

The representation stores one matrix row per distinct sub-block mask. During
feature selection every sub-block remains a coordinate of the joint occurrence
vectors. Exact duplicate matrix rows are removed independently at every
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
