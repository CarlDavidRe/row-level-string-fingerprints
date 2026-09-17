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
