from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from block_level.config import ExperimentConfig


SWEEP = {
    "feature_selection_methods": ["local_split_entropy"],
    "widths_by_method": {"local_split_entropy": [8]},
    "ngram_sizes": [3],
    "min_block_frequencies_by_method": {"local_split_entropy": [0.0]},
}


class MultiWorkloadConfigTests(unittest.TestCase):
    def test_shared_sweep_and_separate_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "sweep": SWEEP,
                "output_dir": "results",
                "workloads": [
                    {"workload_name": "job", "database_path": "job.duckdb"},
                    {"workload_name": "tpch_sf10", "database_path": "tpch.duckdb"},
                ],
            }))
            job, tpch = ExperimentConfig.load_many(path)
            self.assertEqual(job.output_dir, Path(directory) / "results/job")
            self.assertEqual(tpch.output_dir, Path(directory) / "results/tpch_sf10")
            self.assertEqual(job.query_source_dir, Path(directory))
            self.assertEqual(job.sweep, tpch.sweep)
            with self.assertRaisesRegex(ValueError, "multiple workloads"):
                ExperimentConfig.load(path)

    def test_single_workload_config_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "database_path": "job.duckdb",
                "workload_name": "job",
                "sweep": SWEEP,
            }))
            self.assertEqual(ExperimentConfig.load(path).workload_name, "job")


if __name__ == "__main__":
    unittest.main()
