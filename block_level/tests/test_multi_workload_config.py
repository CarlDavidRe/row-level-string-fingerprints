from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from block_level.config import ExperimentConfig
from block_level.experiment import (
    FingerprintEvaluator,
    ResultExporter,
    SweepRunner,
)


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

    def test_sweep_points_use_flat_numbered_directories_with_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "database_path": "job.duckdb",
                "workload_name": "job",
                "output_dir": "results",
                "sweep": SWEEP,
            }))
            config = ExperimentConfig.load(path)
            fingerprint = next(config.sweep.experiments())
            runner = SweepRunner(config)
            run_dir = runner.run_directory(1)

            self.assertEqual(run_dir, Path(directory) / "results/config_0001")
            parameters_path = runner.write_parameters(run_dir, fingerprint)
            parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
            self.assertEqual(parameters_path, run_dir / "parameters.json")
            self.assertEqual(
                parameters["feature_selection_method"],
                "local_split_entropy",
            )
            self.assertEqual(parameters["widths"], [8])
            evaluator = FingerprintEvaluator(config, fingerprint, [], run_dir)
            exporter = ResultExporter(config, fingerprint, run_dir)
            for component in (evaluator, exporter):
                self.assertEqual(component.output_dir, run_dir)
                self.assertEqual(component.fingerprint_dir, run_dir)


if __name__ == "__main__":
    unittest.main()
