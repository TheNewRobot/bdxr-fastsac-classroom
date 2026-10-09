"""Filesystem-only checks: organization must not break saved policy paths."""

import json
from pathlib import Path
import tempfile
import unittest

from bdxr_classroom.experiments import (
    create_experiment, experiment_root, resolve_experiment_path,
)


class ExperimentLayoutTests(unittest.TestCase):
    def test_historical_checkpoint_and_config_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiments"
            checkpoint = root / "lesson/old_run/checkpoints/final.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"unchanged weights")
            (root / ".metadata").mkdir()
            (root / ".metadata/relocations.json").write_text(
                json.dumps({"old_run": "lesson/old_run"}))
            old = root / "old_run/checkpoints/final.pt"
            self.assertEqual(resolve_experiment_path(old), checkpoint)
            self.assertEqual(experiment_root(old), checkpoint.parent.parent)
            self.assertEqual(resolve_experiment_path(checkpoint), checkpoint)
            unknown = root / "unknown/checkpoints/final.pt"
            self.assertEqual(resolve_experiment_path(unknown), unknown)

    def test_new_runs_grouped_custom_trial_roots_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiments"
            train = create_experiment("walk", root, category="candidates")
            evaluation = create_experiment("course", root)
            nested = create_experiment("seed7", evaluation / "evaluation/trials")
            self.assertEqual(train.parent, root / "candidates")
            self.assertEqual(evaluation.parent, root / "evaluations")
            self.assertEqual(nested.parent, evaluation / "evaluation/trials")
            self.assertTrue((train / "checkpoints").is_dir())


if __name__ == "__main__":
    unittest.main()
