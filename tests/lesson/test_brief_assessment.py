"""The brief assessment must retain failures and never trigger extra training."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bdxr_classroom.evaluation.walking_checks import checkpoint_digest
from bdxr_classroom.lesson.brief import assess_walking

ROOT = Path(__file__).resolve().parents[2]


class BriefAssessmentTests(unittest.TestCase):
    def assess(self, speed_error=0.02, drift=0.1, wrong_hash=False):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "configs").mkdir()
            (work / "configs/evaluation_targets.json").write_bytes(
                (ROOT / "configs/evaluation_targets.json").read_bytes()
            )
            run = work / "run"
            (run / "checkpoints").mkdir(parents=True)
            weights = run / "checkpoints/final.pt"
            weights.write_bytes(b"unchanged policy fixture")
            (run / "metrics").mkdir()
            timing = {
                "step": 12000,
                "session_training_seconds": 720,
                "hardware": "test GPU",
            }
            (run / "metrics/timing.json").write_text(json.dumps(timing))

            def evaluate(module, *arguments):
                self.assertEqual(module, "bdxr_classroom.evaluation.command_evaluate")
                self.assertEqual(arguments[arguments.index("--cases") + 1], "forward")
                folder = Path(arguments[arguments.index("--output") + 1])
                folder.mkdir(parents=True)
                row = {
                    "fell": False,
                    "duration_s": 10,
                    "speed_mae": speed_error,
                    "yaw_mae": 0.02,
                    "max_cross_track_m": drift,
                }
                summary = {
                    "checkpoint": str(weights),
                    "checkpoint_sha256": "wrong"
                    if wrong_hash
                    else checkpoint_digest(weights),
                    "cases": {"forward": [row.copy() for _ in range(3)]},
                }
                (folder / "summary.json").write_text(json.dumps(summary))

            lab = SimpleNamespace(
                work=work,
                run=Mock(side_effect=evaluate),
                presentation_arguments=list,
                train=Mock(),
            )
            with (
                patch("bdxr_classroom.lesson.brief.command_table"),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                verdict = assess_walking(lab, run)
            lab.run.assert_called_once()
            lab.train.assert_not_called()
            self.assertEqual(verdict["training_context"], timing)
            self.assertIn("12,000 steps in 720s on test GPU", output.getvalue())
            self.assertEqual(weights.read_bytes(), b"unchanged policy fixture")
            self.assertGreaterEqual(verdict["wall_seconds"], 0)
            return verdict, output.getvalue()

    def test_good_forward_trials_pass_without_claiming_full_validation(self):
        verdict, output = self.assess()
        self.assertTrue(verdict["passed"])
        self.assertIn("not full command validation", verdict["scope"])
        self.assertIn("does not establish turning or terrain capability", output)

    def test_drift_and_tracking_failures_remain_visible(self):
        verdict, output = self.assess(speed_error=0.09, drift=0.8)
        self.assertFalse(verdict["passed"])
        self.assertIn("forward: speed_mae exceeds 0.05", verdict["failures"])
        self.assertIn("forward: mean drift too large", verdict["failures"])
        self.assertIn("Stayed upright", output)
        self.assertEqual(verdict["targets"]["mean_max_cross_track_limit_m"], 0.2)

    def test_mismatched_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            self.assess(wrong_hash=True)
