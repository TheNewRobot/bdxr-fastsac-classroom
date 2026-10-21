"""Acceptance must follow the evaluated snapshot, including earlier saved steps."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bdxr_classroom.lesson.assessment import check_recent_walking, evaluate_walking

ROOT = Path(__file__).resolve().parents[2]


class AssessmentIdentityTests(unittest.TestCase):
    def test_snapshot_search_is_bounded_and_never_promotes_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "metrics").mkdir()
            (run / "metrics/timing.json").write_text('{"step":10000}')
            (run / "checkpoints").mkdir()
            for step in (7000, 8000, 9000, 10000):
                (run / "checkpoints" / f"step_{step:06d}.pt").touch()
            failed = {"passed": False, "failures": ["backward: mean drift too large"]}
            lab = Mock()
            with patch(
                "bdxr_classroom.lesson.assessment.evaluate_walking",
                return_value=({}, failed),
            ) as evaluate:
                _, result = check_recent_walking(lab, run, {}, failed)
                self.assertFalse(result["passed"])
                self.assertEqual(
                    [call.kwargs["checkpoint"] for call in evaluate.call_args_list],
                    ["step_009000", "step_008000"],
                )
                evaluate.reset_mock()
                check_recent_walking(lab, run, {}, {"passed": True})
                evaluate.assert_not_called()

    def test_earlier_snapshot_and_wrong_weights_cannot_be_confused(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "configs").mkdir()
            targets = json.loads((ROOT / "configs/evaluation_targets.json").read_text())
            (work / "configs/evaluation_targets.json").write_text(json.dumps(targets))
            run = work / "run"
            (run / "checkpoints").mkdir(parents=True)
            weights = run / "checkpoints/step_009000.pt"
            weights.write_bytes(b"earlier saved policy")
            (run / "checkpoints/final.pt").write_bytes(b"different final policy")
            folder = run / "evaluation/step_009000_commands"
            folder.mkdir(parents=True)
            # The full policy suite is required even when selecting an earlier step.
            row = dict(
                duration_s=15,
                speed_mae=0.01,
                yaw_mae=0.01,
                max_cross_track_m=0.02,
                fell=False,
            )
            summary = dict(
                checkpoint=str(weights),
                checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                cases={case: [row] * 5 for case in targets["required_cases"]},
            )
            report = folder / "summary.json"
            report.write_text(json.dumps(summary))
            lab = Mock(work=work)
            lab.presentation_arguments.return_value = []
            _, verdict = evaluate_walking(lab, run, checkpoint="step_009000")
            self.assertTrue(verdict["passed"])
            self.assertEqual(verdict["checkpoint"], str(weights))
            self.assertEqual(verdict["evaluation_directory"], str(folder))
            self.assertEqual(lab.run.call_args.args[1], weights)
            summary["checkpoint_sha256"] = "stale evaluation of another policy"
            report.write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError, "different weights"):
                evaluate_walking(lab, run, checkpoint="step_009000")
