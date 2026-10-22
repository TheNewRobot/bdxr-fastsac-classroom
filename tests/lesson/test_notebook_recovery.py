"""Check recovery routing without pretending mock policies establish quality."""

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from bdxr_classroom.lesson.assessment import drift_only_failure


class NotebookRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[2]
            / "notebooks/sections/06_train_and_evaluate.py"
        )
        cls.source = "\n".join(
            cell
            for cell in path.read_text().split("# %%")
            if cell.lstrip().startswith(
                (
                    "# @title Additional recovery only if needed",
                    "SELECTED_WALK_CHECKPOINT =",
                )
            )
        )

    def state(self, already_passed=False, recovery_passed=True):
        checkpoint = "/work/new_recovery/checkpoints/step_002000.pt"
        return {
            "ENV_CONFIG": {"yaw_bias_weight": 6.0, "yaw_tracking": "imu"},
            "LEARNING_CONFIG": {"seed": 7, "steps": 10000, "minutes": 15},
            "INITIAL_BASELINE": Path("/work/original"),
            "BASELINE": Path("/work/first_recovery"),
            "BASELINE_CHECK": {"passed": already_passed, "checkpoint": checkpoint},
            "INITIAL_BASELINE_CHECK": {
                "passed": False,
                "failures": ["fast: mean drift too large"],
            },
            "drift_only_failure": drift_only_failure,
            "BASELINE_SUMMARY": {},
            "lab": SimpleNamespace(train=Mock(return_value=Path("/work/new_recovery"))),
            "evaluate_walking": Mock(return_value=({}, {"passed": False})),
            "check_recent_walking": Mock(
                return_value=(
                    {},
                    {"passed": recovery_passed, "checkpoint": checkpoint},
                )
            ),
            "selected_walking": Mock(
                side_effect=lambda reports: (
                    reports[0]["checkpoint"] if reports[0]["passed"] else None
                )
            ),
            "command_table": Mock(),
        }

    def execute(self, state):
        with contextlib.redirect_stdout(io.StringIO()):
            # Execute the repository's authored cell, never external notebook input.
            exec(compile(self.source, "notebook_recovery_cell", "exec"), state)  # noqa: S102

    def test_retry_uses_original_parent_and_preserves_initial_settings(self):
        state = self.state()
        self.execute(state)
        call = state["lab"].train.call_args
        self.assertEqual(
            call.kwargs["parent"], Path("/work/original/checkpoints/final.pt")
        )
        self.assertEqual(call.args[1]["yaw_bias_weight"], 10)
        self.assertEqual(call.args[1]["yaw_tracking"], "imu")
        self.assertEqual(call.args[2]["steps"], 3000)
        self.assertEqual(state["ENV_CONFIG"]["yaw_bias_weight"], 6)
        self.assertEqual(state["LEARNING_CONFIG"]["steps"], 10000)
        self.assertEqual(state["BASELINE"], Path("/work/new_recovery"))
        self.assertTrue(state["SELECTED_WALK_CHECKPOINT"].endswith("step_002000.pt"))

    def test_passing_policy_does_not_trigger_more_training(self):
        state = self.state(already_passed=True)
        self.execute(state)
        state["lab"].train.assert_not_called()
        state["evaluate_walking"].assert_not_called()
        state["check_recent_walking"].assert_not_called()

    def test_failed_extra_attempt_is_not_promoted_or_repeated(self):
        state = self.state(recovery_passed=False)
        with self.assertRaisesRegex(RuntimeError, "No checked snapshot passed"):
            self.execute(state)
        state["lab"].train.assert_called_once()
        self.assertIsNone(state["SELECTED_WALK_CHECKPOINT"])

    def test_unbalanced_parent_does_not_trigger_yaw_refinement(self):
        state = self.state()
        state["INITIAL_BASELINE_CHECK"]["failures"].append("stop: fall")
        with self.assertRaisesRegex(RuntimeError, "No checked snapshot passed"):
            self.execute(state)
        state["lab"].train.assert_not_called()

    def test_missing_or_non_drift_failures_are_not_eligible(self):
        for failures in ([], ["fast: speed_mae exceeds 0.05"], ["missing case: slow"]):
            self.assertFalse(
                drift_only_failure({"passed": False, "failures": failures})
            )
