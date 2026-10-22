"""Keep failed hopping stages out of automatic game selection."""

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class HoppingStageSelectionTests(unittest.TestCase):
    def run_selection(self, verdicts):
        from bdxr_classroom.lesson.hopping import select_hopper

        stages = [Path(f"/runs/stage_{i}") for i in range(8)]
        assessor = Mock(side_effect=verdicts)
        lab = SimpleNamespace(
            learning_curve=Mock(return_value="/curve.png"),
            record=Mock(return_value="/selected.mp4"),
        )
        state = dict(
            Path=Path,
            HOP_STAGES=stages,
            SELECTED_HOP_CHECKPOINT=None,
            SELECTED_WALK_CHECKPOINT="/walker.pt",
            validate_pair=assessor,
            lab=lab,
            read_json=lambda _: {
                "candidates": [
                    {"checkpoint": value, "checkpoint_sha256": "validated-hash"}
                    for value in verdicts
                    if value
                ]
            },
        )
        with (
            patch("bdxr_classroom.lesson.hopping.validate_pair", assessor),
            patch("bdxr_classroom.lesson.presentation.read_json", state["read_json"]),
            patch("IPython.display.Image"),
            patch("IPython.display.Video"),
            patch("IPython.display.display"),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            state["SELECTED_HOP_CHECKPOINT"] = select_hopper(lab, stages, "/walker.pt")
        return state, assessor, output.getvalue()

    def test_selects_qualified_earlier_stage_and_its_curve(self):
        selected = "/runs/stage_5/checkpoints/step_001000.pt"
        state, assessor, _ = self.run_selection([None, None, selected])
        self.assertEqual(
            [c.args[1] for c in assessor.call_args_list],
            [Path("/runs/stage_7"), Path("/runs/stage_6"), Path("/runs/stage_5")],
        )
        self.assertEqual(state["SELECTED_HOP_CHECKPOINT"], selected)
        state["lab"].learning_curve.assert_called_once_with(Path("/runs/stage_5"))
        state["lab"].record.assert_called_once_with(
            Path("/runs/stage_5"),
            checkpoint="step_001000",
            expected_sha256="validated-hash",
        )

    def test_stops_after_first_qualified_stage(self):
        state, assessor, _ = self.run_selection(["/runs/stage_7/checkpoints/final.pt"])
        self.assertEqual(assessor.call_count, 1)

    def test_no_qualified_stage_does_not_substitute_a_policy(self):
        state, assessor, output = self.run_selection([None, None, None])
        self.assertEqual(assessor.call_count, 3)
        self.assertIsNone(state["SELECTED_HOP_CHECKPOINT"])
        state["lab"].learning_curve.assert_not_called()
        state["lab"].record.assert_not_called()
        self.assertIn("No hopper passed every check", output)
