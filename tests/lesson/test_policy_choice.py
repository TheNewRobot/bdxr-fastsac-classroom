"""Only current, passing student checkpoints may be selected automatically."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from bdxr_classroom.evaluation.walking_checks import checkpoint_digest
from bdxr_classroom.lesson.policy_choice import choose_walking_policy


class PolicyChoiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runs, self.checks = {}, {}
        for name in ("baseline", "variation"):
            run = Path(self.temporary.name) / name
            (run / "checkpoints").mkdir(parents=True)
            (run / "evaluation").mkdir()
            checkpoint = run / "checkpoints/final.pt"
            checkpoint.write_bytes(name.encode())
            digest = checkpoint_digest(checkpoint)
            (run / "evaluation/summary.json").write_text(json.dumps({
                "checkpoint_sha256": digest,
                "cases": {"forward": [{"mean_speed": 0.27}] * 3},
            }))
            self.runs[name] = run
            self.checks[name] = {
                "passed": True, "failures": [],
                "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                "evaluation_directory": str(run / "evaluation"),
            }

    def choose(self, preference="auto"):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            selected = choose_walking_policy(self.runs, self.checks, preference)
        return selected, output.getvalue()

    def test_failed_baseline_selects_passing_variation(self):
        self.checks["baseline"].update(
            passed=False, failures=["forward: speed_mae exceeds 0.05"]
        )
        selected, text = self.choose()
        self.assertEqual(selected, str(self.runs["variation"] / "checkpoints/final.pt"))
        self.assertIn("Baseline: FAIL", text)
        self.assertIn("Variation: PASS", text)

    def test_both_pass_prefer_baseline_but_allow_variation(self):
        self.assertEqual(self.choose()[0], self.checks["baseline"]["checkpoint"])
        self.assertEqual(self.choose("variation")[0], self.checks["variation"]["checkpoint"])

    def test_neither_passes_does_not_select_weights(self):
        for check in self.checks.values():
            check["passed"] = False
        selected, text = self.choose()
        self.assertIsNone(selected)
        self.assertIn("The game will not start", text)

    def test_manual_failed_choice_is_not_silently_replaced(self):
        self.checks["baseline"]["passed"] = False
        self.assertIsNone(self.choose("baseline")[0])

    def test_changed_weights_cannot_reuse_old_pass(self):
        (self.runs["baseline"] / "checkpoints/final.pt").write_bytes(b"retrained")
        self.assertEqual(self.choose()[0], self.checks["variation"]["checkpoint"])
        self.assertIn("CHECK OUTDATED", self.choose()[1])

    def test_missing_assessment_requires_check(self):
        self.checks["baseline"] = None
        self.assertEqual(self.choose()[0], self.checks["variation"]["checkpoint"])

    def test_mismatched_summary_cannot_qualify(self):
        path = self.runs["baseline"] / "evaluation/summary.json"
        summary = json.loads(path.read_text())
        summary["checkpoint_sha256"] = "different"
        path.write_text(json.dumps(summary))
        self.assertEqual(self.choose()[0], self.checks["variation"]["checkpoint"])
