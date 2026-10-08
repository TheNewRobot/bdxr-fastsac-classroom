"""Reject invalid metrics and stale checkpoint reports before automatic loading."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("walking_checks", ROOT / "scripts/check_command_results.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WalkingSelection(unittest.TestCase):
    def test_selects_passing_continuation_and_rejects_replaced_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"checkpoint identity fixture")
            good = dict(checkpoint=str(path), passed=True,
                        checkpoint_sha256=module.checkpoint_digest(path))
            bad = dict(good, passed=False)
            self.assertEqual(module.select_checkpoint([bad, good]), str(path))
            path.write_bytes(b"new weights have not been evaluated")
            self.assertIsNone(module.select_checkpoint([good]))

    def test_nonfinite_motion_cannot_pass_the_gate(self):
        targets = json.loads((ROOT / "configs/evaluation_targets.json").read_text())
        row = dict(duration_s=15, speed_mae=0, yaw_mae=0, max_cross_track_m=0, fell=False)
        summary = dict(checkpoint="fixture", cases={
            case: [dict(row) for _ in range(5)] for case in targets["required_cases"]})
        self.assertTrue(module.check(summary, targets)["passed"])
        summary["cases"]["fast"][0]["max_cross_track_m"] = float("nan")
        self.assertFalse(module.check(summary, targets)["passed"])
