"""A notebook must not label another checkpoint's recording as this result."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bdxr_classroom.lesson.runtime import Lesson


class RecordingIdentityTests(unittest.TestCase):
    def test_rejects_weights_changed_after_assessment_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            run = work / "experiments/run"
            (run / "checkpoints").mkdir(parents=True)
            weights = run / "checkpoints/final.pt"
            weights.write_bytes(b"assessed weights")
            assessed_digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            weights.write_bytes(b"replacement weights")
            lab = Lesson(work, "unused", dict(os.environ))
            with patch.object(lab, "run") as inference:
                with self.assertRaisesRegex(RuntimeError, "changed since assessment"):
                    lab.record(run, expected_sha256=assessed_digest)
                inference.assert_not_called()

    def test_rejects_a_recording_with_another_checkpoint_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            run = work / "experiments/run"
            (run / "checkpoints").mkdir(parents=True)
            weights = run / "checkpoints/final.pt"
            weights.write_bytes(b"test saved weights")
            output = run / "evaluation/eval_final"
            output.mkdir(parents=True)
            video = run / "video.mp4"
            video.write_bytes(b"video fixture")
            report = {
                "checkpoint_sha256": "wrong",
                "video_path": str(video),
                "training_step": 10000,
            }
            (output / "evaluation.json").write_text(json.dumps(report))
            (work / "configs").mkdir()
            for name in ("head_motion.json", "game_head_motion.json"):
                (work / "configs" / name).write_text("{}")
            lab = Lesson(work, "unused", dict(os.environ))
            with patch.object(lab, "run") as inference:
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    lab.record(run)
                report["checkpoint_sha256"] = hashlib.sha256(
                    weights.read_bytes()
                ).hexdigest()
                (output / "evaluation.json").write_text(json.dumps(report))
                self.assertEqual(lab.record(run), video)
                self.assertIn(
                    str(work / "configs/head_motion.json"), inference.call_args.args
                )
                self.assertNotIn(
                    str(work / "configs/game_head_motion.json"),
                    inference.call_args.args,
                )
                self.assertEqual(
                    lab.presentation_arguments(game=True),
                    ["--head-motion-config", str(work / "configs/game_head_motion.json")],
                )
