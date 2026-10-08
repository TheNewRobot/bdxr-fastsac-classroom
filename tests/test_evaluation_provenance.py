"""Repeated evaluations must retain the evidence behind earlier selections."""

import json
from pathlib import Path
import tempfile
import unittest

from bdxr_classroom.command_evaluate import preserve_previous_evaluation, checkpoint_hash


class EvaluationProvenanceTests(unittest.TestCase):
    def test_repeat_preserves_failed_report_and_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evaluation" / "final_commands"
            video = root / "videos" / "final_commands"
            output.mkdir(parents=True)
            video.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps({"passed": False}))
            (video / "forward.mp4").write_bytes(b"previous recording")
            preserve_previous_evaluation(output, video)
            archived = list(output.parent.glob("final_commands_previous_*"))
            self.assertEqual(len(archived), 1)
            self.assertFalse(json.loads((archived[0] / "summary.json").read_text())["passed"])
            archived_video = video.parent / archived[0].name / "forward.mp4"
            self.assertEqual(archived_video.read_bytes(), b"previous recording")
            self.assertFalse(output.exists())
            self.assertFalse(video.exists())

    def test_digest_identifies_weights_independent_of_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "final.pt", Path(directory) / "step.pt"
            a.write_bytes(b"saved weights")
            b.write_bytes(a.read_bytes())
            self.assertEqual(checkpoint_hash(a), checkpoint_hash(b))
            b.write_bytes(b"new weights")
            self.assertNotEqual(checkpoint_hash(a), checkpoint_hash(b))


if __name__ == "__main__":
    unittest.main()
