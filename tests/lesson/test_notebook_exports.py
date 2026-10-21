"""Protect complete, deduplicated, atomic backups including replay."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from bdxr_classroom.lesson.exports import export_results


class NotebookExports(unittest.TestCase):
    def test_setup_evidence_is_included_without_environment_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "runtime_info.json").write_text('{"gpu":"test"}')
            (work / ".lesson_release.json").write_text('{"sha256":"test"}')
            (work / ".env").write_text("PRIVATE=not-for-export")
            logs = work / "artifacts/logs"
            logs.mkdir(parents=True)
            (logs / "failure.log").write_text("diagnostic")
            backup = export_results(work, [])
            with zipfile.ZipFile(backup) as archive:
                self.assertIn("runtime_info.json", archive.namelist())
                self.assertIn(".lesson_release.json", archive.namelist())
                self.assertIn("artifacts/logs/failure.log", archive.namelist())
                self.assertNotIn(".env", archive.namelist())

    def test_all_training_stages_and_replay_survive_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            runs = []
            for name in (
                "baseline",
                "variation",
                "control",
                "tiny",
                "moving",
                "terrain",
                "level",
                "level_terrain",
                "natural",
                "blocks",
                "soft_landing",
            ):
                run = work / "experiments/candidates" / ("2026-01-01_" + name)
                (run / "checkpoints").mkdir(parents=True)
                (run / "checkpoints/final.pt").write_bytes(b"policy fixture")
                (run / "checkpoints/resume.pt").write_bytes(b"replay fixture")
                (run / "wandb").mkdir()
                (run / "wandb/private.log").write_text("excluded fixture")
                runs.append(run)
            backup = export_results(work, runs)
            compact = export_results(work, runs, include_replay=False)
            with zipfile.ZipFile(backup) as archive:
                for run in runs:
                    prefix = run.relative_to(work).as_posix()
                    self.assertEqual(
                        archive.read(prefix + "/checkpoints/resume.pt"),
                        b"replay fixture",
                    )
                    self.assertIn(prefix + "/checkpoints/final.pt", archive.namelist())
                self.assertFalse(any("wandb/" in path for path in archive.namelist()))
            with zipfile.ZipFile(compact) as archive:
                self.assertFalse(
                    any(path.endswith("resume.pt") for path in archive.namelist())
                )
                self.assertEqual(len(archive.namelist()), len(runs))
            previous_backup = backup.read_bytes()
            with patch.object(
                zipfile.ZipFile, "write", side_effect=OSError("interrupted export")
            ):
                with self.assertRaisesRegex(OSError, "interrupted export"):
                    export_results(work, runs)
            self.assertEqual(backup.read_bytes(), previous_backup)

    def test_nested_trials_and_game_are_exported_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            run = work / "experiments/candidates/2026_walk"
            trial = (
                run
                / "evaluation/2026_transition_validation_suite/evaluation/2026_transition_validation"
            )
            game = work / "experiments/evaluations/courses/2026_desert_droid_game"
            files = {
                run / "checkpoints/resume.pt": b"replay",
                trial / "evaluation/result.json": b'{"passed":true}',
                game / "metrics/gameplay.jsonl": b'{"time":1}',
            }
            for path, value in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            backup = export_results(work, [run, trial, run])
            with zipfile.ZipFile(backup) as archive:
                self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
                self.assertEqual(len(archive.namelist()), len(files))
                for path, value in files.items():
                    self.assertEqual(
                        archive.read(path.relative_to(work).as_posix()), value
                    )
