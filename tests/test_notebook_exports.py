"""Protect downloadable training lineage and optional replay backups."""

import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile


class NotebookExports(unittest.TestCase):
    def test_all_training_stages_and_replay_survive_export(self):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "notebooks/BDXR_Walking_101.ipynb").read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and
                      "def export_results" in "".join(cell["source"]))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "export_results")
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            runs = []
            for name in ("command_walking_recovery", "variation", "control", "tiny", "moving", "terrain",
                         "desert_droid_game", "transition_validation", "initial",
                         "level_hops", "level_terrain_hops", "natural_hops"):
                run = work / "experiments" / ("2026-01-01_" + name)
                (run / "checkpoints").mkdir(parents=True)
                # File-preservation fixtures, not purported trained policies.
                (run / "checkpoints/final.pt").write_bytes(b"policy fixture")
                (run / "checkpoints/resume.pt").write_bytes(b"replay fixture")
                (run / "wandb").mkdir()
                (run / "wandb/private.log").write_text("excluded fixture")
                runs.append(run)
            namespace = dict(WORK=work, zipfile=zipfile, INCLUDE_REPLAY_IN_EXPORT=True,
                             INITIAL_BASELINE=runs[8],
                             HOP_STAGES=runs[3:6] + runs[9:],
                             **dict(zip(("BASELINE", "VARIATION", "CONTROL", "TINY_HOPS",
                                         "MOVING_HOPS", "HOP_RESULTS"), runs[:6])))
            namespace["HOP_RESULTS"] = runs[11]
            exec(compile(ast.Module(body=[function], type_ignores=[]), "<export-cell>", "exec"),
                 namespace)
            colab = types.ModuleType("google.colab")
            colab.files = types.SimpleNamespace(download=lambda _: None)
            with patch.dict("sys.modules", {"google.colab": colab}):
                backup = namespace["export_results"]()
                compact = namespace["export_results"](False)
            with zipfile.ZipFile(backup) as archive:
                for run in runs:
                    prefix = run.relative_to(work).as_posix()
                    self.assertEqual(archive.read(prefix + "/checkpoints/resume.pt"),
                                     b"replay fixture")
                    self.assertIn(prefix + "/checkpoints/final.pt", archive.namelist())
                self.assertFalse(any("wandb/" in path for path in archive.namelist()))
            with zipfile.ZipFile(compact) as archive:
                self.assertFalse(any(path.endswith("resume.pt") for path in archive.namelist()))
                self.assertEqual(len(archive.namelist()), len(runs))
            previous_backup = backup.read_bytes()
            with patch.object(zipfile.ZipFile, "write", side_effect=OSError("interrupted export")):
                with self.assertRaisesRegex(OSError, "interrupted export"):
                    namespace["export_results"]()
            self.assertEqual(backup.read_bytes(), previous_backup)
