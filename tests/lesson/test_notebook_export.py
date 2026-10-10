"""Execute the lesson's export function with nested validation evidence."""

import ast
from pathlib import Path
import tempfile
import unittest
import zipfile


class NotebookExportTests(unittest.TestCase):
    def test_nested_trials_are_exported_once_with_replay(self):
        builder = Path(__file__).resolve().parents[2] / "scripts/lesson/build_notebook.py"
        source = next(node.value for node in ast.walk(ast.parse(builder.read_text()))
                      if isinstance(node, ast.Constant) and isinstance(node.value, str)
                      and node.value.startswith("INCLUDE_REPLAY_IN_EXPORT ="))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "export_results")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "experiments/candidates/2026_walk"
            trial = run / "evaluation/2026_transition_validation_suite/evaluation/2026_transition_validation"
            game = root / "experiments/evaluations/courses/2026_desert_droid_game"
            expected = {}
            for path, data in [
                (run / "checkpoints/resume.pt", b"replay and networks"),
                (trial / "evaluation/result.json", b'{"passed": true}'),
                (game / "metrics/gameplay.jsonl", b'{"time": 1}'),
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                expected[str(path.relative_to(root))] = data
            namespace = dict(Path=Path, WORK=root, zipfile=zipfile,
                             INCLUDE_REPLAY_IN_EXPORT=True,
                             INITIAL_BASELINE=run, BASELINE=run, VARIATION=run,
                             CONTROL=run, HOP_STAGES=[])
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(builder), "exec"), namespace)
            output = namespace["export_results"]()
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
                self.assertEqual(set(archive.namelist()), set(expected))
                for name, data in expected.items():
                    self.assertEqual(archive.read(name), data)
