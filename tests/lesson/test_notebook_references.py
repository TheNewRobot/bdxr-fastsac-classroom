"""Instructor examples must be visible without fabricating student execution."""

import base64
import importlib.util
import io
from pathlib import Path
import re
import unittest

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


class NotebookReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(
            ROOT / "notebooks/BDXR_Walking_101.ipynb", as_version=4
        )

    def test_only_instructor_cells_have_saved_outputs(self):
        references = []
        for cell in self.notebook.cells:
            if cell.cell_type != "code":
                continue
            self.assertIsNone(cell.execution_count)
            if cell.metadata.get("instructor_reference"):
                references.append(cell)
                self.assertTrue(cell.outputs)
                self.assertTrue(
                    all(output.output_type == "display_data" for output in cell.outputs)
                )
            else:
                self.assertEqual(cell.outputs, [], cell.id)
        self.assertEqual(len(references), 6)  # Intro, three gaits, hopping, game.

    def test_markdown_assets_are_self_contained(self):
        count = 0
        for cell in self.notebook.cells:
            if cell.cell_type != "markdown":
                continue
            self.assertNotIn("lesson-asset:", cell.source)
            for name in re.findall(r"\(attachment:([^)]+)\)", cell.source):
                bundle = cell.attachments[name]
                mime, data = next(iter(bundle.items()))
                self.assertIn(mime, ("image/png", "image/svg+xml"))
                raw = base64.b64decode(data)
                self.assertTrue(raw)
                if mime == "image/png":
                    image = Image.open(io.BytesIO(raw))
                    large_figure = name in {
                        "reference-robot.png",
                        "reference-annotated_robot.png",
                        "reference-kinematic_tree.png",
                        "course_preview.png",
                    }
                    self.assertLessEqual(image.width, 800 if large_figure else 720)
                count += 1
        self.assertGreaterEqual(count, 12)

    def test_student_training_is_not_executed_when_rendering_references(self):
        spec = importlib.util.spec_from_file_location(
            "lesson_builder", ROOT / "scripts/lesson/build_notebook.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        activity = nbformat.v4.new_code_cell(
            'raise AssertionError("Student activity was executed during the build")'
        )
        reference = nbformat.v4.new_code_cell(
            "from IPython.display import HTML, display\n"
            "display(HTML('<p>Instructor example</p>'))",
            metadata={"instructor_reference": True},
        )
        builder.embed_references([activity, reference])
        self.assertEqual(activity.outputs, [])
        self.assertEqual(
            reference.outputs[0].data["text/html"], "<p>Instructor example</p>"
        )


if __name__ == "__main__":
    unittest.main()
