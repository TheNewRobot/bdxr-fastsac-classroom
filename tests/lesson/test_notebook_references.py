"""Teaching examples must live in Markdown and survive clearing all outputs."""

import ast
import hashlib
import importlib.util
import re
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


class NotebookReferences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(
            ROOT / "notebooks/BDXR_Walking_101.ipynb", as_version=4
        )

    def test_no_examples_or_saved_results_in_code_cells(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [], cell.id)
                self.assertNotIn("instructor_reference", cell.metadata)
                self.assertNotIn("reference_comparison(", cell.source)
                self.assertNotIn("recorded_details(", cell.source)

    def test_media_activities_import_their_display_helpers(self):
        for cell in self.notebook.cells:
            if cell.cell_type != "code" or cell.source.startswith("%%"):
                continue
            tree = ast.parse(cell.source)
            used = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in {"Video", "Image"}
            }
            imports = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "IPython.display"
                for alias in node.names
            }
            self.assertTrue(
                used <= imports,
                f"Missing display imports in {cell.id}: {used - imports}",
            )

    def test_opening_photo_and_walking_example_belong_to_the_intro_text(self):
        opening = self.notebook.cells[0]
        self.assertEqual(opening.cell_type, "markdown")
        self.assertIn("Meet the droid", opening.source)
        self.assertIn("/notebook/reference-disney_bdx.png", opening.source)
        self.assertIn("/inline/walking.gif", opening.source)
        self.assertIn("D23 / Disney", opening.source)
        self.assertFalse(opening.get("attachments"))

    def test_all_markdown_media_have_public_urls_and_keep_their_sizes(self):
        animations = 0
        images = 0
        prefix = "/TheNewRobot/bdxr-fastsac-classroom/main/"
        for cell in self.notebook.cells:
            if cell.cell_type != "markdown":
                continue
            self.assertNotIn("lesson-asset:", cell.source)
            self.assertNotIn("attachment:", cell.source)
            for url in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", cell.source):
                parsed = urlparse(url)
                self.assertEqual(parsed.scheme, "https")
                self.assertEqual(parsed.netloc, "raw.githubusercontent.com")
                self.assertTrue(parsed.path.startswith(prefix))
                path = ROOT / parsed.path.removeprefix(prefix)
                self.assertTrue(path.is_file(), path)
                expected = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
                self.assertEqual(parse_qs(parsed.query)["v"], [expected])
                image = Image.open(path)
                large = path.name in {
                    "reference-robot.png",
                    "reference-annotated_robot.png",
                    "reference-kinematic_tree.png",
                    "course_preview.png",
                }
                self.assertLessEqual(image.width, 800 if large else 720)
                if path.suffix == ".gif":
                    self.assertGreater(image.n_frames, 1)
                    animations += 1
                images += 1
        self.assertEqual(animations, 8)
        self.assertEqual(images, 22)

    def test_builder_never_executes_code(self):
        spec = importlib.util.spec_from_file_location(
            "lesson_builder", ROOT / "scripts/lesson/build_notebook.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        activity = nbformat.v4.new_code_cell(
            'raise AssertionError("No code may execute while building Markdown")'
        )
        reference = nbformat.v4.new_markdown_cell(
            "Explain the robot.\n\n![Robot](lesson-asset:reference/robot.png)"
        )
        builder.embed_references([activity, reference])
        self.assertEqual(activity.outputs, [])
        self.assertIn("/notebook/reference-robot.png", reference.source)
        self.assertFalse(reference.get("attachments"))


if __name__ == "__main__":
    unittest.main()
