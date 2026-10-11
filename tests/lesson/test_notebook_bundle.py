"""The single-file lesson carries data in metadata, not in Python source."""

import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[2]


class NotebookBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(ROOT / "notebooks/BDXR_Walking_101.ipynb", 4)

    def loader(self):
        source = next(
            c.source for c in self.notebook.cells
            if c.cell_type == "code" and c.source.startswith("# @title Unpack")
        )
        function = next(
            n for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "read_lesson_archive"
        )
        namespace = dict(
            Path=Path, os=os, json=json, base64=base64, hashlib=hashlib,
            BUNDLE_SHA256=self.notebook.metadata.bdxr_lesson.sha256,
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<loader>", "exec"),
             namespace)
        return namespace["read_lesson_archive"]

    def test_no_encoded_payload_in_code_and_bundle_matches_repo(self):
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                self.assertLess(max(map(len, cell.source.splitlines()), default=0), 160)
                self.assertNotIn("SOURCE_BUNDLE =", cell.source)
        payload = base64.b64decode(self.notebook.metadata.bdxr_lesson.archive)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIsNone(archive.testzip())
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes(), name)

    def test_local_renamed_notebook_without_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "my_class.ipynb"
            nbformat.write(self.notebook, path)
            with patch("pathlib.Path.cwd", return_value=Path(directory)), patch.dict(
                "sys.modules", {"google.colab": None}
            ), patch.dict(os.environ, {"JPY_SESSION_NAME": ""}):
                payload = self.loader()()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            self.notebook.metadata.bdxr_lesson.sha256,
        )

    def test_colab_frontend_reply_and_integrity_failure(self):
        document = json.loads(nbformat.writes(self.notebook))
        bridge = types.SimpleNamespace(
            blocking_request=lambda request, timeout_sec: {"ipynb": document}
        )
        module = types.ModuleType("google.colab")
        module._message = bridge
        with patch.dict("sys.modules", {"google.colab": module}):
            self.assertTrue(self.loader()())
            document["metadata"]["bdxr_lesson"]["archive"] = base64.b64encode(
                b"damaged archive"
            ).decode()
            with self.assertRaisesRegex(RuntimeError, "damaged"):
                self.loader()()
