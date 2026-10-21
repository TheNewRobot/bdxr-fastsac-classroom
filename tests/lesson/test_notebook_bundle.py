"""The notebook downloads a verified release without a browser-message dependency."""

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import threading
import time
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "lesson_download", ROOT / "scripts/setup/lesson_download.py"
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def payload(extra=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scripts/setup/setup.py", "# setup fixture\n")
        archive.writestr("configs/example.json", "{}")
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


@contextlib.contextmanager
def server(responses):
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_GET(self):
            body = responses[min(type(self).calls, len(responses) - 1)]
            type(self).calls += 1
            if body is None:
                time.sleep(0.4)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{service.server_port}/lesson.zip", Handler
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=2)


class NotebookBundleTests(unittest.TestCase):
    def test_release_notebook_has_no_browser_transfer_or_encoded_payload(self):
        notebook = nbformat.read(ROOT / "notebooks/BDXR_Walking_101.ipynb", 4)
        self.assertEqual(notebook.metadata.bdxr_lesson.version, 2)
        self.assertNotIn("archive", notebook.metadata.bdxr_lesson)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertLess(max(map(len, cell.source.splitlines()), default=0), 160)
                self.assertNotIn("blocking_request", cell.source)
                self.assertNotIn("SOURCE_BUNDLE", cell.source)
                self.assertNotIn("files.upload", cell.source)
        release = notebook.metadata.bdxr_lesson
        path = (
            ROOT / "artifacts/lesson_releases" / release.tag / "BDXR_lesson_files.zip"
        )
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), release.sha256)
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes(), name)

    def test_download_retries_corruption_and_preserves_student_results_on_rerun(self):
        content = payload()
        digest = hashlib.sha256(content).hexdigest()
        with (
            tempfile.TemporaryDirectory() as directory,
            server([b"broken", content]) as (url, handler),
        ):
            work = Path(directory) / "lesson"
            bootstrap.install_lesson(url, digest, work, timeout=2)
            self.assertEqual(handler.calls, 2)
            student = work / "experiments/my_result.txt"
            student.parent.mkdir()
            student.write_text("keep")
            bootstrap.install_lesson(url, digest, work, timeout=2)
            self.assertEqual(handler.calls, 2)
            self.assertEqual(student.read_text(), "keep")
            # An interrupted extraction is repaired without deleting student work.
            (work / "scripts/setup/setup.py").unlink()
            bootstrap.install_lesson(url, digest, work, timeout=2)
            self.assertTrue((work / "scripts/setup/setup.py").is_file())
            self.assertEqual(student.read_text(), "keep")

    def test_corrupt_and_unsafe_archives_never_mark_installation_ready(self):
        for content, digest, message in [
            (b"broken", "0" * 64, "Could not download"),
            (payload({"../escape.py": "bad"}), None, "invalid"),
        ]:
            digest = digest or hashlib.sha256(content).hexdigest()
            with (
                tempfile.TemporaryDirectory() as directory,
                server([content]) as (url, _),
            ):
                work = Path(directory) / "lesson"
                with self.assertRaisesRegex(RuntimeError, message):
                    bootstrap.install_lesson(url, digest, work, attempts=1)
                self.assertFalse((work / ".lesson_release.json").exists())
                self.assertFalse((Path(directory) / "escape.py").exists())

    def test_different_release_never_overwrites_existing_experiment_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / ".lesson_release.json").write_text(json.dumps({"sha256": "old"}))
            with self.assertRaisesRegex(RuntimeError, "different lesson release"):
                bootstrap.install_lesson("https://unused.invalid", "new", work)

    def test_unresponsive_server_fails_promptly_without_a_ready_marker(self):
        with tempfile.TemporaryDirectory() as directory, server([None]) as (url, _):
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "Could not download"):
                bootstrap.install_lesson(
                    url, "0" * 64, directory, timeout=0.05, attempts=1
                )
            self.assertLess(time.monotonic() - started, 1)
            self.assertFalse((Path(directory) / ".lesson_release.json").exists())
