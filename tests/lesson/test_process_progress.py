"""Long or failed commands must leave progress, diagnostics, and no live parent."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

from bdxr_classroom.lesson.process import run_logged


class ProcessTests(unittest.TestCase):
    def test_quiet_command_reports_progress_and_keeps_filtered_output(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            log = Path(directory) / "command.log"
            run_logged(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time;print('detail');time.sleep(.15);print('Result: done')",
                ],
                cwd=directory,
                log_path=log,
                heartbeat=0.03,
                show=lambda line: line.startswith("Result:"),
                timeout=3,
            )
            self.assertIn("still running", output.getvalue())
            self.assertIn("Result: done", output.getvalue())
            self.assertNotIn("detail", output.getvalue())
            self.assertIn("detail", log.read_text())

    def test_timeout_and_error_include_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "command.log"
            with self.assertRaisesRegex(TimeoutError, "exceeded"):
                run_logged(
                    [sys.executable, "-u", "-c", "import time;time.sleep(20)"],
                    cwd=directory,
                    log_path=log,
                    timeout=0.15,
                    heartbeat=0.03,
                )
            with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                run_logged(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        "raise ValueError('deliberate failure')",
                    ],
                    cwd=directory,
                    log_path=log,
                    timeout=3,
                )
