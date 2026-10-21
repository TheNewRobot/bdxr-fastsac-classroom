"""Interrupted checkouts must be repaired without discarding local modifications."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "lesson_setup", ROOT / "scripts/setup/setup.py"
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def git(*args):
    return subprocess.check_output(
        ["git", *map(str, args)], stderr=subprocess.DEVNULL, text=True
    ).strip()


class SetupRecoveryTests(unittest.TestCase):
    def test_empty_git_directory_from_failed_clone_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            upstream.mkdir()
            git("-C", upstream, "init")
            git("-C", upstream, "config", "user.name", "Fixture")
            git("-C", upstream, "config", "user.email", "fixture@example.invalid")
            (upstream / "asset.txt").write_text("robot asset")
            git("-C", upstream, "add", ".")
            git("-C", upstream, "commit", "-m", "fixture")
            revision = git("-C", upstream, "rev-parse", "HEAD")
            dest = root / "vendor/robot"
            dest.mkdir(parents=True)
            git("-C", dest, "init")  # Interrupted before adding a remote/fetching.
            with patch.object(setup, "ROOT", root):
                setup.checkout("robot", str(upstream), revision)
                setup.checkout("robot", str(upstream), revision)
            self.assertEqual((dest / "asset.txt").read_text(), "robot asset")
            self.assertEqual(git("-C", dest, "rev-parse", "HEAD"), revision)
            (dest / "asset.txt").write_text("local modification")
            with (
                patch.object(setup, "ROOT", root),
                self.assertRaisesRegex(RuntimeError, "local changes"),
            ):
                setup.checkout("robot", str(upstream), "0" * 40)
            self.assertEqual((dest / "asset.txt").read_text(), "local modification")
