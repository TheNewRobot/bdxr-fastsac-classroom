"""GPU rendering selection must be verified before changing child-process settings."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bdxr_classroom.lesson.rendering import configure_rendering

NVIDIA = {"vendor": "NVIDIA Corporation", "renderer": "Tesla T4"}
SOFTWARE = {"vendor": "Mesa", "renderer": "llvmpipe"}


class RenderingSetupTests(unittest.TestCase):
    def test_existing_nvidia_backend_needs_no_override(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "bdxr_classroom.lesson.rendering.probe_renderer", return_value=NVIDIA
        ) as probe:
            result = configure_rendering(folder, "python", {"KEEP": "value"})
            self.assertNotIn("__EGL_VENDOR_LIBRARY_FILENAMES", result)
            self.assertEqual(result["KEEP"], "value")
            self.assertEqual(probe.call_count, 1)

    def test_software_backend_is_replaced_only_after_successful_probe(self):
        original = {"KEEP": "value", "__EGL_VENDOR_LIBRARY_FILENAMES": "mesa.json"}
        with tempfile.TemporaryDirectory() as folder, patch(
            "bdxr_classroom.lesson.rendering.probe_renderer",
            side_effect=[SOFTWARE, NVIDIA],
        ) as probe:
            result = configure_rendering(folder, "python", original)
            manifest = Path(result["__EGL_VENDOR_LIBRARY_FILENAMES"])
            self.assertEqual(
                json.loads(manifest.read_text())["ICD"]["library_path"],
                "libEGL_nvidia.so.0",
            )
            self.assertEqual(original["__EGL_VENDOR_LIBRARY_FILENAMES"], "mesa.json")
            self.assertEqual(probe.call_args.args[1], result)
            report = json.loads((Path(folder) / ".runtime/rendering.json").read_text())
            self.assertTrue(report["nvidia_verified"])
            self.assertEqual(report["before"], SOFTWARE)

    def test_missing_driver_preserves_working_settings_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "bdxr_classroom.lesson.rendering.probe_renderer",
            side_effect=[SOFTWARE, RuntimeError("NVIDIA library unavailable")],
        ):
            result = configure_rendering(
                folder, "python", {"__EGL_VENDOR_LIBRARY_FILENAMES": "mesa.json"}
            )
            self.assertEqual(result["__EGL_VENDOR_LIBRARY_FILENAMES"], "mesa.json")
            report = json.loads((Path(folder) / ".runtime/rendering.json").read_text())
            self.assertFalse(report["nvidia_verified"])
            self.assertIn("NVIDIA library unavailable", report["errors"])

    def test_failed_default_backend_can_be_repaired(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "bdxr_classroom.lesson.rendering.probe_renderer",
            side_effect=[RuntimeError("No EGL device"), NVIDIA],
        ):
            result = configure_rendering(folder, "python", {})
            self.assertIn("__EGL_VENDOR_LIBRARY_FILENAMES", result)

    def test_non_nvidia_candidate_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "bdxr_classroom.lesson.rendering.probe_renderer",
            side_effect=[SOFTWARE, SOFTWARE],
        ):
            result = configure_rendering(folder, "python", {})
            self.assertNotIn("__EGL_VENDOR_LIBRARY_FILENAMES", result)
            report = json.loads((Path(folder) / ".runtime/rendering.json").read_text())
            self.assertFalse(report["nvidia_verified"])
