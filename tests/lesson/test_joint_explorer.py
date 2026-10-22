"""The widget must render before interaction and update all linked views."""

import base64
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from bdxr_classroom.lesson.presentation import joint_explorer


class JointExplorerTests(unittest.TestCase):
    def test_colab_uses_browser_events_without_kernel_widget_callbacks(self):
        document = (
            '<input id="angle" type="range"><img id="pose">'
            "<script>document.getElementById('angle').oninput = "
            "() => document.getElementById('pose').alt = 'changed';</script>"
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                "sys.modules", {"google.colab": types.ModuleType("google.colab")}
            ),
            patch("IPython.display.display") as display,
        ):
            path = Path(folder) / "sweep.html"
            path.write_text(document)
            result = joint_explorer(path)
            markup, javascript = display.call_args.args
            self.assertIs(result, markup)
            self.assertNotIn("<script>", markup.data)
            self.assertIn(".oninput", javascript.data)
            self.assertNotIn("kernel", javascript.data)

    def test_initial_frame_and_slider_update(self):
        data = {
            "angles": [-5, 0, 5],
            "frames": [
                base64.b64encode(value).decode()
                for value in (b"first", b"neutral", b"last")
            ],
            "positions": [[0, 0, 0], [1, 0, 1], [2, 0, 0]],
        }
        with tempfile.TemporaryDirectory() as folder, patch("IPython.display.display"):
            path = Path(folder) / "sweep.html"
            path.with_suffix(".frames.json").write_text(json.dumps(data))
            widget = joint_explorer(path)
            image, controls = widget.children
            slider, coordinates, trace = controls.children
            self.assertEqual(bytes(image.value), b"neutral")
            self.assertIn("1.000 m", coordinates.value)
            before = trace.value
            slider.value = 2
            self.assertEqual(bytes(image.value), b"last")
            self.assertIn("2.000 m", coordinates.value)
            self.assertNotEqual(before, trace.value)
            widget.close()
