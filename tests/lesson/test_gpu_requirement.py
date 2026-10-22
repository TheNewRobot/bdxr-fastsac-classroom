"""The classroom requires L4; local benchmark machines remain usable."""

import unittest
from unittest.mock import patch

from bdxr_classroom.lesson.setup import require_colab_l4


class ClassroomGpuRequirement(unittest.TestCase):
    def test_local_machine_is_not_restricted(self):
        with patch("pathlib.Path.exists", return_value=False), patch(
            "subprocess.check_output"
        ) as probe:
            require_colab_l4()
            probe.assert_not_called()

    def test_colab_accepts_l4(self):
        with patch("pathlib.Path.exists", return_value=True), patch(
            "subprocess.check_output", return_value="NVIDIA L4\n"
        ):
            require_colab_l4()

    def test_other_colab_gpus_get_actionable_error(self):
        for gpu in ("Tesla T4", "NVIDIA L40S", "NVIDIA A100-SXM4-40GB"):
            with (
                self.subTest(gpu=gpu),
                patch("pathlib.Path.exists", return_value=True),
                patch("subprocess.check_output", return_value=gpu),
                self.assertRaisesRegex(RuntimeError, "requires an L4"),
            ):
                require_colab_l4()

    def test_cpu_runtime_gets_actionable_error(self):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("subprocess.check_output", side_effect=FileNotFoundError),
            self.assertRaisesRegex(RuntimeError, "runtime type → L4"),
        ):
            require_colab_l4()
