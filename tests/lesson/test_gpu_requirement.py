"""An available Colab GPU must not be rejected just because it is a T4."""

import unittest
from unittest.mock import patch

from bdxr_classroom.lesson.setup import require_colab_gpu


class ClassroomGpuRequirement(unittest.TestCase):
    def test_local_machine_is_not_restricted(self):
        with patch("pathlib.Path.exists", return_value=False), patch(
            "subprocess.check_output"
        ) as probe:
            require_colab_gpu()
            probe.assert_not_called()

    def test_colab_accepts_l4(self):
        with patch("pathlib.Path.exists", return_value=True), patch(
            "subprocess.check_output", return_value="NVIDIA L4\n"
        ):
            require_colab_gpu()

    def test_other_colab_gpus_are_not_artificially_blocked(self):
        for gpu in ("Tesla T4", "NVIDIA L40S", "NVIDIA A100-SXM4-40GB"):
            with (
                self.subTest(gpu=gpu),
                patch("pathlib.Path.exists", return_value=True),
                patch("subprocess.check_output", return_value=gpu),
            ):
                require_colab_gpu()

    def test_empty_gpu_list_gets_actionable_error(self):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("subprocess.check_output", return_value=""),
            self.assertRaisesRegex(RuntimeError, "needs a GPU runtime"),
        ):
            require_colab_gpu()

    def test_cpu_runtime_gets_actionable_error(self):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("subprocess.check_output", side_effect=FileNotFoundError),
            self.assertRaisesRegex(RuntimeError, "runtime type → GPU"),
        ):
            require_colab_gpu()
