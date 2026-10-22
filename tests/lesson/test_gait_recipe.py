"""The student's first run is the control; only one new reward run is launched."""

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class GaitRecipeTests(unittest.TestCase):
    def run_recipe(self, completed_steps):
        path = Path(__file__).resolve().parents[2] / "notebooks/sections/06_reward.py"
        cells = path.read_text().split("# %%")
        settings = next(c for c in cells if c.lstrip().startswith("PREDICTION ="))
        training = next(
            c for c in cells if c.lstrip().startswith("# No parent checkpoint")
        )
        environment = {"yaw_bias_weight": 10.0, "command_mode": "balanced"}
        learning = {"seed": 7, "steps": 12000, "minutes": 15}
        lab = SimpleNamespace(train=Mock(), record=Mock())
        state = {
            "ENV_CONFIG": environment,
            "LEARNING_CONFIG": learning,
            "lab": lab,
            "BASELINE": Path("/saved_baseline"),
            "read_json": lambda _: {"step": completed_steps},
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(settings + training, str(path), "exec"), state)  # noqa: S102 -- checked-in lesson cells
        lab.train.assert_called_once_with(
            "level_feet",
            {**environment, "level_feet_weight": 0.75},
            {**learning, "steps": completed_steps},
        )
        self.assertNotIn("level_feet_weight", environment)
        self.assertEqual(learning["steps"], 12000)
        self.assertNotIn("CONTROL = lab.train", path.read_text())

    def test_one_scalar_change_and_fresh_training(self):
        self.run_recipe(12000)

    def test_time_limited_baseline_gets_matching_variant_steps(self):
        self.run_recipe(8000)
