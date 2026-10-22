"""Keep the classroom comparison an actual matched reward experiment."""

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.lesson.validate_notebook import replace_assignment


class GaitRecipeTests(unittest.TestCase):
    def run_recipe(self, experiment):
        path = (
            Path(__file__).resolve().parents[2]
            / "notebooks/sections/07_gait_experiment.py"
        )
        cells = path.read_text().split("# %%")
        settings = next(c for c in cells if c.lstrip().startswith("EXPERIMENT ="))
        training = next(
            c for c in cells if c.lstrip().startswith("# Neither call has a parent")
        )
        settings = replace_assignment(settings, "EXPERIMENT", repr(experiment))
        state = {
            "ENV_CONFIG": {"yaw_bias_weight": 6.0, "command_mode": "balanced"},
            "LEARNING_CONFIG": {"seed": 7, "steps": 10000, "minutes": 15},
            "lab": SimpleNamespace(train=Mock()),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            # Execute only the checked-in teaching cells, never external input.
            exec(compile(settings + training, str(path), "exec"), state)  # noqa: S102
        return state

    def test_level_feet_changes_only_one_reward_between_fresh_runs(self):
        state = self.run_recipe("level_feet")
        control, variant = state["lab"].train.call_args_list
        common = control.args[1]
        changed = variant.args[1]
        differences = {
            key
            for key in common.keys() | changed.keys()
            if common.get(key) != changed.get(key)
        }
        self.assertEqual(differences, {"level_feet_weight"})
        self.assertEqual(common["yaw_bias_weight"], 10)
        self.assertEqual(changed["level_feet_weight"], 0.75)
        self.assertEqual(control.args[2], variant.args[2])
        self.assertEqual(control.args[2], {"seed": 7, "steps": 10000, "minutes": 25})
        self.assertFalse(control.kwargs)
        self.assertFalse(variant.kwargs)
        self.assertEqual(state["ENV_CONFIG"]["yaw_bias_weight"], 6)
        self.assertEqual(state["LEARNING_CONFIG"]["minutes"], 15)

    def test_other_styles_keep_the_original_common_settings(self):
        for name, seed, steps in (("quick_steps", 7, 10000), ("crouched", 8, 9000)):
            with self.subTest(experiment=name):
                state = self.run_recipe(name)
                control, variant = state["lab"].train.call_args_list
                self.assertEqual(control.args[1], state["ENV_CONFIG"])
                self.assertEqual(variant.args[1]["yaw_bias_weight"], 6)
                self.assertEqual(
                    control.args[2],
                    {"seed": seed, "steps": steps, "minutes": 15},
                )
