"""A swing reward must use the completed step and clear state on reset."""
import types
import unittest
from unittest.mock import patch

import torch

from bdxr_classroom.behaviors import sole_swing_height_cost


class SoleSwingReward(unittest.TestCase):
    def test_peaks_are_charged_at_landing_and_reset_per_environment(self):
        found = torch.zeros(2, 2, dtype=torch.bool)
        landing = torch.zeros_like(found)
        sensor = types.SimpleNamespace(data=types.SimpleNamespace(found=found),
                                       compute_first_contact=lambda dt: landing)
        env = types.SimpleNamespace(num_envs=2, step_dt=.02,
                                    episode_length_buf=torch.tensor([10, 10]),
                                    scene={"feet_ground_contact": sensor},
                                    command_manager=types.SimpleNamespace(
                                        get_command=lambda _: torch.tensor([[.3, 0, 0], [.3, 0, 0]])))
        with patch("bdxr_classroom.behaviors.sampled_sole_clearance",
                   return_value=torch.tensor([[.06, .06], [.03, .03]])):
            self.assertTrue((sole_swing_height_cost(env) == 0).all())
        found.fill_(True)
        landing.fill_(True)
        with patch("bdxr_classroom.behaviors.sampled_sole_clearance",
                   return_value=torch.zeros(2, 2)):
            result = sole_swing_height_cost(env)
            self.assertAlmostEqual(float(result[0]), 0)
            self.assertAlmostEqual(float(result[1]), .5)
            self.assertTrue((env._sole_swing_peak == 0).all())
            env._sole_swing_peak.fill_(.06)
            env.episode_length_buf[0] = 0
            result = sole_swing_height_cost(env)
            self.assertAlmostEqual(float(result[0]), 2)
            self.assertAlmostEqual(float(result[1]), 0)
