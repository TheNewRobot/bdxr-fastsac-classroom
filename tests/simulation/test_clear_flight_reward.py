"""A flight bonus must reject support, insufficient clearance, and wrong timing."""
import types
import unittest
from unittest.mock import patch

import torch

from bdxr_classroom.simulation.behaviors import clear_hop_flight, level_feet_reward


class ClearFlightReward(unittest.TestCase):
    def test_optional_level_reward_does_not_pay_for_standing(self):
        env = types.SimpleNamespace(num_envs=2, scene={
            "feet_ground_contact": types.SimpleNamespace(data=types.SimpleNamespace(
                found=torch.tensor([[1, 1], [0, 0]])))})
        with patch("bdxr_classroom.simulation.behaviors.sole_tilt", return_value=torch.zeros(2, 2)):
            torch.testing.assert_close(level_feet_reward(env), torch.ones(2))
            torch.testing.assert_close(level_feet_reward(env, flight_only=True),
                                       torch.tensor([0., 1.]))

    def test_only_clear_bilateral_flight_in_requested_phase_is_paid(self):
        env = types.SimpleNamespace(
            num_envs=4, step_dt=.02,
            episode_length_buf=torch.tensor([23, 23, 23, 40]),
            scene={"feet_ground_contact": types.SimpleNamespace(
                data=types.SimpleNamespace(found=torch.tensor(
                    [[0, 0], [0, 1], [0, 0], [0, 0]])))})
        with patch("bdxr_classroom.simulation.behaviors.sampled_sole_clearance",
                   return_value=torch.tensor([[.04, .04], [.04, .04],
                                              [.04, .01], [.04, .04]])):
            torch.testing.assert_close(clear_hop_flight(env), torch.tensor([4., 0., 0., 0.]))
