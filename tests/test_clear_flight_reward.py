"""A flight bonus must reject support, insufficient clearance, and wrong timing."""
import types
import unittest
from unittest.mock import patch

import torch

from bdxr_classroom.behaviors import clear_hop_flight


class ClearFlightReward(unittest.TestCase):
    def test_only_clear_bilateral_flight_in_requested_phase_is_paid(self):
        env = types.SimpleNamespace(
            num_envs=4, step_dt=.02,
            episode_length_buf=torch.tensor([23, 23, 23, 40]),
            scene={"feet_ground_contact": types.SimpleNamespace(
                data=types.SimpleNamespace(found=torch.tensor(
                    [[0, 0], [0, 1], [0, 0], [0, 0]])))})
        with patch("bdxr_classroom.behaviors.sampled_sole_clearance",
                   return_value=torch.tensor([[.04, .04], [.04, .04],
                                              [.04, .01], [.04, .04]])):
            torch.testing.assert_close(clear_hop_flight(env), torch.tensor([4., 0., 0., 0.]))
