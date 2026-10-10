"""The bilateral objective must distinguish jumping from raising one foot."""

import types
import unittest
from unittest.mock import patch
import torch

from bdxr_classroom.simulation.behaviors import hop_height_tracking


class BilateralHopReward(unittest.TestCase):
    def test_one_foot_cannot_substitute_for_two_clear_feet(self):
        robot = types.SimpleNamespace(
            find_sites=lambda _: ([0, 1], ["left", "right"]),
            data=types.SimpleNamespace(site_pos_w=torch.zeros(3, 2, 3)))
        env = types.SimpleNamespace(
            episode_length_buf=torch.full((3,), .475),
            step_dt=1, scene={"robot": robot})
        clearances = torch.tensor([[.04, .04], [.04, 0], [0, 0]])
        with patch("bdxr_classroom.simulation.behaviors.sampled_sole_clearance", return_value=clearances):
            average = hop_height_tracking(env, flight_only=True, clearance_mode="sole")
            bilateral = hop_height_tracking(
                env, flight_only=True, clearance_mode="sole", bilateral=True)
        self.assertAlmostEqual(float(bilateral[0]), 4.0, places=5)
        self.assertEqual(float(bilateral[0]), float(average[0]))
        self.assertLess(float(bilateral[1]), float(average[1]))
        self.assertAlmostEqual(float(bilateral[1]), float(bilateral[2]), places=5)
