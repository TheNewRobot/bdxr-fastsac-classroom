"""Check collision-plane tilt independently of arbitrary site frames."""
import math
import types
import unittest

import torch

from bdxr_classroom.simulation.behaviors import sole_tilt, level_feet_reward


class SoleTilt(unittest.TestCase):
    def test_horizontal_and_pitched_capsules(self):
        # Capsule local Z follows the long sole axis. Rotate it onto world X,
        # then pitch that axis 30 degrees toward world Z.
        angles = torch.tensor([math.pi / 2, math.pi / 3])
        quats = torch.zeros(2, 4, 4)
        quats[:, :, 0] = torch.cos(angles[:, None] / 2)
        quats[:, :, 2] = torch.sin(angles[:, None] / 2)
        positions = torch.zeros(2, 4, 3)
        positions[:, 1::2, 1] = .024
        robot = types.SimpleNamespace(data=types.SimpleNamespace(
            geom_pos_w=positions, geom_quat_w=quats))
        env = types.SimpleNamespace(num_envs=2, scene={"robot": robot},
                                    _sole_tilt_ids=[0, 1, 2, 3])
        tilt = sole_tilt(env)
        self.assertTrue(torch.allclose(tilt[0], torch.zeros(2), atol=1e-4))
        self.assertTrue(torch.allclose(tilt[1], torch.full((2,), math.pi / 6), atol=1e-5))
        self.assertGreater(float(level_feet_reward(env)[0]),
                           float(level_feet_reward(env)[1]))
