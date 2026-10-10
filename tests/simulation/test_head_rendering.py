"""Cosmetic articulation must not write back into physics or policy inputs."""

import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from bdxr_classroom.simulation.environment import make_env
from bdxr_classroom.simulation.head_motion import HeadMotion, HeadMotionConfig


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class HeadRenderingTests(unittest.TestCase):
    def test_level_gaze_preserves_simulation_state(self):
        env = make_env(1, render=True)
        try:
            env.reset(seed=7)
            head = HeadMotion(
                env, HeadMotionConfig(enabled=True, level_gaze_in_render=True)
            )
            for _ in range(3):
                head.step(env.step_dt, 0.3, 0)
                observations, *_ = env.step(torch.zeros(1, 10, device=env.device))
                before = {
                    name: getattr(env.sim.data, name).clone()
                    for name in ("qpos", "qvel", "ctrl")
                }
                actor = observations["actor"].clone()
                image = head.render()
                for name, value in before.items():
                    self.assertTrue(torch.equal(value, getattr(env.sim.data, name)))
                self.assertTrue(torch.equal(actor, observations["actor"]))
                self.assertEqual(image.shape, (640, 960, 3))
                self.assertLess(abs(head.last_render["gaze_elevation_deg"]), 1)
                self.assertEqual(head.last_render["scope"], "render-only")
        finally:
            env.close()
