"""Physical expression must preserve leg inputs and agree with model gravity."""

import unittest

import mujoco
import numpy as np
import torch

from bdxr_classroom.simulation.environment import make_env
from bdxr_classroom.simulation.head_motion import HeadMotionConfig, make_head_motion


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PhysicalHeadTests(unittest.TestCase):
    def test_render_uses_the_physical_joint_pose(self):
        env = make_env(1, render=True)
        try:
            env.reset(seed=7)
            head = make_head_motion(
                env, HeadMotionConfig(enabled=True, physical_expression=True)
            )
            for _ in range(3):
                head.step(env.step_dt, .3, 0)
                env.step(torch.zeros(1, 10, device=env.device))
            before = env.sim.data.qpos.clone()
            head.render()
            torch.testing.assert_close(env.sim.data.qpos.clone(), before, rtol=0, atol=0)
            np.testing.assert_allclose(
                env._offline_renderer._data.qpos,
                before[0].cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
            )
        finally:
            env.close()

    def test_parallel_reference_and_gravity(self):
        env = make_env(3)
        try:
            env.reset(seed=7)
            head = make_head_motion(
                env, HeadMotionConfig(enabled=True, physical_expression=True)
            )
            before = env.sim.data.qpos.clone()
            target = head.desired(.3, 0, "walk")
            self.assertEqual(target.shape, (3, 4))
            self.assertTrue(torch.equal(before, env.sim.data.qpos))
            # Compare the collision-free gravity pipeline with the complete
            # MuJoCo forward pipeline at the exact same scratch configuration.
            gravity = head.scratch.qfrc_bias.copy()
            mujoco.mj_forward(head.model, head.scratch)
            np.testing.assert_allclose(
                gravity, head.scratch.qfrc_bias, rtol=1e-9, atol=1e-9
            )
            # Global position and absolute heading must not leak through the
            # procedural reference into the actor's head channels.
            env.sim.data.qpos[:] = before[0].expand_as(before)
            env.sim.data.qpos[1, :3] += torch.tensor(
                [4.0, -2.0, 1.5], device=env.device
            )
            env.sim.data.qpos[2, :3] += torch.tensor(
                [-3.0, 5.0, -.7], device=env.device
            )
            quaternion = before[0, 3:7].cpu().numpy()
            for row, angle in ((1, 1.3), (2, -2.1)):
                c, s = np.cos(angle / 2), np.sin(angle / 2)
                w, x, y, z = quaternion
                rotated = [c*w - s*z, c*x - s*y, c*y + s*x, c*z + s*w]
                env.sim.data.qpos[row, 3:7] = torch.tensor(
                    rotated, dtype=before.dtype, device=env.device
                )
            references = head.desired(.3, 0, "walk")
            np.testing.assert_allclose(references[1:], references[:1].repeat(2, 0),
                                       rtol=1e-5, atol=1e-6)
            env.sim.data.qpos[:] = before
            head.position[:] = [.1, .2, -.1, .05]
            head.velocity[:] = [.2, -.1, .3, .1]
            for width in (47, 49):
                raw = torch.randn(3, width, device=env.device)
                original = raw.clone()
                result = head.policy_observations(raw)
                expected = raw.clone()
                expected[:, head.position_columns] -= torch.tensor(
                    head.position, dtype=raw.dtype, device=env.device
                )
                expected[:, head.velocity_columns] -= torch.tensor(
                    head.velocity, dtype=raw.dtype, device=env.device
                )
                torch.testing.assert_close(result, expected)
                torch.testing.assert_close(raw, original)
            head.reset(7)
            head.step(env.step_dt, .3, 0)
            self.assertEqual(head.position.shape, (3, 4))
            self.assertTrue(np.isfinite(head.position).all())
        finally:
            env.close()
