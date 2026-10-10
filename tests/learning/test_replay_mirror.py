"""Mirror real BDX-R coordinates without mixing actor and critic information."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import unittest
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

from bdxr_classroom.simulation.environment import make_env
from bdxr_classroom.learning.fastsac import FastSAC
from bdxr_classroom.learning.mirror import build_maps, joint_map, ReplayMirror


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ReplayReflection(unittest.TestCase):
    def test_joint_signs_match_forward_kinematics(self):
        env = make_env(1)
        try:
            model = env.sim.mj_model
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            robot = env.scene["robot"]
            local_sites, _ = robot.find_sites(["left_foot", "right_foot"])
            sites = robot.indexing.site_ids[local_sites].cpu().tolist()
            neutral = data.site_xpos[sites].copy()
            for joint in ("Hip_Yaw", "Hip_Roll", "Hip_Pitch", "Knee", "Ankle"):
                displacements = []
                for foot, side, sign in [(0, "Left", 1), (1, "Right", -1)]:
                    mujoco.mj_resetData(model, data)
                    index = int(robot.indexing.joint_ids[
                        robot.joint_names.index(f"{side}_{joint}")])
                    data.qpos[model.jnt_qposadr[index]] += sign * .02
                    mujoco.mj_forward(model, data)
                    displacements.append(data.site_xpos[sites[foot]] - neutral[foot])
                np.testing.assert_allclose(
                    displacements[0] * [1, -1, 1], displacements[1], atol=.0005,
                    err_msg=joint)
        finally:
            env.close()

    def test_reflection_and_sac_update_preserve_information_boundaries(self):
        env = make_env(2, task_version="hop", command_mode="balanced")
        try:
            obs, _ = env.reset()
            maps = build_maps(env)
            mirror = ReplayMirror(maps, "cuda")
            for group, width in (("actor", 49), ("critic", 64), ("actions", 10)):
                values = torch.randn(2, width, device="cuda")
                torch.testing.assert_close(mirror.reflect(mirror.reflect(values, group), group), values)
            self.assertLess(max(maps["actor"]["indices"]), 49)
            joints = joint_map(list(env.scene["robot"].joint_names))
            neutral = env.scene["robot"].data.default_joint_pos
            torch.testing.assert_close(
                neutral[:, joints["indices"]] * neutral.new_tensor(joints["signs"]), neutral)
            cfg = dict(num_envs=2, actor_hidden=32, critic_hidden=32, buffer_size=8,
                       batch_size=4, updates=2, gamma=.97, compile=False,
                       mirror_replay=True, mirror_maps=maps)
            agent = FastSAC(49, 64, 10, cfg)
            flags = torch.tensor([False, True], device="cuda")
            agent.add(obs, torch.zeros(2, 10, device="cuda"), obs,
                      torch.ones(2, device="cuda"), flags, ~flags)
            batch = agent.buffer.sample(1)
            augmented = mirror.augment(batch)
            for field in ("rewards", "dones", "truncations"):
                torch.testing.assert_close(augmented["next", field][2:], batch["next", field])
            actor_copy = augmented["observations"].clone()
            batch["critic_observations"].add_(1000)
            torch.testing.assert_close(mirror.augment(batch)["observations"], actor_copy)
            metrics = agent.learn()
            self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
            from bdxr_classroom.evaluation.command_evaluate import load_agent
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "mirror.pt"
                agent.save(checkpoint, 1)
                restored, restored_cfg = load_agent(checkpoint, 2)
                self.assertEqual(restored_cfg["mirror_maps"], maps)
                self.assertIsNotNone(restored.replay_mirror)
                torch.testing.assert_close(
                    restored.act(obs["actor"], True), agent.act(obs["actor"], True),
                    rtol=0, atol=0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
