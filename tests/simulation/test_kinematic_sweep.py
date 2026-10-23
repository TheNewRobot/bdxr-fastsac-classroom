"""The slider must move and measure the camera-facing leg, without stepping physics."""

import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from bdxr_classroom.lesson.kinematics import lesson_font, model_state, sweep_joint
from bdxr_classroom.simulation.environment import make_env


class KinematicSweepTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_right_knee_moves_right_foot_and_preserves_simulated_state(self):
        env = make_env(num_envs=1, seed=7)
        try:
            env.reset(seed=7)
            before = env.sim.data.qpos.clone()
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "sweep.html"
                positions = sweep_joint(env, "Right_Knee", [-40, 0, 20], output)
                record = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(record["foot_site"], "right_foot")
                self.assertIn("Right foot in torso frame", output.read_text())
                model, data = model_state(env)
                knee = model.joint("robot/Right_Knee").id
                right = model.site("robot/right_foot").id
                left = model.site("robot/left_foot").id
                root = next(
                    i
                    for i in range(model.njnt)
                    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
                )
                torso = int(model.jnt_bodyid[root])
                left_positions = []
                for angle, recorded in zip([-40, 0, 20], positions):
                    data.qpos[model.jnt_qposadr[knee]] = np.deg2rad(angle)
                    mujoco.mj_forward(model, data)
                    expected = data.xmat[torso].reshape(3, 3).T @ (
                        data.site_xpos[right] - data.xpos[torso]
                    )
                    np.testing.assert_allclose(expected, recorded, atol=1e-10)
                    left_positions.append(data.site_xpos[left].copy())
                self.assertLess(np.max(np.ptp(left_positions, axis=0)), 1e-10)
                self.assertGreater(np.max(np.ptp(positions, axis=0)), 0.03)
            np.testing.assert_array_equal(
                before.cpu().numpy(), env.sim.data.qpos.cpu().numpy()
            )
        finally:
            env.close()

    def test_fonts_are_loaded_from_installed_package_files(self):
        for size in (15, 17, 19):
            for bold in (False, True):
                font = lesson_font(size, bold=bold)
                self.assertTrue(Path(font.path).is_absolute())
                self.assertTrue(Path(font.path).is_file())
