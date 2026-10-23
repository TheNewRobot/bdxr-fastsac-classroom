"""Real-simulator reset/finish tests using explicit test-only state injection."""

import os
import json
import tempfile
import unittest
from pathlib import Path
import mujoco
import torch
from bdxr_classroom.game.session import GameSession


@unittest.skipUnless(
    os.environ.get("BDXR_TEST_WALK"), "Set BDXR_TEST_WALK to a compatible checkpoint"
)
class SimulatorGameRules(unittest.TestCase):
    def test_gate_contact_stays_finite_with_every_link_collision_enabled(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/gate_contact_state.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            game = GameSession(
                os.environ["BDXR_TEST_WALK"], render=False, record=False,
                course=fixture["course"],
                run_root=directory, operator="captured_gate_contact_regression",
            )
            try:
                self.assertTrue(
                    game.env.sim.mj_model.opt.disableflags
                    & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
                )
                for name in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
                    getattr(game.env.sim.data, name)[:] = torch.tensor(
                        fixture[name], device=game.env.device
                    )
                game.env.sim.step()
                self.assertTrue(torch.isfinite(game.env.sim.data.qacc.clone()).all())
                self.assertTrue(torch.isfinite(game.env.sim.data.qpos.clone()).all())
                self.assertGreater(int(game.env.sim.wp_data.nacon.numpy()[0]), 0)
            finally:
                game.close()

    def test_nonfinite_state_resets_and_pauses_without_a_false_success(self):
        with tempfile.TemporaryDirectory() as directory:
            game = GameSession(
                os.environ["BDXR_TEST_WALK"], render=False, record=False,
                run_root=directory, operator="injected_nonfinite_state_test",
            )
            try:
                game.command(.3, 0)
                game.env.sim.data.qpos[0, 0] = float("nan")
                status = game.step()
                self.assertEqual(status["falls"], 1)
                self.assertFalse(status["finished"])
                self.assertTrue(status["paused"])
                self.assertTrue(torch.isfinite(game.env.sim.data.qpos.clone()).all())
                self.assertIn("Simulation error", status["message"])
                self.assertEqual(
                    len(list((game.output / "evaluation").glob("physics_failure_*.npz"))), 1
                )
            finally:
                game.close()

    def test_physical_fall_reset_and_supported_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            game = GameSession(
                os.environ["BDXR_TEST_WALK"],
                render=False,
                record=False,
                run_root=directory,
                operator="injected_state_contract_test",
            )
            try:
                game.command(0, 0)
                for _ in range(20):
                    game.step()
                elapsed = game.score.elapsed
                robot = game.env.scene["robot"]
                # Tip the physical body to force the environment's fall termination.
                state = torch.tensor(
                    [[0, 0, 0.10, 0.7071068, 0.7071068, 0, 0, 0, 0, 0, 0, 0, 0]],
                    device="cuda",
                )
                robot.write_root_state_to_sim(state)
                for _ in range(30):
                    game.step()
                    if game.score.falls:
                        break
                self.assertEqual(game.score.falls, 1)
                self.assertGreater(game.score.elapsed, elapsed)
                self.assertTrue(abs(float(robot.data.root_link_pos_w[0, 0])) < 0.1)
                self.assertFalse(game.score.finished)
                # A shortcut to the platform must not earn a course completion.
                state = torch.tensor(
                    [[12, 0, 0.455, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device="cuda"
                )
                robot.write_root_state_to_sim(state)
                for _ in range(30):
                    game.step()
                self.assertFalse(game.score.finished)
                # Explicit scoring-test injection, not traversal evidence.
                game.progress.completed = len(game.progress.checkpoints)
                for _ in range(200):
                    game.step()
                    if game.score.finished:
                        break
                self.assertTrue(game.score.finished)
                self.assertEqual(game.score.falls, 1)
                finished_time = game.score.elapsed
                game.step()
                self.assertEqual(game.score.elapsed, finished_time)
                game.new_game()
                self.assertEqual(game.score.elapsed, 0)
                self.assertEqual(game.score.falls, 0)
                self.assertFalse(game.score.finished)
            finally:
                game.close()


if __name__ == "__main__":
    unittest.main()
