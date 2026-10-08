"""Checks for terminal targets and checkpoint continuation (requires CUDA)."""

import tempfile
import unittest
from pathlib import Path

import torch

from bdxr_classroom.fastsac import FastSAC


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LearningContract(unittest.TestCase):
    def make_agent(self, amp="off"):
        return FastSAC(3, 4, 2, dict(
            num_envs=3, actor_hidden=32, critic_hidden=32, buffer_size=8,
            batch_size=6, updates=2, gamma=0.97, compile=False, amp=amp,
        ))

    def test_terminal_targets_and_resume(self):
        agent = self.make_agent()
        obs = {"actor": torch.zeros(3, 3, device="cuda"),
               "critic": torch.zeros(3, 4, device="cuda")}
        final = {"actor": torch.full((3, 3), 7.0, device="cuda"),
                 "critic": torch.full((3, 4), 8.0, device="cuda")}
        # Ordinary timeout, fall, simultaneous timeout and fall.
        agent.add(obs, torch.zeros(3, 2, device="cuda"), final,
                  torch.ones(3, device="cuda"),
                  torch.tensor([False, True, True], device="cuda"),
                  torch.tensor([True, False, True], device="cuda"))
        self.assertEqual(agent.buffer.truncations[:, 0].tolist(), [1, 0, 0])
        self.assertTrue(torch.equal(agent.buffer.next_observations[:, 0], final["actor"]))
        agent.learn()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resume.pt"
            agent.save(path, 123, include_replay=True)
            restored = self.make_agent()
            self.assertEqual(restored.load(path), 123)
            self.assertEqual(restored.buffer.ptr, 1)
            self.assertEqual(restored.update_count, agent.update_count)
            self.assertTrue(torch.equal(restored.buffer.observations, agent.buffer.observations))
            with torch.no_grad():
                self.assertTrue(torch.equal(restored.act(obs["actor"], True),
                                            agent.act(obs["actor"], True)))
            self.assertTrue(restored.actor_opt.state)

    def test_fp16_scaler_continuation(self):
        agent = self.make_agent("fp16")
        obs = {"actor": torch.randn(3, 3, device="cuda"),
               "critic": torch.randn(3, 4, device="cuda")}
        flags = torch.zeros(3, dtype=torch.bool, device="cuda")
        agent.add(obs, torch.zeros(3, 2, device="cuda"), obs,
                  torch.ones(3, device="cuda"), flags, flags)
        for _ in range(3):
            metrics = agent.learn()
            self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fp16.pt"
            agent.save(path, 3, include_replay=True)
            restored = self.make_agent("fp16")
            restored.load(path)
            self.assertEqual(restored.scaler.state_dict(), agent.scaler.state_dict())

    def test_finetune_preserves_new_random_stream_but_resume_restores_parent(self):
        agent = self.make_agent()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "parent.pt"
            torch.manual_seed(10)
            parent_cpu = torch.get_rng_state()
            parent_cuda = torch.cuda.get_rng_state()
            agent.save(path, 7)
            child = self.make_agent()
            torch.manual_seed(20)
            child_cpu = torch.get_rng_state()
            child_cuda = torch.cuda.get_rng_state()
            child.load(path, replay=False, restore_rng=False)
            self.assertTrue(torch.equal(torch.get_rng_state(), child_cpu))
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(), child_cuda))
            child.load(path)
            self.assertTrue(torch.equal(torch.get_rng_state(), parent_cpu))
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(), parent_cuda))


if __name__ == "__main__":
    unittest.main()
