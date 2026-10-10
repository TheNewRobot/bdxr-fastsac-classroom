"""Explicit coverage of straight walking, turning, stopping, and combined motion."""

from dataclasses import dataclass
import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class BalancedVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        category = torch.rand(len(env_ids), device=self.device)
        # 10% standing; 40% straight at signed speed; 20% turning; 30% combined.
        self.is_standing_env[env_ids] = category < .1
        straight = env_ids[(category >= .1) & (category < .5)]
        turning = env_ids[(category >= .5) & (category < .7)]
        self.vel_command_b[straight, 2] = 0
        self.vel_command_b[turning, :2] = 0


@dataclass(kw_only=True)
class BalancedVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return BalancedVelocityCommand(self, env)
