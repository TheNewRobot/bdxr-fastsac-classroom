"""Optional procedural head expression, separate from learned leg control.

Only commands, mode, and time drive this presentation layer. It never changes
leg actions, actor observations, rewards, or the head actuator gains. Neutral is
the default until an explicit configuration enables validated expression.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

JOINT_NAMES = ("Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll")


@dataclass(frozen=True)
class HeadMotionConfig:
    enabled: bool = False
    walk_amplitude: float = 0.025  # radians; gentle nod around a forward gaze.
    idle_yaw: float = 0.08  # radians; occasional curious glances.
    idle_tilt: float = 0.0075  # radians.
    walk_frequency: float = 1.5  # bobs/second before the motion filter.
    idle_period: float = 12.0  # seconds for a left/right glance cycle with pauses.
    max_velocity: float = 0.40  # radians/second.
    max_acceleration: float = 1.20  # radians/second squared.
    response_frequency: float = 4.0  # radians/second; lower means slower blending.
    seed: int = 7

    def __post_init__(self):
        values = (
            self.walk_amplitude,
            self.idle_yaw,
            self.idle_tilt,
            self.walk_frequency,
            self.idle_period,
            self.max_velocity,
            self.max_acceleration,
            self.response_frequency,
        )
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Head motion parameters must be finite and nonnegative.")
        if (
            min(
                self.max_velocity,
                self.max_acceleration,
                self.response_frequency,
                self.idle_period,
            )
            <= 0
        ):
            raise ValueError(
                "Head motion rate, acceleration, and response limits must be positive."
            )


class HeadMotion:
    """Smooth four-joint targets; actual servo dynamics remain in the simulator."""

    def __init__(self, env, config=None):
        self.config = config or HeadMotionConfig()
        self.robot = env.scene["robot"]
        self.joint_ids, names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        if tuple(names) != JOINT_NAMES:
            raise ValueError(f"Unexpected head joint order: {names}")
        self.device = env.device
        self.num_envs = env.num_envs
        limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids].cpu().numpy()
        self.lower, self.upper = limits[:, 0] * 0.9, limits[:, 1] * 0.9
        self.reset()

    def reset(self, seed=None):
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.phase = rng.uniform(-math.pi, math.pi)
        self.time = 0.0
        self.position = np.zeros(4)
        self.velocity = np.zeros(4)
        self.last_acceleration = np.zeros(4)

    def desired(self, speed, yaw, mode):
        if not self.config.enabled or mode != "walk":
            # Quiet the head for braking, policy handovers, and hopping.
            return np.zeros(4)
        moving = min(1.0, max(abs(speed) / 0.10, abs(yaw) / 0.15))
        bob = self.config.walk_amplitude * math.sin(
            2 * math.pi * self.config.walk_frequency * self.time + self.phase
        )
        # Smooth occasional glances, with a pause between left and right looks.
        cycle = (self.time + (self.phase + math.pi)) % self.config.idle_period
        cycle *= 12.0 / self.config.idle_period
        glance = 0.0
        if 2 < cycle < 5:
            glance = math.sin(math.pi * (cycle - 2) / 3) ** 2
        elif 7 < cycle < 10:
            glance = -(math.sin(math.pi * (cycle - 7) / 3) ** 2)
        idle = self.config.idle_tilt * math.sin(2 * math.pi * 0.28 * self.time)
        walking = np.array([bob, -0.6 * bob, 0.0, 0.18 * bob])
        standing = np.array(
            [0.4 * idle, idle, self.config.idle_yaw * glance, 0.6 * idle]
        )
        return moving * walking + (1 - moving) * standing

    def step(self, dt, speed, yaw, mode="walk"):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Head motion timestep must be positive and finite.")
        self.time += dt
        if (
            (not self.config.enabled or mode != "walk")
            and not np.any(self.position)
            and not np.any(self.velocity)
        ):
            # Preserve the simulator's neutral path exactly when no gesture is
            # active. A handover with residual motion still settles below.
            self.last_acceleration.fill(0)
            return self.position.copy()
        target = np.clip(self.desired(speed, yaw, mode), self.lower, self.upper)
        frequency = self.config.response_frequency
        acceleration = (
            frequency**2 * (target - self.position) - 2 * frequency * self.velocity
        )
        acceleration = np.clip(
            acceleration, -self.config.max_acceleration, self.config.max_acceleration
        )
        velocity = np.clip(
            self.velocity + dt * acceleration,
            -self.config.max_velocity,
            self.config.max_velocity,
        )
        position = np.clip(self.position + dt * velocity, self.lower, self.upper)
        if (
            not np.any(target)
            and max(np.max(np.abs(position)), np.max(np.abs(velocity))) < 1e-8
        ):
            position.fill(0)
            velocity.fill(0)
        self.last_acceleration = (velocity - self.velocity) / dt
        self.velocity, self.position = velocity, position
        targets = torch.as_tensor(
            position, dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        self.robot.set_joint_position_target(targets, joint_ids=self.joint_ids)
        return position.copy()

    def metadata(self):
        return {
            **asdict(self.config),
            "joint_names": list(JOINT_NAMES),
            "type": "procedural expression; not learned locomotion",
        }
