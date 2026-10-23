"""Optional procedural head expression, separate from learned leg control.

Commands, mode, and time drive the physical gestures. Leg actions, observation
definitions, reward settings, and actuator gains remain unchanged. Optional
level gaze affects only the rendered state, not the simulated articulation.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

JOINT_NAMES = ("Neck_Pitch", "Head_Pitch", "Head_Yaw", "Head_Roll")


@dataclass(frozen=True)
class HeadMotionConfig:
    enabled: bool = False
    physical_expression: bool = False
    level_gaze_in_render: bool = False  # cosmetic only; never changes physics.
    walk_amplitude: float = 0.025  # radians; small nod around nominal targets.
    idle_yaw: float = 0.08  # radians; occasional curious glances.
    idle_tilt: float = 0.0075  # radians.
    walk_frequency: float = 1.5  # bobs/second before the motion filter.
    idle_period: float = 12.0  # seconds for a left/right glance cycle with pauses.
    max_velocity: float = 0.40  # radians/second.
    max_acceleration: float = 1.20  # radians/second squared.
    response_frequency: float = 4.0  # radians/second; lower means slower blending.
    seed: int = 7
    display_walk_bob: float = 0.16  # radians; cosmetic neck motion only.
    display_idle_yaw: float = 0.25  # radians; gentle idle looks.

    def __post_init__(self):
        if self.physical_expression and self.level_gaze_in_render:
            raise ValueError("Physical head expression cannot use render-only offsets.")
        values = (
            self.walk_amplitude,
            self.idle_yaw,
            self.idle_tilt,
            self.walk_frequency,
            self.idle_period,
            self.max_velocity,
            self.max_acceleration,
            self.response_frequency,
            self.display_walk_bob,
            self.display_idle_yaw,
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
        self.env = env
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
        self.moving_fraction = 0.0
        self.idle_fraction = 0.0

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
        moving = min(1.0, max(abs(speed) / 0.10, abs(yaw) / 0.15))
        self.moving_fraction += (1 - math.exp(-dt / 0.35)) * (
            moving - self.moving_fraction
        )
        idle = 1 - moving if mode == "walk" else 0.0
        self.idle_fraction += (1 - math.exp(-dt / 0.35)) * (
            idle - self.idle_fraction
        )
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
            "gaze_correction": (
                "render-only articulation; physical head and locomotion unchanged"
                if self.config.level_gaze_in_render
                else "none"
            ),
        }

    def policy_observations(self, observations):
        """Legacy expression preserves the actor's observation reference."""
        return observations

    def display_offsets(self):
        """Continuous gestures around neutral gaze.

        The legacy controller applies these only when rendering; the physical
        controller uses them as bounded servo references in the final game.
        """
        moving = self.moving_fraction
        bounce = math.sin(2 * math.pi * 1.25 * self.time + self.phase)
        idle = self.idle_fraction
        cycle = (self.time + self.phase + math.pi) % self.config.idle_period
        cycle *= 12 / self.config.idle_period
        glance = 0.0
        if 2 < cycle < 5:
            glance = math.sin(math.pi * (cycle - 2) / 3) ** 2
        elif 7 < cycle < 10:
            glance = -math.sin(math.pi * (cycle - 7) / 3) ** 2
        return {
            "neck": moving * self.config.display_walk_bob * bounce
            + idle * 0.025 * math.sin(2 * math.pi * 0.32 * self.time),
            "yaw": idle * self.config.display_idle_yaw * glance,
            "roll": idle * 0.025 * math.sin(2 * math.pi * 0.23 * self.time),
            "nod": 0.018 * moving * bounce,
        }

    def render(self):
        """Level the displayed gaze in the renderer's private state copy.

        The simulated head retains its validated servo targets and inertia.
        This cosmetic articulation is deliberately not presented as a physical
        head-control result. No observation, reward, or simulation state is edited.
        """
        if not self.config.level_gaze_in_render:
            return self.env.render()
        import mujoco

        renderer = self.env._offline_renderer
        if renderer is None:
            raise RuntimeError("Head presentation requires an initialized renderer.")
        renderer.update(self.env.sim.data)
        model, data = renderer._model, renderer._data
        expression = self.display_offsets()
        for name, offset in (
            ("Neck_Pitch", expression["neck"]),
            ("Head_Yaw", expression["yaw"]),
            ("Head_Roll", expression["roll"]),
        ):
            hinge = model.joint("robot/" + name).id
            address = model.jnt_qposadr[hinge]
            data.qpos[address] = np.clip(
                data.qpos[address] + offset, *model.jnt_range[hinge]
            )
        mujoco.mj_forward(model, data)
        joint = model.joint("robot/Head_Pitch").id
        body = model.body("robot/Head_Base_Plate").id
        forward = data.xmat[body].reshape(3, 3)[:, 0]
        axis = data.xaxis[joint]
        constant = axis[2] * float(axis @ forward)
        cosine = forward[2] - constant
        sine = np.cross(axis, forward)[2]
        # Retain a restrained nod around the horizon instead of a fixed stare.
        target_z = math.sin(expression["nod"])
        correction = 0.0
        for _ in range(5):
            value = (
                cosine * math.cos(correction) + sine * math.sin(correction) + constant
            )
            derivative = -cosine * math.sin(correction) + sine * math.cos(correction)
            if abs(derivative) < 1e-6:
                break
            correction = float(
                np.clip(correction - (value - target_z) / derivative, -1.2, 1.2)
            )
        address = model.jnt_qposadr[joint]
        physical_position = float(data.qpos[address])
        data.qpos[address] = np.clip(
            physical_position + correction, *model.jnt_range[joint]
        )
        mujoco.mj_forward(model, data)
        self.last_render = {
            "physical_head_pitch": physical_position,
            "displayed_head_pitch": float(data.qpos[address]),
            "gaze_elevation_deg": float(
                np.rad2deg(
                    np.arcsin(np.clip(data.xmat[body].reshape(3, 3)[2, 0], -1, 1))
                )
            ),
            "scope": "render-only",
            "expression_offsets": expression,
        }
        renderer.renderer.update_scene(data, camera=renderer._cam)
        return renderer.render()


def make_head_motion(env, config):
    """Select an explicit expression implementation without changing defaults."""
    if config is None or not (config.enabled or config.level_gaze_in_render):
        return None
    if config.physical_expression:
        from bdxr_classroom.simulation.physical_head import PhysicalHeadMotion

        return PhysicalHeadMotion(env, config)
    return HeadMotion(env, config)
