"""Small, interpretable reward experiments for the classroom."""

import math
import torch

from .training_terrain import ground_height, sampled_sole_clearance


def imu_turn_rate(env):
    """Yaw rate from body gyro and gravity, without an absolute heading sensor.

    ZYX kinematics gives psi_dot = (sin(roll)*wy + cos(roll)*wz)/cos(pitch).
    Projected gravity supplies those tilt terms directly. Body gyro Z alone
    includes roll/pitch coupling when the robot leans.
    """
    robot = env.scene["robot"].data
    gravity_yz = robot.projected_gravity_b[:, 1:]
    return -(gravity_yz * robot.root_link_ang_vel_b[:, 1:]).sum(-1) / (
        gravity_yz.square().sum(-1).clamp_min(.1))


def yaw_rate_tracking(env, std=0.2, tilt_compensation=False):
    """Separate yaw-command tracking from the robot's normal roll/pitch motion."""
    command = env.command_manager.get_command("twist")[:, 2]
    actual = imu_turn_rate(env) if tilt_compensation else env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-(actual - command).square() / std**2)


def yaw_bias_tracking(env, std=0.025, time_constant=0.3, tilt_compensation=False):
    """Penalize sustained gyro bias while allowing normal alternating gait motion.

    This is a training reward, not a heading controller. It uses neither world
    heading nor position, and adds nothing to the actor's observation vector.
    """
    actual = imu_turn_rate(env) if tilt_compensation else env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    error = actual - env.command_manager.get_command("twist")[:, 2]
    old = getattr(env, "_yaw_bias_error", torch.zeros_like(error))
    old = torch.where(env.episode_length_buf <= 1, torch.zeros_like(old), old)
    fraction = 1 - math.exp(-env.step_dt / time_constant)
    env._yaw_bias_error = torch.lerp(old, error, fraction)
    return torch.exp(-env._yaw_bias_error.square() / std**2)


def average_linear_velocity_tracking(env, std=0.3, time_constant=0.3):
    """Reward average progress without penalizing every hop's velocity oscillation.

    Simulator velocity is privileged reward information; it is never added to
    the actor or used by the deployed mode-switching controller.
    """
    error = (env.scene["robot"].data.root_link_lin_vel_b[:, :2]
             - env.command_manager.get_command("twist")[:, :2])
    old = getattr(env, "_linear_tracking_error", torch.zeros_like(error))
    old = torch.where((env.episode_length_buf <= 1)[:, None], torch.zeros_like(old), old)
    fraction = 1 - math.exp(-env.step_dt / time_constant)
    env._linear_tracking_error = torch.lerp(old, error, fraction)
    return torch.exp(-env._linear_tracking_error.square().sum(-1) / std**2)


def hop_phase(env):
    phase = (env.episode_length_buf * env.step_dt) % 1.0
    angle = 2 * math.pi * phase
    return torch.stack((angle.sin(), angle.cos()), dim=-1)


def hop_height_tracking(env, height=0.04, flight_only=False, clearance_mode="site"):
    """Request a short flight in the middle of each one-second cycle."""
    phase = (env.episode_length_buf * env.step_dt) % 1.0
    flight_phase = ((phase - 0.35) / 0.25).clamp(0, 1)
    desired = height * torch.sin(math.pi * flight_phase)
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_sites(["left_foot", "right_foot"])
    positions = robot.data.site_pos_w[:, foot_ids]
    actual = (sampled_sole_clearance(env) if clearance_mode == "sole" else
              positions[..., 2] - ground_height(env, positions))
    error = (actual - desired[:, None]).square().mean(-1)
    reward = torch.exp(-error / 0.025**2)
    if flight_only:
        # Award the height objective during the requested flight, rather than
        # paying most of it for keeping feet on the floor between jumps.
        # Normalize the quarter-cycle window to preserve its possible mean reward.
        reward = reward * ((phase > .35) & (phase < .60)) * 4
    return reward


def hop_contact_timing(env):
    phase = (env.episode_length_buf * env.step_dt) % 1.0
    flight = (phase > 0.39) & (phase < 0.56)
    contact = env.scene["feet_ground_contact"].data.found > 0
    # Both feet leave and return together. Partial support earns partial credit.
    contact = contact.reshape(env.num_envs, -1)
    return (contact == (~flight[:, None])).float().mean(-1)


def hop_body_height_tracking(env, height=0.06):
    phase = (env.episode_length_buf * env.step_dt) % 1.0
    flight_phase = ((phase - 0.35) / 0.25).clamp(0, 1)
    desired = 0.265 + height * torch.sin(math.pi * flight_phase)
    position = env.scene["robot"].data.root_link_pos_w
    actual = position[:, 2] - ground_height(env, position)
    return torch.exp(-(actual - desired).square() / 0.035**2)


def hop_landing_cost(env):
    sensor = env.scene["feet_ground_contact"].data
    force = sensor.force.reshape(env.num_envs, -1, 3)
    return force.square().sum(dim=(-1, -2))
