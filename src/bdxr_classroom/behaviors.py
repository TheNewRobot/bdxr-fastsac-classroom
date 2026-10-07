"""Small, interpretable reward experiments for the classroom."""

import math
import torch


def hop_phase(env):
    phase = (env.episode_length_buf * env.step_dt) % 1.0
    angle = 2 * math.pi * phase
    return torch.stack((angle.sin(), angle.cos()), dim=-1)


def hop_height_tracking(env, height=0.04):
    """Request a short flight in the middle of each one-second cycle."""
    phase = (env.episode_length_buf * env.step_dt) % 1.0
    flight_phase = ((phase - 0.35) / 0.25).clamp(0, 1)
    desired = height * torch.sin(math.pi * flight_phase)
    robot = env.scene["robot"]
    foot_ids, _ = robot.find_sites(["left_foot", "right_foot"])
    actual = robot.data.site_pos_w[:, foot_ids, 2]
    error = (actual - desired[:, None]).square().mean(-1)
    return torch.exp(-error / 0.025**2)


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
    actual = env.scene["robot"].data.root_link_pos_w[:, 2]
    return torch.exp(-(actual - desired).square() / 0.035**2)


def hop_landing_cost(env):
    sensor = env.scene["feet_ground_contact"].data
    force = sensor.force.reshape(env.num_envs, -1, 3)
    return force.square().sum(dim=(-1, -2))
