"""Flat-ground BDX-R task built on the credited upstream mjlab environment."""

from dataclasses import replace
import copy
from functools import partial

import torch
from bdx_r_mjlab.tasks.velocity.config.bdxr.env_cfgs import bdxr_flat_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from . import behaviors


def make_env(num_envs=1024, speed=0.3, smoothness=1.0, seed=42, render=False,
             task_version="v2", hop_height=0.04, hop_body_weight=0.0,
             initial_jitter=0.0, command_mode="fixed", speed_min=-0.25,
             speed_max=0.5, yaw_max=0.6, yaw_weight=4.0, proprio_only=False,
             yaw_tracking="full", yaw_std=0.2, course=None,
             linear_weight=None, swing_height=None, training_terrain_height=0.0,
             training_terrain_style="steps", yaw_bias_weight=0.0, reward_scale=1.0):
    cfg = bdxr_flat_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False  # Keep terminal observations for SAC bootstrapping.
    cfg.episode_length_s = 10.0
    cfg.curriculum = {}
    # Start with a clean simulation lesson; sim-to-real randomization is an extension.
    cfg.events = {k: v for k, v in cfg.events.items() if k.startswith("reset_")}
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.01, 0.01),
        "yaw": (-0.05, 0.05),
    }
    if initial_jitter:
        cfg.events["reset_base"].params["pose_range"].update(
            roll=(-initial_jitter, initial_jitter), pitch=(-initial_jitter, initial_jitter))
        cfg.events["reset_robot_joints"].params["position_range"] = (-initial_jitter, initial_jitter)
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (speed, speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = None
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.debug_vis = False
    # Legacy fixed-speed checkpoints retain their original task configuration.
    if command_mode in ["mixed", "balanced"]:
        if command_mode == "balanced":
            from dataclasses import fields
            from .commands import BalancedVelocityCommandCfg
            command = BalancedVelocityCommandCfg(**{
                field.name: copy.deepcopy(getattr(command, field.name)) for field in fields(command)})
            cfg.commands["twist"] = command
        command.ranges.lin_vel_x = (speed_min, speed_max)
        command.ranges.ang_vel_z = (-yaw_max, yaw_max)
        command.resampling_time_range = (3.0, 6.0)
        command.rel_standing_envs = 0.1
        command.rel_forward_envs = 0.0
        command.rel_world_envs = 0.0
        command.init_velocity_prob = 0.0
        cfg.rewards["track_angular_velocity"].weight = yaw_weight
        cfg.rewards["track_angular_velocity"].params["std"] = 0.5
        if yaw_tracking == "rate":
            cfg.rewards["track_angular_velocity"] = RewardTermCfg(
                func=behaviors.yaw_rate_tracking, weight=yaw_weight, params={"std": yaw_std})
    cfg.observations["actor"].enable_corruption = False
    cfg.rewards["action_rate_l2"].weight *= smoothness
    # Head stays under position control at neutral; physical head mass remains.
    cfg.actions["joint_pos"].actuator_names = (
        ".*_Hip_Yaw", ".*_Hip_Roll", ".*_Hip_Pitch", ".*_Knee", ".*_Ankle",
    )
    cfg.actions["joint_pos"].scale = {
        k: v for k, v in cfg.actions["joint_pos"].scale.items()
        if "Head" not in k and "Neck" not in k
    }
    if task_version != "v1":
        cfg.actions["joint_pos"].scale = {
            ".*_Hip_Yaw": 0.2, ".*_Hip_Roll": 0.25,
            ".*_Hip_Pitch": 0.6, ".*_Knee": 0.8, ".*_Ankle": 0.6,
        }
        cfg.rewards["track_linear_velocity"].weight = 8.0
        cfg.rewards["track_linear_velocity"].params["std"] = 0.3
    if task_version == "hop":
        for group in cfg.observations.values():
            group.terms["hop_phase"] = ObservationTermCfg(func=behaviors.hop_phase)
        cfg.rewards["track_linear_velocity"].weight = 2.0
        cfg.rewards["hop_height"] = RewardTermCfg(
            func=behaviors.hop_height_tracking, weight=8.0, params={"height": hop_height})
        cfg.rewards["hop_contacts"] = RewardTermCfg(func=behaviors.hop_contact_timing, weight=4.0)
        cfg.rewards["hop_landing"] = RewardTermCfg(func=behaviors.hop_landing_cost, weight=-1e-6)
        if hop_body_weight:
            cfg.rewards["hop_body_height"] = RewardTermCfg(
                func=behaviors.hop_body_height_tracking, weight=hop_body_weight,
                params={"height": hop_height})
        for key in ["air_time", "foot_clearance", "foot_swing_height"]:
            cfg.rewards.pop(key, None)
    if linear_weight is not None:
        cfg.rewards["track_linear_velocity"].weight = linear_weight
    if swing_height is not None and "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].params["target_height"] = swing_height
    if yaw_bias_weight:
        cfg.rewards["yaw_bias"] = RewardTermCfg(
            func=behaviors.yaw_bias_tracking, weight=yaw_bias_weight,
            params={"std": .025, "time_constant": .3})
    if reward_scale <= 0:
        raise ValueError("Reward scale must be positive")
    # Preserve relative objectives while keeping returns inside the categorical
    # critic's support. Stronger hopping weights can otherwise saturate its top bin.
    for reward in cfg.rewards.values():
        reward.weight *= reward_scale
    allowed = {"base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "hop_phase"}
    if not set(cfg.observations["actor"].terms) <= allowed:
        raise ValueError("Unexpected observation in proprioceptive policy")
    if proprio_only:
        cfg.observations["critic"] = copy.deepcopy(cfg.observations["actor"])
    robot = cfg.scene.entities["robot"]
    robot.articulation = replace(
        robot.articulation,
        actuators=tuple(replace(a, delay_min_lag=0, delay_max_lag=0)
                        for a in robot.articulation.actuators),
    )
    cfg.viewer.width, cfg.viewer.height = 960, 640
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -12
    cfg.viewer.azimuth = 140
    cfg.viewer.max_extra_envs = 0
    if training_terrain_height:
        from .training_terrain import build_training_terrain
        if course is not None or training_terrain_height < 0:
            raise ValueError("Practice terrain requires positive height and no game course")
        if training_terrain_style not in ("steps", "mixed"):
            raise ValueError("Unknown practice terrain style")
        cfg.scene.spec_fn = partial(build_training_terrain, height=training_terrain_height,
                                   style=training_terrain_style)
        cfg.events["reset_base"].params["pose_range"]["x"] = (-.2, .2)
    if course is not None:
        from .course import CourseConfig, build_course
        if num_envs != 1:
            raise ValueError("The game course uses one player robot")
        cfg.scene.terrain.textures = ()
        cfg.scene.terrain.materials = ()
        cfg.scene.spec_fn = partial(build_course, cfg=CourseConfig(**course))
        cfg.episode_length_s = 1e9
        cfg.viewer.distance = 2.0
        cfg.viewer.elevation = -18
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0",
                           render_mode="rgb_array" if render else None)
    env.training_terrain_height = training_terrain_height
    env.training_terrain_style = training_terrain_style
    if training_terrain_height:
        # Each Warp environment is an independent physics world. Give each robot
        # the same tile coordinates rather than mjlab's spaced display origins.
        env.scene.env_origins.zero_()
    return env


def step_and_reset(env, actions):
    """Return terminal observations separately from post-reset policy inputs."""
    obs, rewards, terminated, truncated, info = env.step(actions)
    final_obs = {k: v.clone() for k, v in obs.items()}
    terminated, truncated = terminated.clone(), truncated.clone()
    done = terminated | truncated
    episode_lengths = env.episode_length_buf.clone()
    logs = dict(info.get("log", {}))
    if done.any():
        obs, reset_info = env.reset(env_ids=done.nonzero().flatten())
        logs.update(reset_info.get("log", {}))
    return obs, final_obs, rewards.clone(), terminated, truncated, episode_lengths, logs
