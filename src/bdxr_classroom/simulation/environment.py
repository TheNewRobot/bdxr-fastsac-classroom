"""Flat-ground BDX-R task built on the credited upstream mjlab environment."""

from dataclasses import replace
import copy
import math
from functools import partial

from bdx_r_mjlab.tasks.velocity.config.bdxr.env_cfgs import bdxr_flat_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from bdxr_classroom.simulation import behaviors


def make_env(
    num_envs=1024,
    speed=0.3,
    smoothness=1.0,
    seed=42,
    render=False,
    task_version="v2",
    hop_height=0.04,
    hop_body_weight=0.0,
    initial_jitter=0.0,
    command_mode="fixed",
    speed_min=-0.25,
    speed_max=0.5,
    yaw_max=0.6,
    yaw_weight=4.0,
    proprio_only=False,
    yaw_tracking="full",
    yaw_std=0.2,
    course=None,
    full_link_collisions=False,
    quiet_standing_weight=0.0,
    idle_velocity_weight=0.0,
    contact_spin_weight=0.0,
    contact_spin_rate_clip=None,
    contact_spin_turn_only=False,
    linear_weight=None,
    swing_height=None,
    training_terrain_height=0.0,
    training_terrain_style="steps",
    yaw_bias_weight=0.0,
    reward_scale=1.0,
    lateral_bias_weight=0.0,
    linear_tracking="instant",
    hop_height_reward="continuous",
    hop_clearance="site",
    training_start_span=0.2,
    level_feet_weight=0.0,
    hop_flight_weight=0.0,
    swing_weight=None,
    level_feet_flight_only=False,
    walking_height=None,
    walking_width=None,
    air_time_min=None,
    air_time_max=None,
    air_time_weight=None,
    hop_flight_duration=0.25,
    hop_crouch=0.0,
    swing_clearance="site",
):
    """Create parallel BDX-R tasks with explicit command and reward settings.

    Distances are metres, angles are radians, speeds are metres/second, and yaw
    rates are radians/second. ``task_version`` preserves the action/observation
    conventions of saved policies. The returned ``env.cfg`` exposes the resolved
    physics, actuators, rewards, observations, and termination configuration.
    """
    if not 0.12 <= hop_flight_duration <= 0.5 or not 0 <= hop_crouch <= 0.08:
        raise ValueError("Invalid hop duration or crouch depth")
    if walking_height is not None and (
        not math.isfinite(walking_height) or walking_height <= 0
    ):
        raise ValueError("Walking height must be a positive finite number")
    if walking_width is not None and (
        not math.isfinite(walking_width) or walking_width <= 0
    ):
        raise ValueError("Walking width must be a positive finite number")
    if hop_clearance not in ("site", "sole"):
        raise ValueError("Hop clearance must use site or sole measurements")
    if not math.isfinite(initial_jitter) or initial_jitter < 0:
        raise ValueError("Initial jitter must be finite and nonnegative")
    cfg = bdxr_flat_env_cfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False  # Keep terminal observations for SAC bootstrapping.
    cfg.episode_length_s = 10.0
    cfg.curriculum = {}
    # Start with a clean simulation lesson; sim-to-real randomization is an extension.
    cfg.events = {k: v for k, v in cfg.events.items() if k.startswith("reset_")}
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.01, 0.01),
        "yaw": (-0.05, 0.05),
    }
    if initial_jitter:
        cfg.events["reset_base"].params["pose_range"].update(
            roll=(-initial_jitter, initial_jitter),
            pitch=(-initial_jitter, initial_jitter),
        )
        cfg.events["reset_robot_joints"].params["position_range"] = (
            -initial_jitter,
            initial_jitter,
        )
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
            from bdxr_classroom.simulation.commands import BalancedVelocityCommandCfg

            command = BalancedVelocityCommandCfg(
                **{
                    field.name: copy.deepcopy(getattr(command, field.name))
                    for field in fields(command)
                }
            )
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
        if yaw_tracking in ("rate", "imu"):
            cfg.rewards["track_angular_velocity"] = RewardTermCfg(
                func=behaviors.yaw_rate_tracking,
                weight=yaw_weight,
                params={"std": yaw_std, "tilt_compensation": yaw_tracking == "imu"},
            )
    cfg.observations["actor"].enable_corruption = False
    cfg.rewards["action_rate_l2"].weight *= smoothness
    # Head stays under position control at neutral; physical head mass remains.
    cfg.actions["joint_pos"].actuator_names = (
        ".*_Hip_Yaw",
        ".*_Hip_Roll",
        ".*_Hip_Pitch",
        ".*_Knee",
        ".*_Ankle",
    )
    cfg.actions["joint_pos"].scale = {
        k: v
        for k, v in cfg.actions["joint_pos"].scale.items()
        if "Head" not in k and "Neck" not in k
    }
    if task_version != "v1":
        cfg.actions["joint_pos"].scale = {
            ".*_Hip_Yaw": 0.2,
            ".*_Hip_Roll": 0.25,
            ".*_Hip_Pitch": 0.6,
            ".*_Knee": 0.8,
            ".*_Ankle": 0.6,
        }
        cfg.rewards["track_linear_velocity"].weight = 8.0
        cfg.rewards["track_linear_velocity"].params["std"] = 0.3
    if task_version == "hop":
        for group in cfg.observations.values():
            group.terms["hop_phase"] = ObservationTermCfg(func=behaviors.hop_phase)
        cfg.rewards["track_linear_velocity"].weight = 2.0
        cfg.rewards["hop_height"] = RewardTermCfg(
            func=behaviors.hop_height_tracking,
            weight=8.0,
            params={
                "height": hop_height,
                "flight_only": hop_height_reward in ("flight", "flight_both"),
                "bilateral": hop_height_reward == "flight_both",
                "clearance_mode": hop_clearance,
                "flight_duration": hop_flight_duration,
            },
        )
        cfg.rewards["hop_contacts"] = RewardTermCfg(
            func=behaviors.hop_contact_timing,
            weight=4.0,
            params={"flight_duration": hop_flight_duration},
        )
        if hop_flight_weight:
            cfg.rewards["clear_flight"] = RewardTermCfg(
                func=behaviors.clear_hop_flight,
                weight=hop_flight_weight,
                params={"flight_duration": hop_flight_duration},
            )
        cfg.rewards["hop_landing"] = RewardTermCfg(
            func=behaviors.hop_landing_cost, weight=-1e-6
        )
        if hop_body_weight:
            cfg.rewards["hop_body_height"] = RewardTermCfg(
                func=behaviors.hop_body_height_tracking,
                weight=hop_body_weight,
                params={
                    "height": hop_height,
                    "flight_duration": hop_flight_duration,
                    "crouch": hop_crouch,
                },
            )
        for key in ["air_time", "foot_clearance", "foot_swing_height"]:
            cfg.rewards.pop(key, None)
    if linear_weight is not None:
        cfg.rewards["track_linear_velocity"].weight = linear_weight
    if linear_tracking == "average":
        previous = cfg.rewards["track_linear_velocity"]
        cfg.rewards["track_linear_velocity"] = RewardTermCfg(
            func=behaviors.average_linear_velocity_tracking,
            weight=previous.weight,
            params={"std": previous.params["std"], "time_constant": 0.3},
        )
    if swing_height is not None and "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].params["target_height"] = swing_height
    if swing_weight is not None and "foot_swing_height" in cfg.rewards:
        cfg.rewards["foot_swing_height"].weight = -abs(swing_weight)
    if swing_clearance not in ("site", "sole"):
        raise ValueError("Swing clearance must be site or sole")
    if swing_clearance == "sole" and "foot_swing_height" in cfg.rewards:
        previous = cfg.rewards["foot_swing_height"]
        cfg.rewards["foot_swing_height"] = RewardTermCfg(
            func=behaviors.sole_swing_height_cost,
            weight=previous.weight,
            params={"target_height": previous.params["target_height"]},
        )
    if "air_time" in cfg.rewards:
        if air_time_weight is not None:
            cfg.rewards["air_time"].weight = air_time_weight
        if air_time_min is not None:
            cfg.rewards["air_time"].params["threshold_min"] = air_time_min
        if air_time_max is not None:
            cfg.rewards["air_time"].params["threshold_max"] = air_time_max
    if walking_height is not None:
        cfg.rewards["walking_height"] = RewardTermCfg(
            func=behaviors.walking_height_reward,
            weight=4.0,
            params={"target_height": walking_height},
        )
    if walking_width is not None:
        cfg.rewards["walking_width"] = RewardTermCfg(
            func=behaviors.foot_spacing_reward,
            weight=2.0,
            params={"target_spacing": walking_width},
        )
    if level_feet_weight:
        cfg.rewards["level_feet"] = RewardTermCfg(
            func=behaviors.level_feet_reward,
            weight=level_feet_weight,
            params={"flight_only": level_feet_flight_only},
        )
    if yaw_bias_weight:
        cfg.rewards["yaw_bias"] = RewardTermCfg(
            func=behaviors.yaw_bias_tracking,
            weight=yaw_bias_weight,
            params={
                "std": 0.025,
                "time_constant": 0.3,
                "tilt_compensation": yaw_tracking == "imu",
            },
        )
    if lateral_bias_weight:
        cfg.rewards["lateral_bias"] = RewardTermCfg(
            func=behaviors.lateral_bias_tracking,
            weight=lateral_bias_weight,
            params={"std": 0.025, "time_constant": 0.3},
        )
    if reward_scale <= 0:
        raise ValueError("Reward scale must be positive")
    if quiet_standing_weight:
        if task_version == "hop":
            raise ValueError("Quiet standing is a walking-mode objective; explicit hopping remains active.")
        cfg.rewards["quiet_standing"] = RewardTermCfg(
            func=behaviors.quiet_standing_reward,
            weight=quiet_standing_weight,
        )
    if idle_velocity_weight:
        cfg.rewards["idle_velocity"] = RewardTermCfg(
            func=behaviors.idle_motion_cost,
            weight=-idle_velocity_weight,
        )
    if contact_spin_rate_clip is not None and (
        not math.isfinite(contact_spin_rate_clip) or contact_spin_rate_clip <= 0
    ):
        raise ValueError("Contact spin rate clip must be positive and finite.")
    if contact_spin_weight:
        cfg.rewards["contact_spin"] = RewardTermCfg(
            func=behaviors.contact_spin_cost,
            weight=-contact_spin_weight,
            params={
                "max_rate": contact_spin_rate_clip,
                "turn_only": contact_spin_turn_only,
            },
        )
    # Preserve relative objectives while keeping returns inside the categorical
    # critic's support. Stronger hopping weights can otherwise saturate its top bin.
    for reward in cfg.rewards.values():
        reward.weight *= reward_scale
    allowed = {
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "hop_phase",
    }
    if not set(cfg.observations["actor"].terms) <= allowed:
        raise ValueError("Unexpected observation in proprioceptive policy")
    if proprio_only:
        cfg.observations["critic"] = copy.deepcopy(cfg.observations["actor"])
    robot = cfg.scene.entities["robot"]
    if full_link_collisions:
        from bdxr_classroom.game.collisions import full_link_collision_config

        robot = full_link_collision_config(robot)
        cfg.scene.entities["robot"] = robot
        # The training heuristic sizes buffers for a small primitive model.
        # Full-link falls and wall contacts can involve many more mesh pairs.
        cfg.sim.nconmax = 512
        cfg.sim.njmax = 2048
        cfg.sim.contact_sensor_maxmatch = 256
        # Multi-point convex contact can generate distant, invalid contact
        # positions for a leg cover against a gate. Single-point CCD retains
        # every collision shape and resolves the captured failing state.
        cfg.sim.mujoco.disableflags = (
            *cfg.sim.mujoco.disableflags,
            "multiccd",
        )
    robot.articulation = replace(
        robot.articulation,
        actuators=tuple(
            replace(a, delay_min_lag=0, delay_max_lag=0)
            for a in robot.articulation.actuators
        ),
    )
    cfg.viewer.width, cfg.viewer.height = 960, 640
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -12
    cfg.viewer.azimuth = 140
    cfg.viewer.max_extra_envs = 0
    if training_terrain_height:
        from bdxr_classroom.simulation.training_terrain import build_training_terrain

        if course is not None or training_terrain_height < 0:
            raise ValueError(
                "Practice terrain requires positive height and no game course"
            )
        if training_terrain_style not in ("steps", "mixed"):
            raise ValueError("Unknown practice terrain style")
        cfg.scene.spec_fn = partial(
            build_training_terrain,
            height=training_terrain_height,
            style=training_terrain_style,
        )
        if not 0 <= training_start_span < float("inf"):
            raise ValueError("Training start span must be finite and nonnegative")
        cfg.events["reset_base"].params["pose_range"]["x"] = (
            -training_start_span,
            training_start_span,
        )
    if course is not None:
        from bdxr_classroom.game.course import CourseConfig, build_course

        if num_envs != 1:
            raise ValueError("The game course uses one player robot")
        cfg.scene.terrain.textures = ()
        cfg.scene.terrain.materials = ()
        cfg.scene.spec_fn = partial(build_course, cfg=CourseConfig(**course))
        cfg.episode_length_s = 1e9
        cfg.viewer.distance = 2.0
        cfg.viewer.elevation = -18
    env = ManagerBasedRlEnv(
        cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None
    )
    env.training_terrain_height = training_terrain_height
    env.training_terrain_style = training_terrain_style
    env.hop_clearance = hop_clearance
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
