"""Evaluate every episode, stream an MP4, and export real motion measurements."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .environment import make_env
from .fastsac import FastSAC
from .experiments import experiment_root
from .training_terrain import sampled_sole_clearance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--speed", type=float)
    p.add_argument("--output")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--initial-jitter", type=float, default=0.015,
                   help="Initial joint and roll/pitch perturbation bound in radians")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    path = Path(args.checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(state["config"])
    cfg.update(num_envs=1, buffer_size=1, compile=False)
    speed = cfg["speed"] if args.speed is None else args.speed
    root = experiment_root(path)
    out = Path(args.output or str(root / "evaluation" / f"eval_{path.stem}"))
    out.mkdir(parents=True, exist_ok=True)
    video_dir = root / "videos" / out.name
    if not args.no_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(1, speed, cfg["smoothness"], render=not args.no_video,
                   task_version=cfg.get("task_version", "v1"),
                   hop_height=cfg.get("hop_height", 0.04),
                   hop_height_reward=cfg.get("hop_height_reward", "continuous"),
                   hop_clearance=cfg.get("hop_clearance", "site"),
                   hop_body_weight=cfg.get("hop_body_weight", 0.0),
                   proprio_only=cfg.get("proprio_only", False),
                   linear_weight=cfg.get("linear_weight"),
                   linear_tracking=cfg.get("linear_tracking", "instant"),
                   swing_height=cfg.get("swing_height"),
                   level_feet_weight=cfg.get("level_feet_weight", 0),
                   hop_flight_weight=cfg.get("hop_flight_weight", 0),
                   level_feet_flight_only=cfg.get("level_feet_flight_only", False),
                   walking_height=cfg.get("walking_height"),
                   swing_weight=cfg.get("swing_weight"),
                   swing_clearance=cfg.get("swing_clearance", "site"),
                   air_time_min=cfg.get("air_time_min"), air_time_max=cfg.get("air_time_max"),
                   air_time_weight=cfg.get("air_time_weight"),
                   hop_flight_duration=cfg.get("hop_flight_duration", .25),
                   hop_crouch=cfg.get("hop_crouch", 0),
                   yaw_bias_weight=cfg.get("yaw_bias_weight", 0),
                   lateral_bias_weight=cfg.get("lateral_bias_weight", 0),
                   reward_scale=cfg.get("reward_scale", 1),
                   initial_jitter=args.initial_jitter)
    env.cfg.episode_length_s = args.seconds
    foot_ids, _ = env.scene["robot"].find_sites(["left_foot", "right_foot"])
    agent = FastSAC(*state["dimensions"], cfg)
    if state.get("format") == "bdxr-inference-only-v1":
        agent.actor.load_state_dict(state["actor"])
        agent.actor_norm.load_state_dict(state["actor_norm"])
    else:
        agent.load(path, replay=False)
    episode_reports, samples = [], []
    writer = None
    if not args.no_video:
        writer = imageio.get_writer(video_dir / "rollout.mp4", fps=25,
                                     codec="libx264", quality=8, macro_block_size=16)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=1000 + episode)
            reward_total = 0.0
            fell = False
            previous_action = torch.zeros(1, env.action_manager.total_action_dim, device="cuda")
            steps = int(args.seconds / env.step_dt)
            for t in range(steps):
                with torch.no_grad():
                    actions = agent.act(obs["actor"], deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(actions)
                robot = env.scene["robot"].data
                sample = {
                    "episode": episode, "time": t * env.step_dt,
                    "speed": float(robot.root_link_lin_vel_b[0, 0]),
                    "height": float(robot.root_link_pos_w[0, 2]),
                    "reward": float(reward[0]),
                    "joint_angles": robot.joint_pos[0].cpu().numpy().tolist(),
                    "upright": float(-robot.projected_gravity_b[0, 2]),
                    "foot_heights": (sampled_sole_clearance(env)[0] if env.hop_clearance == "sole"
                                     else robot.site_pos_w[0, foot_ids, 2]).cpu().tolist(),
                    "foot_contacts": (env.scene["feet_ground_contact"].data.found[0] > 0).cpu().flatten().tolist(),
                    "action_change_rms": float((actions - previous_action).square().mean().sqrt()),
                    "joint_torque_rms": float(robot.qfrc_actuator[0].square().mean().sqrt()),
                }
                previous_action = actions.clone()
                samples.append(sample)
                reward_total += sample["reward"]
                if writer and episode == 0 and t % 2 == 0:
                    writer.append_data(env.render())
                if terminated.item() or truncated.item():
                    fell = bool(terminated.item())
                    break
            episode_samples = [s for s in samples if s["episode"] == episode]
            steady = [s for s in episode_samples if s["time"] >= 1.0]
            # Ignore the initial drop onto the floor; count only clear flights
            # followed by sustained upright, two-foot contact.
            flights = []
            in_flight = False
            for index, sample in enumerate(steady):
                airborne = not any(sample["foot_contacts"]) and min(sample["foot_heights"]) > 0.015
                if airborne and not in_flight:
                    flight_start = index
                if in_flight and not airborne:
                    if index - flight_start >= 2:
                        flights.append((flight_start, index))
                in_flight = airborne
            landed_hops = 0
            for flight_index, (flight_start, flight_end) in enumerate(flights):
                next_flight = flights[flight_index + 1][0] if flight_index + 1 < len(flights) else len(steady)
                # A rebound before settling is not a separate stable landing.
                after = steady[flight_end:min(next_flight, flight_end + int(1.0 / env.step_dt))]
                stable_count = 0
                for sample in after:
                    stable_count = stable_count + 1 if all(sample["foot_contacts"]) and sample["upright"] > 0.9 else 0
                    if stable_count >= math.ceil(0.15 / env.step_dt):
                        landed_hops += 1
                        break
            episode_reports.append({
                "seed": 1000 + episode, "fell": fell,
                "duration_seconds": (t + 1) * env.step_dt,
                "mean_speed": float(np.mean([s["speed"] for s in episode_samples])),
                "return": reward_total,
                "peak_foot_height": max(max(s["foot_heights"]) for s in episode_samples),
                "airborne_fraction": float(np.mean([not any(s["foot_contacts"]) for s in episode_samples])),
                "steady_peak_min_foot_height": max((min(s["foot_heights"]) for s in steady), default=0.0),
                "clear_flights": len(flights),
                "landed_hops": landed_hops,
                "steady_action_change_rms": float(np.mean([s["action_change_rms"] for s in steady])) if steady else 0,
                "contact_transitions": [
                    sum(episode_samples[i]["foot_contacts"][foot] !=
                        episode_samples[i-1]["foot_contacts"][foot]
                        for i in range(1, len(episode_samples)))
                    for foot in range(2)
                ],
            })
            if writer and episode == 0:
                writer.close()
                writer = None
    finally:
        if writer:
            writer.close()
        env.close()
    report = {
        "checkpoint": str(path), "training_step": state["step"],
        "video_path": str(video_dir / "rollout.mp4") if not args.no_video else None,
        "target_speed": speed, "episodes": episode_reports,
        "foot_height_measurement": ("Sampled collision-sole clearance" if cfg.get("hop_clearance") == "sole"
                                    else "Foot reference-site height"),
        "initial_jitter_radians": args.initial_jitter,
        "fall_fraction": float(np.mean([r["fell"] for r in episode_reports])),
        "mean_speed": float(np.mean([r["mean_speed"] for r in episode_reports])),
        "label": (
            ("starting policy for fine-tuning" if cfg.get("finetune") else "untrained initial policy")
            if state["step"] == 0 else
            ("fine-tuned" if cfg.get("finetune") else "trained from scratch")
        ),
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    (out / "motion.json").write_text(json.dumps(samples))
    first = [s for s in samples if s["episode"] == 0]
    ts = [s["time"] for s in first]
    is_hop = cfg.get("task_version") == "hop"
    fig, axs = plt.subplots(3 if is_hop else 2, 1, figsize=(9, 8 if is_hop else 6),
                            constrained_layout=True)
    if is_hop:
        axs[0].plot(ts, [s["height"] for s in first], label="Body")
        axs[0].set(ylabel="Body height (m)")
        for foot, label in enumerate(["Left foot", "Right foot"]):
            axs[1].plot(ts, [s["foot_heights"][foot] for s in first], label=label)
        axs[1].set(ylabel=("Sampled sole clearance (m)" if cfg.get("hop_clearance") == "sole"
                          else "Foot site height (m)"))
        axs[1].legend()
    else:
        axs[0].plot(ts, [s["speed"] for s in first], label="Measured")
        axs[0].axhline(speed, color="orange", linestyle="--", label="Target")
        axs[0].set(ylabel="Forward speed (m/s)")
        axs[0].legend()
    names = env.scene["robot"].joint_names
    for joint in ["Left_Knee", "Right_Knee"]:
        idx = names.index(joint)
        axs[-1].plot(ts, [s["joint_angles"][idx] for s in first], label=joint)
    axs[-1].set(xlabel="Time (s)", ylabel="Joint angle (rad)")
    axs[-1].legend()
    fig.savefig(out / "motion.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
