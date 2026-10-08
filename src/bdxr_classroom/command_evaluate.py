"""Evaluate command tracking and path following with full world trajectories."""

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
from PIL import Image, ImageDraw

from .environment import make_env
from .experiments import experiment_root, capture_source, jsonable
from .fastsac import FastSAC
from .motion_metrics import count_hops
from .training_terrain import ground_height


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def set_command(env, vx, wz):
    term = env.command_manager.get_term("twist")
    term.time_left.fill_(1e9)
    for field in ["is_standing_env", "is_heading_env", "is_world_env", "is_forward_env"]:
        getattr(term, field).fill_(False)
    term.vel_command_b[:, 0] = vx
    term.vel_command_b[:, 1] = 0
    term.vel_command_b[:, 2] = wz
    # compute() can return the observation cached before the command changed.
    # Refresh groups without advancing any observation history a second time.
    return {name: env.observation_manager.compute_group(name)
            for name in env.observation_manager.active_terms}


def load_agent(checkpoint, num_envs):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(state["config"], num_envs=num_envs, buffer_size=1, compile=False)
    agent = FastSAC(*state["dimensions"], cfg)
    if state.get("format") == "bdxr-inference-only-v1":
        agent.actor.load_state_dict(state["actor"])
        agent.actor_norm.load_state_dict(state["actor_norm"])
    else:
        agent.load(checkpoint, replay=False)
    return agent, cfg


def run_case(env, agent, case, seconds, video=None, seed=1000):
    env.reset(seed=seed)
    env.cfg.episode_length_s = seconds + 1
    robot = env.scene["robot"].data
    origin = robot.root_link_pos_w[:, :2].clone()
    heading0 = robot.heading_w.clone()
    alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    writer = imageio.get_writer(video, fps=25, codec="libx264", quality=7,
                                 macro_block_size=16) if video else None
    rows = []
    previous_action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    foot_ids, _ = env.scene["robot"].find_sites(["left_foot", "right_foot"])
    try:
        for step in range(round(seconds / env.step_dt)):
            time = step * env.step_dt
            speed, yaw = {
                "slow": (0.15, 0), "forward": (0.3, 0), "fast": (0.45, 0),
                "backward": (-0.2, 0), "stop": (0, 0),
                "left": (0.15, 0.4), "right": (0.15, -0.4),
                "turn_left": (0, 0.4), "turn_right": (0, -0.4),
                "straight": (0.3, 0), "straight_slow": (0.15, 0),
                "straight_fast": (0.45, 0), "straight_back": (-0.2, 0),
                "sequence": (0, 0),
            }[case]
            if case == "sequence":
                sequence = [(0.15, 0), (0.45, 0), (0, 0), (-0.2, 0),
                            (0, 0.4), (0, -0.4), (0.3, 0)]
                speed, yaw = sequence[min(int(time / (seconds / len(sequence))), len(sequence)-1)]
            delta = robot.root_link_pos_w[:, :2] - origin
            lateral = -torch.sin(heading0) * delta[:, 0] + torch.cos(heading0) * delta[:, 1]
            obs = set_command(env, speed, yaw)
            with torch.no_grad():
                actions = agent.act(obs["actor"], deterministic=True)
                _, reward, terminated, truncated, _ = env.step(actions)
            current_alive = alive.clone()
            alive &= ~(terminated | truncated)
            delta = robot.root_link_pos_w[:, :2] - origin
            longitudinal = torch.cos(heading0) * delta[:, 0] + torch.sin(heading0) * delta[:, 1]
            lateral = -torch.sin(heading0) * delta[:, 0] + torch.cos(heading0) * delta[:, 1]
            commands = env.command_manager.get_command("twist").cpu().numpy()
            values = torch.stack([
                robot.root_link_pos_w[:, 0], robot.root_link_pos_w[:, 1],
                robot.heading_w, robot.root_link_lin_vel_b[:, 0],
                robot.root_link_lin_vel_b[:, 1], robot.root_link_ang_vel_b[:, 2],
                longitudinal, lateral, wrap(robot.heading_w - heading0),
            ], dim=1).cpu().numpy()
            changes = (actions - previous_action).square().mean(-1).sqrt().cpu().tolist()
            previous_action = actions.clone()
            foot_positions = robot.site_pos_w[:, foot_ids]
            feet = (foot_positions[..., 2] - ground_height(env, foot_positions)).cpu().tolist()
            contacts = (env.scene["feet_ground_contact"].data.found > 0).reshape(env.num_envs, -1).cpu().tolist()
            joint_angles = robot.joint_pos.cpu().tolist()
            upright = (-robot.projected_gravity_b[:, 2]).cpu().tolist()
            for index in range(env.num_envs):
                if not current_alive[index]:
                    continue
                rows.append({
                    "episode": index, "time": time, "x": float(values[index, 0]),
                    "y": float(values[index, 1]), "heading": float(values[index, 2]),
                    "speed": float(values[index, 3]), "lateral_speed": float(values[index, 4]),
                    "yaw_rate": float(values[index, 5]), "progress": float(values[index, 6]),
                    "cross_track": float(values[index, 7]), "heading_error": float(values[index, 8]),
                    "command_speed": float(commands[index, 0]), "command_yaw": float(commands[index, 2]),
                    "fell": bool(terminated[index]),
                    "action_change_rms": changes[index], "foot_heights": feet[index],
                    "foot_contacts": contacts[index], "joint_angles": joint_angles[index],
                    "upright": upright[index],
                })
            if writer and step % 2 == 0 and current_alive[0]:
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, frame.width, 35), fill="#173f32")
                draw.text((12, 10), f"{case} | {time:.1f}s | vx={commands[0, 0]:.2f} | yaw={commands[0, 2]:.2f}", fill="white")
                writer.append_data(np.asarray(frame))
            if not alive.any():
                break
            done = terminated | truncated
            if done.any():
                # mjlab requires every terminated world to be reset before the
                # next batched step. Its first trial remains marked dead above:
                # reset worlds keep simulating but never contribute new results.
                env.reset(env_ids=done.nonzero().flatten())
    finally:
        if writer:
            writer.close()
    reports = []
    for index in range(env.num_envs):
        episode = [r for r in rows if r["episode"] == index]
        steady = [r for r in episode if r["time"] >= 1] or episode
        clear_flights, landed_hops = count_hops(episode, env.step_dt)
        reports.append({
            "episode": index, "fell": any(r["fell"] for r in episode),
            "duration_s": episode[-1]["time"] + env.step_dt,
            "speed_mae": float(np.mean([abs(r["speed"]-r["command_speed"]) for r in steady])),
            "yaw_mae": float(np.mean([abs(r["yaw_rate"]-r["command_yaw"]) for r in steady])),
            "mean_speed": float(np.mean([r["speed"] for r in steady])),
            "max_cross_track_m": max(abs(r["cross_track"]) for r in episode),
            "final_cross_track_m": episode[-1]["cross_track"],
            "final_progress_m": episode[-1]["progress"],
            "heading_change_deg": float(episode[-1]["heading_error"] * 180 / math.pi),
            "action_change_rms": float(np.mean([r["action_change_rms"] for r in steady])),
            "clear_flights": clear_flights, "landed_hops": landed_hops,
            "peak_simultaneous_foot_clearance": max(min(r["foot_heights"]) for r in steady),
            "swing_height_p95": float(np.percentile([max(r["foot_heights"]) for r in steady], 95)),
        })
    return reports, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--cases", nargs="+", default=["slow", "forward", "fast", "backward", "stop",
                                                      "turn_left", "turn_right", "left", "right", "sequence"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--output")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--terrain-height", type=float, default=0.0,
                        help="Evaluate on physical practice steps; default is flat ground")
    parser.add_argument("--terrain-style", choices=["steps", "mixed"], default="steps")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    checkpoint = Path(args.checkpoint)
    root = experiment_root(checkpoint)
    suffix = f"_{args.terrain_style}_{args.terrain_height:g}" if args.terrain_height else ""
    output = Path(args.output) if args.output else root / "evaluation" / (checkpoint.stem + "_commands" + suffix)
    output.mkdir(parents=True, exist_ok=True)
    video_dir = root / "videos" / output.name
    if not args.no_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    agent, cfg = load_agent(checkpoint, args.episodes)
    env = make_env(args.episodes, cfg["speed"], cfg["smoothness"], render=not args.no_video,
                   task_version=cfg.get("task_version", "v2"), initial_jitter=0.015,
                   hop_height=cfg.get("hop_height", .04), hop_body_weight=cfg.get("hop_body_weight", 0),
                   command_mode=cfg.get("command_mode", "fixed"),
                   yaw_weight=cfg.get("yaw_weight", 4.0),
                   yaw_tracking=cfg.get("yaw_tracking", "full"), yaw_std=cfg.get("yaw_std", 0.2),
                   linear_weight=cfg.get("linear_weight"), swing_height=cfg.get("swing_height"),
                   proprio_only=cfg.get("proprio_only", False),
                   training_terrain_height=args.terrain_height,
                   training_terrain_style=args.terrain_style,
                   yaw_bias_weight=cfg.get("yaw_bias_weight", 0),
                   reward_scale=cfg.get("reward_scale", 1))
    result = {"checkpoint": str(checkpoint), "cases": {}, "terrain_height_m": args.terrain_height,
             "terrain_style": args.terrain_style if args.terrain_height else "flat",
             "straightness_cases": ["slow", "forward", "fast", "backward", "straight", "straight_fast", "straight_back"],
             "sampling": f"{args.episodes} parallel independently perturbed starts per case; shared RNG seed 1000",
             "controller": "Direct velocity commands only. No heading or path feedback; world measurements are evaluation-only."}
    (output / "config.json").write_text(json.dumps({
        "arguments": vars(args), "environment": jsonable(env.cfg),
        "source_archive": str(capture_source(root)),
        "hardware": torch.cuda.get_device_name(0),
        "runtime_environment": os.environ.get("BDXR_RUNTIME", "local"),
    }, indent=2))
    try:
        for case in args.cases:
            reports, rows = run_case(env, agent, case, args.seconds,
                video=video_dir / f"{case}.mp4" if not args.no_video else None)
            result["cases"][case] = reports
            (output / f"{case}_motion.json").write_text(json.dumps(rows))
            (output / "summary.json").write_text(json.dumps(result, indent=2))
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
            for index in range(args.episodes):
                data = [r for r in rows if r["episode"] == index]
                axes[0].plot([r["progress"] for r in data], [r["cross_track"] for r in data])
                axes[1].plot([r["time"] for r in data], [r["speed"] for r in data], alpha=.7)
                axes[2].plot([r["time"] for r in data], [r["yaw_rate"] for r in data], alpha=.7)
            axes[0].set(xlabel="Along initial heading (m)", ylabel="Cross-track (m)")
            axes[0].set_aspect("equal")
            for axis, key, title in [(axes[1], "command_speed", "Speed (m/s)"), (axes[2], "command_yaw", "Yaw rate (rad/s)")]:
                data = [r for r in rows if r["episode"] == 0]
                axis.plot([r["time"] for r in data], [r[key] for r in data], color="black", linestyle="--")
                axis.set(xlabel="Time (s)", ylabel=title)
            fig.suptitle(case)
            fig.savefig(output / f"{case}.png", dpi=140)
            plt.close(fig)
            print(f"{case}: falls={sum(r['fell'] for r in reports)}, "
                        f"speedMAE={np.mean([r['speed_mae'] for r in reports]):.3f}, "
                        f"yawMAE={np.mean([r['yaw_mae'] for r in reports]):.3f}, "
                        f"cross-track={np.mean([r['max_cross_track_m'] for r in reports]):.3f}m", flush=True)
    except Exception as error:
        (output / "error.json").write_text(json.dumps(
            {"error_type": type(error).__name__, "error": str(error)}, indent=2))
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
