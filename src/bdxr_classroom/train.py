"""Train BDX-R with FastSAC, writing local metrics and optional W&B runs."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import wandb

from .environment import make_env, step_and_reset
from .fastsac import FastSAC
from .experiments import create_experiment, experiment_root, write_manifest, capture_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="walk")
    p.add_argument("--run-root", default="experiments")
    p.add_argument("--result-file", help="Write the actual timestamped run path for notebook automation")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--smoothness", type=float, default=1)
    p.add_argument("--command-mode", choices=["fixed", "mixed", "balanced"], default="fixed")
    p.add_argument("--speed-min", type=float, default=-0.25)
    p.add_argument("--speed-max", type=float, default=0.5)
    p.add_argument("--yaw-max", type=float, default=0.6)
    p.add_argument("--yaw-weight", type=float, default=4.0)
    p.add_argument("--yaw-tracking", choices=["full", "rate", "imu"], default="full")
    p.add_argument("--yaw-std", type=float, default=0.2)
    p.add_argument("--yaw-bias-weight", type=float, default=0.0)
    p.add_argument("--lateral-bias-weight", type=float, default=0.0)
    p.add_argument("--linear-weight", type=float)
    p.add_argument("--linear-tracking", choices=["instant", "average"], default="instant")
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--swing-height", type=float)
    p.add_argument("--swing-weight", type=float)
    p.add_argument("--swing-clearance", choices=["site", "sole"], default="site")
    p.add_argument("--air-time-min", type=float)
    p.add_argument("--air-time-max", type=float)
    p.add_argument("--air-time-weight", type=float)
    p.add_argument("--level-feet-weight", type=float, default=0.0)
    p.add_argument("--level-feet-flight-only", action="store_true")
    p.add_argument("--walking-height", type=float, help="Optional torso-height reward target in metres")
    p.add_argument("--hop-flight-weight", type=float, default=0.0)
    p.add_argument("--training-terrain-height", type=float, default=0.0)
    p.add_argument("--training-terrain-style", choices=["steps", "mixed"], default="steps")
    p.add_argument("--training-start-span", type=float, default=0.2,
                   help="Terrain practice resets uniformly within +/- this distance in metres")
    p.add_argument("--initial-jitter", type=float, default=0.0,
                   help="Random reset roll, pitch and joint offsets in radians")
    p.add_argument("--proprio-only", action=argparse.BooleanOptionalAction, default=False,
                   help="Optional symmetric-critic ablation; default critic uses privileged observations")
    p.add_argument("--task-version", choices=["v1", "v2", "hop"], default="v2")
    p.add_argument("--hop-height", type=float, default=0.04)
    p.add_argument("--hop-height-reward", choices=["continuous", "flight", "flight_both"],
                   default="continuous")
    p.add_argument("--hop-clearance", choices=["site", "sole"], default="site")
    p.add_argument("--mirror-replay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hop-body-weight", type=float, default=0.0)
    p.add_argument("--hop-flight-duration", type=float, default=0.25)
    p.add_argument("--hop-crouch", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--buffer-size", type=int, default=512)
    p.add_argument("--actor-hidden", type=int, default=256)
    p.add_argument("--critic-hidden", type=int, default=384)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume")
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--project", default="bdxr-fastsac-classroom")
    args = p.parse_args()
    cfg = vars(args)
    cfg["runtime_environment"] = os.environ.get("BDXR_RUNTIME", "local")
    cfg["random_stream_mode"] = ("new_seed" if args.finetune or not args.resume
                                 else "checkpoint")
    cfg["hardware"] = torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    existing_run = bool(args.resume and not args.finetune)
    out = experiment_root(args.resume) if existing_run else create_experiment(args.name, args.run_root)
    if existing_run:
        original = out / "config/training.json"
        if not original.exists():
            raise ValueError("Migrate this legacy run before resuming, or use --finetune for a new experiment")
        previous_cfg = json.loads(original.read_text())
        for key in ["name", "project", "num_envs", "speed", "smoothness", "task_version", "hop_height",
                    "hop_body_weight", "seed", "batch_size", "updates", "buffer_size",
                    "actor_hidden", "critic_hidden", "gamma", "compile", "amp",
                    "command_mode", "speed_min", "speed_max", "yaw_max", "yaw_weight",
                    "yaw_tracking", "yaw_std", "proprio_only", "linear_weight", "swing_height",
                    "training_terrain_height", "training_terrain_style", "training_start_span", "initial_jitter",
                    "yaw_bias_weight", "lateral_bias_weight", "reward_scale",
                    "linear_tracking", "hop_height_reward", "hop_clearance", "mirror_replay",
                    "swing_weight", "air_time_min", "air_time_max", "air_time_weight", "level_feet_weight", "hop_flight_weight", "level_feet_flight_only", "walking_height",
                    "hop_flight_duration", "hop_crouch", "swing_clearance"]:
            if key in previous_cfg:
                setattr(args, key, previous_cfg[key])
    cfg["experiment_path"] = str(out)
    if args.result_file:
        Path(args.result_file).write_text(str(out.resolve()))
    print(f"Experiment: {out}", flush=True)
    config_path = out / "config" / (f"resume_{time.time_ns()}.json" if existing_run else "training.json")
    config_path.write_text(json.dumps(cfg, indent=2))
    previous_timing = {}
    if existing_run and (out / "metrics/timing.json").exists():
        previous_timing = json.loads((out / "metrics/timing.json").read_text())
    previous_training_seconds = previous_timing.get("training_seconds", 0.0)
    wandb_resume = {}
    if existing_run and args.wandb_mode == "online" and (out / "metrics/wandb.json").exists():
        previous_wandb = json.loads((out / "metrics/wandb.json").read_text())
        if previous_wandb.get("url"):
            wandb_resume = {"id": previous_wandb["id"], "resume": "allow"}
    run = wandb.init(project=args.project, name=args.name, config=cfg, dir=str(out),
                     mode=args.wandb_mode, allow_val_change=True,
                     tags=["FastSAC", "BDX-R", "classroom", cfg["runtime_environment"], cfg["hardware"]],
                     **wandb_resume)
    if run:
        (out / "metrics/wandb.json").write_text(json.dumps({"url": run.url, "id": run.id}))
    setup_start = time.monotonic()
    env = make_env(args.num_envs, args.speed, args.smoothness, args.seed,
                 initial_jitter=args.initial_jitter,
                 task_version=args.task_version, hop_height=args.hop_height,
                 hop_body_weight=args.hop_body_weight, command_mode=args.command_mode,
                 speed_min=args.speed_min, speed_max=args.speed_max,
                 yaw_max=args.yaw_max, yaw_weight=args.yaw_weight, proprio_only=args.proprio_only,
                 yaw_tracking=args.yaw_tracking, yaw_std=args.yaw_std,
                 linear_weight=args.linear_weight, swing_height=args.swing_height,
                 training_terrain_height=args.training_terrain_height,
                 training_terrain_style=args.training_terrain_style,
                 training_start_span=args.training_start_span,
                 yaw_bias_weight=args.yaw_bias_weight, reward_scale=args.reward_scale,
                 lateral_bias_weight=args.lateral_bias_weight,
                 linear_tracking=args.linear_tracking, hop_height_reward=args.hop_height_reward,
                 hop_clearance=args.hop_clearance, level_feet_weight=args.level_feet_weight,
                 hop_flight_weight=args.hop_flight_weight,
                 level_feet_flight_only=args.level_feet_flight_only,
                 walking_height=args.walking_height,
                 swing_weight=args.swing_weight, air_time_min=args.air_time_min,
                 air_time_max=args.air_time_max, air_time_weight=args.air_time_weight,
                 hop_flight_duration=args.hop_flight_duration, hop_crouch=args.hop_crouch,
                 swing_clearance=args.swing_clearance)
    if args.mirror_replay:
        from .mirror import build_maps
        cfg["mirror_maps"] = build_maps(env)
        config_path.write_text(json.dumps(cfg, indent=2))
        if run:
            run.config.update({"mirror_maps": cfg["mirror_maps"]}, allow_val_change=True)
    if not existing_run:
        write_manifest(out, cfg, env.cfg)
    else:
        capture_source(out)
    obs, _ = env.reset()
    agent = FastSAC(obs["actor"].shape[1], obs["critic"].shape[1],
                    env.action_manager.total_action_dim, cfg)
    step = 0
    if args.resume:
        previous = agent.load(args.resume, replay=not args.finetune,
                              restore_rng=not args.finetune)
        step = 0 if args.finetune else previous
    setup_seconds = time.monotonic() - setup_start
    checkpoints = out / "checkpoints"
    if not (checkpoints / "initial.pt").exists():
        agent.save(checkpoints / "initial.pt", step)
    (out / "config/dimensions.json").write_text(json.dumps({
        "actor": obs["actor"].shape[1], "critic": obs["critic"].shape[1],
        "actions": env.action_manager.total_action_dim,
        "joint_names": env.scene["robot"].joint_names,
        "setup_seconds": setup_seconds,
    }, indent=2))
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start = time.monotonic()
    initial_step = step
    returns = torch.zeros(args.num_envs, device="cuda")
    recent_returns, recent_lengths = [], []
    terminations = truncations_total = 0
    metrics = {}
    path = out / "metrics/training.jsonl"
    try:
        with path.open("a", buffering=1) as log:
            while step < args.steps and not stop:
                if time.monotonic() - start >= args.minutes * 60:
                    break
                with torch.no_grad():
                    action = agent.act(obs["actor"])
                    next_obs, final, reward, terminated, truncated, lengths, info = step_and_reset(env, action)
                    agent.add(obs, action, final, reward, terminated, truncated)
                    obs = next_obs
                    returns += reward
                    done = terminated | truncated
                    if done.any():
                        recent_returns.extend(returns[done].cpu().tolist())
                        recent_lengths.extend((lengths[done] * env.step_dt).cpu().tolist())
                        recent_returns, recent_lengths = recent_returns[-200:], recent_lengths[-200:]
                        returns[done] = 0
                    terminations += int(terminated.sum())
                    truncations_total += int((truncated & ~terminated).sum())
                if agent.buffer.ptr > 32:
                    metrics.update(agent.learn())
                step += 1
                if step % 50 == 0:
                    elapsed = time.monotonic() - start
                    robot = env.scene["robot"].data
                    row = {
                        "step": step, "transitions": step * args.num_envs,
                        "elapsed_seconds": previous_training_seconds + elapsed,
                        "session_elapsed_seconds": elapsed, "setup_seconds": setup_seconds,
                        "transitions_per_second": (step - initial_step) * args.num_envs / elapsed,
                        "reward_per_step": float(reward.mean()),
                        "episode_return": sum(recent_returns)/max(1, len(recent_returns)),
                        "episode_seconds": sum(recent_lengths)/max(1, len(recent_lengths)),
                        "terminations": terminations, "timeouts": truncations_total,
                        "forward_speed": float(robot.root_link_lin_vel_b[:, 0].mean()),
                        "target_speed": float(env.command_manager.get_command("twist")[:, 0].mean()),
                        "velocity_error": float((robot.root_link_lin_vel_b[:, :2] -
                            env.command_manager.get_command("twist")[:, :2]).abs().mean()),
                        "yaw_error": float((robot.root_link_ang_vel_b[:, 2] -
                            env.command_manager.get_command("twist")[:, 2]).abs().mean()),
                        **{k: float(v) for k, v in metrics.items()},
                        **{k: float(v) for k, v in info.items() if isinstance(v, (float, int, torch.Tensor))},
                    }
                    if not all(torch.isfinite(v).all() for v in obs.values()):
                        raise FloatingPointError("Nonfinite observations; stopping and preserving checkpoint")
                    log.write(json.dumps(row) + "\n")
                    if run:
                        run.log(row, step=step)
                    print(f"step={step} elapsed={elapsed:.0f}s "
                          f"episode={row['episode_seconds']:.1f}s "
                          f"speed={row['forward_speed']:.2f} reward={row['reward_per_step']:.3f}",
                          flush=True)
                if step % args.save_every == 0:
                    agent.save(checkpoints / f"step_{step:06d}.pt", step)
                    agent.save(checkpoints / "resume.pt", step, include_replay=True)
    finally:
        agent.save(checkpoints / "final.pt", step)
        agent.save(checkpoints / "resume.pt", step, include_replay=True)
        session_seconds = time.monotonic() - start
        summary = {"step": step, "training_seconds": previous_training_seconds + session_seconds,
                   "session_training_seconds": session_seconds,
                   "setup_seconds": setup_seconds, "stopped_by_signal": stop,
                   "hardware": torch.cuda.get_device_name(0)}
        (out / "metrics/timing.json").write_text(json.dumps(summary, indent=2))
        if run:
            run.summary.update(summary)
            run.finish()
        env.close()


if __name__ == "__main__":
    main()
