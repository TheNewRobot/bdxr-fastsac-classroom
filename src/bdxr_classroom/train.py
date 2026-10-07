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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="walk")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--smoothness", type=float, default=1)
    p.add_argument("--task-version", choices=["v1", "v2", "hop"], default="v2")
    p.add_argument("--hop-height", type=float, default=0.04)
    p.add_argument("--hop-body-weight", type=float, default=0.0)
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
    cfg["hardware"] = torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    existing_run = (out / "config.json").exists()
    if existing_run and (not args.resume or args.finetune):
        raise FileExistsError(f"{out} already exists; choose a new --name or resume this same experiment")
    config_path = out / (f"resume_config_{time.time_ns()}.json" if existing_run else "config.json")
    config_path.write_text(json.dumps(cfg, indent=2))
    previous_timing = {}
    if existing_run and (out / "timing.json").exists():
        previous_timing = json.loads((out / "timing.json").read_text())
    previous_training_seconds = previous_timing.get("training_seconds", 0.0)
    wandb_resume = {}
    if existing_run and args.wandb_mode == "online" and (out / "wandb.json").exists():
        previous_wandb = json.loads((out / "wandb.json").read_text())
        if previous_wandb.get("url"):
            wandb_resume = {"id": previous_wandb["id"], "resume": "allow"}
    run = wandb.init(project=args.project, name=args.name, config=cfg, dir=str(out),
                     mode=args.wandb_mode, allow_val_change=True,
                     tags=["FastSAC", "BDX-R", "classroom", cfg["runtime_environment"], cfg["hardware"]],
                     **wandb_resume)
    if run:
        (out / "wandb.json").write_text(json.dumps({"url": run.url, "id": run.id}))
    setup_start = time.monotonic()
    env = make_env(args.num_envs, args.speed, args.smoothness, args.seed,
                 task_version=args.task_version, hop_height=args.hop_height,
                 hop_body_weight=args.hop_body_weight)
    obs, _ = env.reset()
    agent = FastSAC(obs["actor"].shape[1], obs["critic"].shape[1],
                    env.action_manager.total_action_dim, cfg)
    step = 0
    if args.resume:
        previous = agent.load(args.resume, replay=not args.finetune)
        step = 0 if args.finetune else previous
    setup_seconds = time.monotonic() - setup_start
    if not (out / "initial.pt").exists():
        agent.save(out / "initial.pt", step)
    (out / "dimensions.json").write_text(json.dumps({
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
    path = out / "metrics.jsonl"
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
                        "target_speed": args.speed,
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
                    agent.save(out / f"step_{step:06d}.pt", step)
                    agent.save(out / "resume.pt", step, include_replay=True)
    finally:
        agent.save(out / "final.pt", step)
        agent.save(out / "resume.pt", step, include_replay=True)
        session_seconds = time.monotonic() - start
        summary = {"step": step, "training_seconds": previous_training_seconds + session_seconds,
                   "session_training_seconds": session_seconds,
                   "setup_seconds": setup_seconds, "stopped_by_signal": stop,
                   "hardware": torch.cuda.get_device_name(0)}
        (out / "timing.json").write_text(json.dumps(summary, indent=2))
        if run:
            run.summary.update(summary)
            run.finish()
        env.close()


if __name__ == "__main__":
    main()
