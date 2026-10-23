"""Measure quiet standing and transitions without changing the deployed actor."""

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

from bdxr_classroom.evaluation.command_evaluate import load_agent, set_command
from bdxr_classroom.simulation.environment import make_env


def assess(checkpoint, starts=3):
    agent, config = load_agent(checkpoint, starts)
    kwargs = {k: v for k, v in config.items() if k in inspect.signature(make_env).parameters}
    kwargs.update(num_envs=starts, render=False)
    env = make_env(**kwargs)
    segments = [
        ("idle_start", 5, 0, 0),
        ("forward", 5, .3, 0),
        ("idle_after_forward", 5, 0, 0),
        ("backward", 4, -.2, 0),
        ("idle_after_backward", 5, 0, 0),
        ("turn_left", 4, 0, .4),
        ("idle_after_turn", 5, 0, 0),
        ("restart", 4, .3, 0),
    ]
    started = time.monotonic()
    try:
        env.cfg.episode_length_s = 100
        env.reset(seed=1701)
        robot = env.scene["robot"].data
        leg_ids, _ = env.scene["robot"].find_joints(
            [".*_Hip_Yaw", ".*_Hip_Roll", ".*_Hip_Pitch", ".*_Knee", ".*_Ankle"]
        )
        alive = torch.ones(starts, dtype=torch.bool, device=env.device)
        output = []
        for name, seconds, speed, yaw in segments:
            values, liftoffs = [], np.zeros(starts, dtype=int)
            previous = None
            for step in range(round(seconds / env.step_dt)):
                obs = set_command(env, speed, yaw)
                with torch.no_grad():
                    _, _, terminated, truncated, _ = env.step(
                        agent.act(obs["actor"], deterministic=True)
                    )
                alive &= ~(terminated | truncated)
                contact = (env.scene["feet_ground_contact"].data.found > 0).reshape(starts, 2).cpu().numpy()
                if step * env.step_dt >= 1:
                    if previous is not None:
                        liftoffs += (previous & ~contact).sum(axis=1)
                    values.append({
                        "speed": robot.root_link_lin_vel_b[:, 0].cpu().numpy(),
                        "yaw": robot.root_link_ang_vel_b[:, 2].cpu().numpy(),
                        "joint_rms": robot.joint_vel[:, leg_ids].square().mean(-1).sqrt().cpu().numpy(),
                        "both_feet": contact.all(axis=1).astype(float),
                    })
                previous = contact
            row = {
                "segment": name, "command": [speed, yaw],
                "falls_so_far": int((~alive).sum()),
                "foot_liftoffs_after_settling": liftoffs.tolist(),
                **{key: np.mean([v[key] for v in values], axis=0).tolist() for key in values[0]},
            }
            output.append(row)
            print(json.dumps(row), flush=True)
        return {"checkpoint": str(checkpoint), "starts": starts, "seconds": time.monotonic()-started, "segments": output}
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
