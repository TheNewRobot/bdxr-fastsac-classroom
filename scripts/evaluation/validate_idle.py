"""Measure quiet standing and transitions without changing the deployed actor."""

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

from bdxr_classroom.evaluation.command_evaluate import checkpoint_hash, load_agent, set_command
from bdxr_classroom.simulation.environment import make_env


def stationary_start_indices(row):
    """A responsive parallel start must not hide another start's stall."""
    if "speed" not in row:
        return []
    return [
        index for index, (speed, yaw) in enumerate(zip(row["speed"], row["yaw"]))
        if (abs(row["command"][0]) >= .1 and abs(speed) < .02)
        or (abs(row["command"][1]) >= .2 and abs(yaw) < .02)
    ]


def assess(checkpoint, starts=3, reverse_speed=None):
    agent, config = load_agent(checkpoint, starts)
    kwargs = {k: v for k, v in config.items() if k in inspect.signature(make_env).parameters}
    kwargs.update(num_envs=starts, render=False)
    env = make_env(**kwargs)
    segments = [
        ("idle_start", 5, 0, 0),
        ("forward", 5, .3, 0),
        ("idle_after_forward", 5, 0, 0),
        ("backward", 4, config.get("speed_min", -.15) if reverse_speed is None else reverse_speed, 0),
        ("idle_after_backward", 5, 0, 0),
        ("turn_left", 4, 0, .4),
        ("idle_after_turn", 5, 0, 0),
        ("turn_right", 4, 0, -.4),
        ("idle_after_right_turn", 5, 0, 0),
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
        foot_ids, _ = env.scene["robot"].find_sites(["left_foot", "right_foot"])
        alive = torch.ones(starts, dtype=torch.bool, device=env.device)
        output = []
        failure = None
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
                        "speed_mae": (robot.root_link_lin_vel_b[:, 0] - speed).abs().cpu().numpy(),
                        "yaw_mae": (robot.root_link_ang_vel_b[:, 2] - yaw).abs().cpu().numpy(),
                        "joint_rms": robot.joint_vel[:, leg_ids].square().mean(-1).sqrt().cpu().numpy(),
                        "both_feet": contact.all(axis=1).astype(float),
                        "contact_spin_rms": (
                            robot.site_ang_vel_w[:, foot_ids, 2].square()
                            * torch.as_tensor(contact, device=env.device)
                        ).mean(-1).sqrt().cpu().numpy(),
                    })
                previous = contact
                if not bool(alive.all()):
                    failure = {
                        "segment": name, "segment_seconds": (step + 1) * env.step_dt,
                        "failed_starts": (~alive).nonzero().flatten().cpu().tolist(),
                        "termination_terms": {
                            term: env.termination_manager.get_term(term).cpu().tolist()
                            for term in env.termination_manager.active_terms
                        },
                    }
                    break
            row = {
                "segment": name, "command": [speed, yaw],
                "falls_so_far": int((~alive).sum()),
                "foot_liftoffs_after_settling": liftoffs.tolist(),
                **({key: np.mean([v[key] for v in values], axis=0).tolist() for key in values[0]} if values else {}),
            }
            output.append(row)
            print(json.dumps(row), flush=True)
            if failure:
                break
        stationary_starts = {
            row["segment"]: stationary_start_indices(row)
            for row in output if stationary_start_indices(row)
        }
        repeated_idle_steps = [
            row["segment"] for row in output
            if row["command"] == [0, 0]
            and max(row["foot_liftoffs_after_settling"], default=0) > 1
        ]
        return {
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash(checkpoint),
            "starts": starts, "seconds": time.monotonic()-started, "segments": output,
            "completed": failure is None, "failure": failure,
            "stationary_motion_segments": list(stationary_starts),
            "stationary_motion_starts": stationary_starts,
            "repeated_idle_step_segments": repeated_idle_steps,
            "quiet_scope": "At most one balance step per start after one second of settling.",
            "reverse_scope": "configured command minimum" if reverse_speed is None else "explicit stress command",
        }
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-speed", type=float)
    args = parser.parse_args()
    result = assess(args.checkpoint, reverse_speed=args.reverse_speed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(
        0 if result["completed"]
        and not result["stationary_motion_segments"]
        and not result["repeated_idle_step_segments"] else 1
    )
