"""Measure full-collision game transitions; this is not a course-completion test."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bdxr_classroom.game.session import GameSession


def motion_command_reached(row):
    """Reject stalls or reversed motion; detailed tracking is assessed separately."""
    means = row.get("means", {})
    for channel in ("speed", "yaw"):
        requested = row["requested"][channel]
        if requested:
            actual = means.get(channel, 0)
            if actual * np.sign(requested) < abs(requested) * .5:
                return False
    return True


def assess(game, include_hopping=False):
    phases = [
        ("idle", 4, 0, 0, "walk"),
        ("forward", 3, .3, 0, "walk"),
        ("stop_forward", 5, 0, 0, "walk"),
        ("backward", 3, -.15, 0, "walk"),
        ("stop_backward", 5, 0, 0, "walk"),
        ("left", 3, 0, .4, "walk"),
        ("stop_left", 5, 0, 0, "walk"),
        ("right", 3, 0, -.4, "walk"),
        ("stop_right", 5, 0, 0, "walk"),
        ("restart", 3, .3, 0, "walk"),
    ]
    if include_hopping:
        phases.extend([
            ("hop", 4, .15, 0, "hop"),
            ("return_to_walk", 5, .15, 0, "walk"),
        ])
    phases.append(("final_stop", 5, 0, 0, "walk"))
    robot = game.env.scene["robot"]
    leg_ids, _ = robot.find_joints(
        [".*_Hip_Yaw", ".*_Hip_Roll", ".*_Hip_Pitch", ".*_Knee", ".*_Ankle"]
    )
    body_ids, _ = robot.find_bodies(["Head_Base_Plate"])
    head_body = body_ids[0]
    started = time.monotonic()
    rows = []
    for name, duration, speed, yaw, mode in phases:
        game.command(speed, yaw, mode)
        previous_contact = None
        liftoffs = 0
        samples = []
        for step in range(round(duration / game.env.step_dt)):
            game.step()
            contact = (
                game.env.scene["feet_ground_contact"].data.found[0] > 0
            ).flatten().cpu().numpy()
            if step * game.env.step_dt >= 1:
                if previous_contact is not None:
                    liftoffs += int((previous_contact & ~contact).sum())
                data = robot.data
                quaternion = data.body_link_quat_w[0, head_body].cpu().numpy()
                w, x, y, z = quaternion
                gaze = np.degrees(np.arcsin(np.clip(2 * (x*z - w*y), -1, 1)))
                status = game.status()
                samples.append({
                    "speed": status["actual_speed"],
                    "yaw": status["actual_yaw"],
                    "leg_velocity_rms": float(
                        data.joint_vel[0, leg_ids].square().mean().sqrt()
                    ),
                    "both_feet": float(contact.all()),
                    "gaze_degrees": float(gaze),
                    "head_relative_height": float(
                        data.body_link_pos_w[0, head_body, 2]
                        - data.root_link_pos_w[0, 2]
                    ),
                })
            previous_contact = contact
            if game.score.falls:
                break
        quiet = speed == yaw == 0 and mode == "walk"
        row = {
            "phase": name,
            "requested": {"speed": speed, "yaw": yaw, "mode": mode},
            "actual_mode": game.stage,
            "quiet_requested": quiet,
            "foot_liftoffs_after_settling": liftoffs,
            "falls": game.score.falls,
            "sim_seconds": game.score.elapsed,
        }
        if samples:
            row["means"] = {
                key: float(np.mean([sample[key] for sample in samples]))
                for key in samples[0]
            }
            row["gaze_range_degrees"] = [
                min(s["gaze_degrees"] for s in samples),
                max(s["gaze_degrees"] for s in samples),
            ]
            row["head_relative_height_span"] = float(np.ptp([
                s["head_relative_height"] for s in samples
            ]))
        rows.append(row)
        print(json.dumps(row), flush=True)
        if game.score.falls:
            break
    game.save_summary()
    return {
        "experiment": str(game.output),
        "wall_seconds": time.monotonic() - started,
        "completed_sequence": len(rows) == len(phases) and game.score.falls == 0,
        "requested_modes_reached": all(
            row["actual_mode"] == row["requested"]["mode"] for row in rows
        ),
        "motion_commands_reached": all(motion_command_reached(row) for row in rows),
        "quiet_phases_without_repeated_steps": all(
            row["foot_liftoffs_after_settling"] <= 1
            for row in rows if row["quiet_requested"]
        ),
        "scope": (
            "Fixed game commands near the course start, full demo collisions. "
            "One balance step per settled idle phase is allowed. "
            "Commanded motion must reach half the requested mean speed/rate "
            "in the correct direction; this rejects stalls, not tracking errors. "
            "This does not establish complete course traversal or robustness."
        ),
        "phases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walking", type=Path)
    parser.add_argument("--hopping", type=Path)
    parser.add_argument("--head-config", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--seed", type=int, default=263)
    parser.add_argument("--name", default="game_transitions")
    parser.add_argument("--run-root", default="experiments/candidates/game_validation")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    course = json.loads(Path("configs/walking_course.json").read_text())
    if args.hopping:
        course["walking_only"] = False
    game = GameSession(
        args.walking,
        args.hopping,
        course=course,
        seed=args.seed,
        render=args.record,
        record=args.record,
        name=args.name,
        run_root=args.run_root,
        head_config=json.loads(args.head_config.read_text()) if args.head_config else None,
        operator="fixed_transition_test_no_navigation",
    )
    try:
        result = assess(game, include_hopping=bool(args.hopping))
        text = json.dumps(result, indent=2) + "\n"
        (game.output / "evaluation/transitions.json").write_text(text)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text)
    finally:
        game.close()
    raise SystemExit(
        0 if result["completed_sequence"]
        and result["requested_modes_reached"]
        and result["motion_commands_reached"]
        and result["quiet_phases_without_repeated_steps"] else 1
    )


if __name__ == "__main__":
    main()
