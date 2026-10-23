"""Record three button-triggered hops and report every attempt."""

import argparse
import json
import time
from pathlib import Path

from bdxr_classroom.hopping_demo.session import HoppingSession


def assess(session, attempts=3):
    started = time.monotonic()
    session.set_recording(True)
    for _ in range(round(10 / session.env.step_dt)):
        session.step()
        if session.state in {"ready", "failed"}:
            break
    for _ in range(attempts):
        if not session.jump():
            break
        print(f"Progress: jump {len(session.results) + 1}/{attempts}", flush=True)
        while session.state not in {"ready", "failed"}:
            session.step()
        if session.state == "failed":
            break
        # Continue the same actor and physics after landing; do not fake a stop.
        contact = session.env.scene["feet_ground_contact"].data.found > 0
        previous_contact = contact.clone()
        idle_lifts = 0
        for _ in range(round(2 / session.env.step_dt)):
            session.step()
            contact = session.env.scene["feet_ground_contact"].data.found > 0
            idle_lifts += int((previous_contact & ~contact).sum())
            previous_contact = contact.clone()
            if session.state == "failed":
                break
        session.results[-1]["idle_foot_lifts"] = idle_lifts
        if idle_lifts > 1 or session.state == "failed":
            session.results[-1]["passed"] = False
            session.results[-1]["reason"] = "Did not maintain a quiet stance after landing."
    session.set_recording(False)
    result = {
        "checkpoint": str(session.checkpoint), "checkpoint_sha256": session.digest,
        "attempts_requested": attempts, "attempts": session.results,
        "final_state": session.state, "message": session.message,
        "passed": (
            len(session.results) == attempts
            and all(row["passed"] for row in session.results)
            and session.state != "failed"
        ),
        "wall_seconds": time.monotonic() - started,
        "video": str(session.output / "videos/hopping_001.mp4"),
        "scope": "Three clicks with one hopper, on flat ground with full-link collisions. "
                 "Physics continues for two seconds after each landing; at most one "
                 "balance foot lift is allowed. Not a robustness guarantee.",
    }
    (session.output / "evaluation/hops.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", default="experiments")
    parser.add_argument("--seed", type=int, default=263)
    args = parser.parse_args()
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    started = time.monotonic()
    session = HoppingSession(args.checkpoint, args.run_root, seed=args.seed)
    try:
        result = assess(session)
        result["simulation_and_recording_seconds"] = result["wall_seconds"]
        result["wall_seconds"] = time.monotonic() - started
        (session.output / "evaluation/hops.json").write_text(json.dumps(result, indent=2))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
