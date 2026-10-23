"""Notebook plumbing for the standalone hopping lesson."""

import json
import socket
import subprocess
import time
from pathlib import Path

from bdxr_classroom.lesson.play import display_game, game_state
from bdxr_classroom.lesson.presentation import media_pair, table
from bdxr_classroom.evaluation.walking_checks import checkpoint_digest


def check_and_watch_hops(lab, run):
    """Use the same one-policy controller for the recording and interactive demo."""
    run = Path(run)
    checkpoint = run / "checkpoints/final.pt"
    output = run / "evaluation" / f"single_hops_{time.time_ns()}.json"
    started = time.monotonic()
    lab.run(
        "bdxr_classroom.hopping_demo.evaluate", checkpoint, "--output", output,
    )
    elapsed = time.monotonic() - started
    result = json.loads(output.read_text())
    result["activity_seconds"] = elapsed
    output.write_text(json.dumps(result, indent=2))
    video = Path(result["video"])
    if not video.is_absolute():
        video = lab.work / video
    demo_run = video.parent.parent
    if demo_run not in lab.runs:
        lab.runs.append(demo_run)
    media_pair([("Your requested hops", video, "Same policy and controller as the Jump button.")])
    table(
        [
            (
                row["request"], f"{100 * row['peak_clearance_m']:.1f}",
                f"{row['longest_flight_s']:.2f}", row["flight_events"],
                "Yes" if row["fell"] else "No", "Pass" if row["passed"] else "Review",
            )
            for row in result["attempts"]
        ],
        ["Jump", "Clearance (cm)", "Flight (s)", "Flights", "Fell?", "Result"],
    )
    print("PASS — three clear hops and landings." if result["passed"]
          else "REVIEW — this policy did not complete three clear hops and landings.")
    if not result["passed"]:
        print(result["message"])
    print(f"Recording and check, including startup: {elapsed:.0f}s.")
    print("Flat-ground trials do not establish reliable jumping on other terrain.")
    return result


def launch_hop_practice(lab, run, port=8766, assessment=None):
    """Start a separate server without touching the walking game's process."""
    checkpoint = Path(run) / "checkpoints/final.pt"
    if assessment is not None and (
        assessment.get("passed") is not True
        or Path(assessment["checkpoint"]).resolve() != checkpoint.resolve()
        or assessment.get("checkpoint_sha256") != checkpoint_digest(checkpoint)
    ):
        raise ValueError("Run the hopping check for these weights before launching.")
    previous = getattr(lab, "hopping_process", None)
    if previous is not None and previous.poll() is None:
        previous.terminate()
        previous.wait(timeout=30)
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(
                f"Port {port} is already in use. Choose another HOP_PORT "
                "and rerun this cell."
            )
    log_path = lab.work / "hopping_demo.log"
    with log_path.open("w") as log:
        lab.hopping_process = subprocess.Popen(
            [
                lab.python, "-u", "-m", "bdxr_classroom.hopping_demo.server",
                "--checkpoint", str(checkpoint), "--port", str(port),
            ],
            cwd=lab.work, env=lab.environment, stdout=log, stderr=subprocess.STDOUT,
        )
    for _ in range(240):
        if lab.hopping_process.poll() is not None:
            raise RuntimeError(log_path.read_text()[-4000:])
        try:
            state = game_state(port)
            if state.get("error"):
                raise RuntimeError(state["error"])
            if state.get("ready"):
                demo_run = Path(state["experiment"])
                if not demo_run.is_absolute():
                    demo_run = lab.work / demo_run
                if demo_run not in lab.runs:
                    lab.runs.append(demo_run)
                display_game(port)
                return state
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("Hopping demo timed out; inspect hopping_demo.log.")


def watch_hop_recording(lab, port=8766, enabled=False):
    """Replay only a closed clip; leave physics and recording state untouched."""
    if not enabled:
        return None
    state = game_state(port)
    if state.get("recording"):
        print("Turn Record off in the panel, then rerun this cell.")
        return None
    folder = Path(state["experiment"])
    if not folder.is_absolute():
        folder = lab.work / folder
    videos = sorted((folder / "videos").glob("hopping_*.mp4"))
    if not videos:
        print("Enable Record, click Jump, then turn Record off to keep a clip.")
        return None
    media_pair([("Your recording", videos[-1], "")])
    return videos[-1]
