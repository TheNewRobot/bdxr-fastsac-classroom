"""Launch the real interactive controller and save its gameplay recordings."""

import json
import subprocess
import time
import urllib.request
from pathlib import Path


def game_state(port):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/api/state", timeout=5
    ) as response:
        return json.load(response)


def launch_game(
    lab, walking, hopping=None, port=8765, course="configs/classroom_course.json"
):
    if not walking:
        raise ValueError("Select a walking checkpoint before starting the game.")
    if lab.game_process is not None and lab.game_process.poll() is None:
        lab.game_process.terminate()
        lab.game_process.wait(timeout=30)
    command = [
        lab.python,
        "-u",
        "-m",
        "bdxr_classroom.game.server",
        "--walking",
        str(walking),
        "--port",
        str(port),
        *lab.presentation_arguments(game=True),
    ]
    if hopping:
        command += ["--hopping", str(hopping)]
    if course:
        course_path = lab.work / course
        if not course_path.is_file():
            raise FileNotFoundError(course_path)
        command += ["--course-config", str(course_path)]
    log_path = lab.work / "game_server.log"
    with log_path.open("w") as log:
        lab.game_process = subprocess.Popen(
            command,
            cwd=lab.work,
            env=lab.environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    for _ in range(120):
        if lab.game_process.poll() is not None:
            raise RuntimeError(log_path.read_text()[-4000:])
        try:
            state = game_state(port)
            if state.get("error"):
                raise RuntimeError(state["error"])
            if state.get("ready"):
                return state
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("Game initialization timed out; inspect game_server.log.")


def display_game(port):
    try:
        from google.colab import output

        output.serve_kernel_port_as_iframe(port, height=950)
    except ImportError:
        from IPython.display import IFrame, display

        display(IFrame(f"http://127.0.0.1:{port}", width="100%", height=950))


def pause_and_recordings(lab, port):
    state = game_state(port)
    if not state.get("paused", True):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/command",
            data=b'{"action":"pause"}',
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=5).close()
        for _ in range(100):
            state = game_state(port)
            if state.get("paused"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Pause the game before exporting.")
    folder = Path(state["experiment"])
    folder = folder if folder.is_absolute() else lab.work / folder
    return sorted((folder / "videos").glob("gameplay_*.mp4"))


def optional_recording(lab, port, enabled=False):
    """Leave the game untouched unless the student requests a recorded replay."""
    if not enabled:
        return None
    from IPython.display import Video, display

    recordings = pause_and_recordings(lab, port)
    if not recordings:
        print("Record a clip in the game, then rerun this cell.")
        return None
    latest = recordings[-1]
    display(Video(str(latest), embed=True, width=640))
    return latest
