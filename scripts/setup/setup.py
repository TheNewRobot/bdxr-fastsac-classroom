"""Create an isolated Python 3.12 environment and fetch pinned upstream assets."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*args, timeout=600, attempts=2):
    """Retry transient downloads; every command has a finite time budget."""
    command = list(map(str, args))
    for attempt in range(1, attempts + 1):
        print(f"Setup [{attempt}/{attempts}]: {' '.join(command)}", flush=True)
        try:
            subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                timeout=timeout,
                env=dict(os.environ, GIT_TERMINAL_PROMPT="0", PIP_NO_INPUT="1"),
            )
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == attempts:
                raise
            print("Setup step failed; retrying once…", flush=True)
            time.sleep(2)


def checkout(folder, url, revision):
    """Complete an interrupted clone without treating its directory as success."""
    dest = ROOT / "vendor" / folder
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").is_dir():
        run("git", "-C", dest, "init")
        run("git", "-C", dest, "remote", "add", "origin", url)
    current = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if current.returncode == 0 and current.stdout.strip() == revision:
        print(
            f"Setup: {folder} already at pinned revision {revision[:12]}.", flush=True
        )
        return
    changed = subprocess.check_output(
        ["git", "-C", str(dest), "status", "--porcelain"],
        text=True,
        timeout=30,
    )
    if changed.strip():
        raise RuntimeError(f"{dest} contains local changes; use a fresh WORK folder.")
    # A failed fetch can leave a valid empty Git repository. Repeat the fetch.
    remote = subprocess.run(
        ["git", "-C", str(dest), "remote", "get-url", "origin"],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    run(
        "git",
        "-C",
        dest,
        "remote",
        "set-url" if remote.returncode == 0 else "add",
        "origin",
        url,
    )
    options = []
    if folder == "holosoma":
        run("git", "-C", dest, "sparse-checkout", "init", "--cone")
        run(
            "git",
            "-C",
            dest,
            "sparse-checkout",
            "set",
            "src/holosoma/holosoma/agents/fast_sac",
            "src/holosoma/holosoma/config_types",
        )
        options = ["--filter=blob:none"]
    run("git", "-C", dest, "fetch", *options, "--depth", "1", "origin", revision)
    run("git", "-C", dest, "checkout", "--detach", revision)
    actual = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()
    if actual != revision:
        raise RuntimeError(f"{dest} has revision {actual}; expected {revision}")


def main():
    uv = shutil.which("uv")
    if not uv:
        run(sys.executable, "-m", "pip", "install", "uv==0.12.18")
        uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv installation did not produce an executable")
    pins = json.loads((ROOT / "configs/upstream_revisions.json").read_text())
    for folder, url, revision in [
        ("bdxr", "https://github.com/BDX-R/BDX-R-MjLab.git", pins["bdxr"]),
        ("holosoma", "https://github.com/amazon-far/holosoma.git", pins["holosoma"]),
    ]:
        checkout(folder, url, revision)
    if not (ROOT / ".venv/bin/python").exists():
        run(uv, "venv", "--python", "3.12", ROOT / ".venv")
    python = ROOT / ".venv/bin/python"
    run(
        uv,
        "pip",
        "install",
        "--python",
        python,
        "torch==2.8.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    )
    lock = ROOT / "requirements.lock.txt"
    run(
        uv,
        "pip",
        "install",
        "--python",
        python,
        "-r",
        lock if lock.exists() else ROOT / "requirements.txt",
        "--index-url",
        "https://pypi.org/simple",
    )
    run(
        uv,
        "pip",
        "install",
        "--python",
        python,
        "--no-deps",
        f"mjlab @ git+https://github.com/mujocolab/mjlab@{pins['mjlab']}",
        "-e",
        ROOT / "vendor/bdxr",
    )
    run(
        python,
        "-c",
        "import torch,mujoco,warp; "
        "print('GPU:', torch.cuda.get_device_name(0)); "
        "print('MuJoCo:', mujoco.__version__, 'Warp:', warp.__version__)",
    )
    run(
        python,
        "-c",
        "import sys; sys.path.insert(0, 'src'); "
        "from bdxr_classroom.lesson.kinematics import lesson_font; "
        "lesson_font(19); lesson_font(17, bold=True); print('Lesson fonts: ready')",
    )
    run(python, ROOT / "scripts/setup/runtime_info.py")


if __name__ == "__main__":
    main()
