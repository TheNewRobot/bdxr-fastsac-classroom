"""Notebook setup plumbing; teaching configuration stays in the lesson."""

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .process import run_logged


def require_colab_gpu():
    """Require a GPU runtime without rejecting a device by its model name."""
    if not Path("/content").exists():
        return  # Local development may use other GPUs; report their actual timings.
    instruction = (
        "This classroom lesson needs a GPU runtime. Select Runtime → Change "
        "runtime type → GPU, then rerun setup."
    )
    try:
        names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(instruction) from error
    if not any(name.strip() for name in names):
        raise RuntimeError(f"{instruction} Detected: {', '.join(names)}.")
    print(f"Detected GPU: {', '.join(names)}.")
    print("Training time depends on this GPU; inspect the completed step count.")


def ensure_widgets(work):
    """Install the standard widget package in the notebook kernel if missing."""
    if importlib.util.find_spec("ipywidgets") is not None:
        return
    uv = shutil.which("uv")
    command = (
        [uv, "pip", "install", "--python", sys.executable]
        if uv
        else [sys.executable, "-m", "pip", "install"]
    )
    run_logged(
        command + ["ipywidgets==8.1.7"],
        cwd=work,
        env=dict(os.environ),
        log_path=Path(work) / "widget_setup.log",
        timeout=180,
        label="Installing notebook widgets",
    )
    importlib.invalidate_caches()


def install_runtime(work):
    """Install the pinned simulator and return its Python and environment."""
    work = Path(work)
    if not (work / "scripts/setup/setup.py").is_file():
        raise RuntimeError("Run 'Unpack the lesson files' before installing.")
    require_colab_gpu()
    started = time.monotonic()
    run_logged(
        [sys.executable, "-u", "scripts/setup/setup.py"],
        cwd=work,
        log_path=work / "setup.log",
        timeout=1800,
        label="Installing pinned dependencies",
        show=lambda line: line.startswith(
            ("Setup", "GPU:", "MuJoCo:", "Lesson fonts:", "Runtime details")
        ),
    )
    ensure_widgets(work)
    python = str(work / ".venv/bin/python")
    environment = dict(
        os.environ,
        PYTHONPATH=str(work / "src"),
        MUJOCO_GL="egl",
        BDXR_RUNTIME="colab" if Path("/content").exists() else "local_notebook",
    )
    gpu = json.loads(
        subprocess.check_output(
            [
                python,
                "-c",
                (
                    "import torch,json; print(json.dumps(dict("
                    "name=torch.cuda.get_device_name(0),"
                    "gib=torch.cuda.get_device_properties(0).total_memory/2**30)))"
                ),
            ],
            cwd=work,
            env=environment,
            text=True,
            timeout=60,
        )
    )
    print("GPU:", gpu)
    print("Setup time:", round(time.monotonic() - started), "seconds")
    return python, environment
