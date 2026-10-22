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
