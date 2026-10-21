"""Record versions and GPU details without collecting environment secrets."""

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "torch",
    "mujoco",
    "warp-lang",
    "mjlab",
    "bdx-r-mjlab",
    "numpy",
    "tensordict",
)


def main():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable"
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "cuda": torch.version.cuda,
        "driver": driver,
        "upstream_revisions": json.loads(
            (ROOT / "configs/upstream_revisions.json").read_text()
        ),
    }
    (ROOT / "runtime_info.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Runtime details saved: {ROOT / 'runtime_info.json'}", flush=True)


if __name__ == "__main__":
    main()
