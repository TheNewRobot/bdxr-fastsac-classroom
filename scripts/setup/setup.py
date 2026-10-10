"""Create an isolated Python 3.12 environment and fetch pinned upstream assets."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    uv = shutil.which("uv")
    if not uv:
        run(sys.executable, "-m", "pip", "install", "uv")
        uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv installation did not produce an executable")
    pins = json.loads((ROOT / "configs/upstream_revisions.json").read_text())
    for folder, url, revision in [
        ("bdxr", "https://github.com/BDX-R/BDX-R-MjLab.git", pins["bdxr"]),
        ("holosoma", "https://github.com/amazon-far/holosoma.git", pins["holosoma"]),
    ]:
        dest = ROOT / "vendor" / folder
        if not dest.exists():
            dest.mkdir(parents=True)
            run("git", "-C", dest, "init")
            run("git", "-C", dest, "remote", "add", "origin", url)
            if folder == "holosoma":
                # We need the learner source, not the other humanoid meshes.
                run("git", "-C", dest, "config", "remote.origin.promisor", "true")
                run("git", "-C", dest, "config", "remote.origin.partialclonefilter", "blob:none")
                run("git", "-C", dest, "sparse-checkout", "init", "--cone")
                run("git", "-C", dest, "sparse-checkout", "set",
                    "src/holosoma/holosoma/agents/fast_sac",
                    "src/holosoma/holosoma/config_types")
                run("git", "-C", dest, "fetch", "--filter=blob:none", "--depth", "1", "origin", revision)
            else:
                run("git", "-C", dest, "fetch", "--depth", "1", "origin", revision)
            run("git", "-C", dest, "checkout", "--detach", "FETCH_HEAD")
        actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"{dest} has revision {actual}; expected {revision}")
    if not (ROOT / ".venv/bin/python").exists():
        run(uv, "venv", "--python", "3.12", ROOT / ".venv")
    python = ROOT / ".venv/bin/python"
    run(uv, "pip", "install", "--python", python, "torch==2.8.0",
        "--index-url", "https://download.pytorch.org/whl/cu128")
    lock = ROOT / "requirements.lock.txt"
    run(uv, "pip", "install", "--python", python, "-r",
        lock if lock.exists() else ROOT / "requirements.txt",
        "--index-url", "https://pypi.org/simple")
    run(uv, "pip", "install", "--python", python, "--no-deps",
        f"mjlab @ git+https://github.com/mujocolab/mjlab@{pins['mjlab']}",
        "-e", ROOT / "vendor/bdxr")
    run(python, "-c", "import torch,mujoco,warp; "
        "print('GPU:', torch.cuda.get_device_name(0)); "
        "print('MuJoCo:', mujoco.__version__, 'Warp:', warp.__version__)")


if __name__ == "__main__":
    main()
