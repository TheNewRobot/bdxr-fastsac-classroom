"""Fine-tune each walking variation from one parent; never chain variations."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path, help="Full training checkpoint, not an inference export")
    parser.add_argument("--prefix", default="comparison")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    parent = args.parent.resolve()
    config = json.loads((parent.parent / "config.json").read_text())
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MUJOCO_GL="egl")
    for variant, speed, smoothness in [
        ("control", config["speed"], config["smoothness"]),
        ("slower", max(0.05, config["speed"] - 0.15), config["smoothness"]),
        ("faster", config["speed"] + 0.15, config["smoothness"]),
        ("smoother", config["speed"], 3 * config["smoothness"]),
    ]:
        name = f"{args.prefix}_{variant}"
        output = ROOT / "runs" / name
        if output.exists():
            raise FileExistsError(f"{output} exists. Resume it explicitly or choose a new prefix.")
        command = [sys.executable, "-m", "bdxr_classroom.train",
                   "--name", name, "--steps", "2000", "--minutes", "15",
                   "--speed", str(speed), "--smoothness", str(smoothness),
                   "--resume", str(parent), "--finetune", "--wandb-mode", args.wandb_mode]
        for key in ["num_envs", "seed", "batch_size", "updates", "buffer_size",
                    "actor_hidden", "critic_hidden", "gamma", "task_version"]:
            command.extend(["--" + key.replace("_", "-"), str(config[key])])
        command.extend(["--amp", config.get("amp", "off")])
        if config.get("compile"):
            command.append("--compile")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, "-m", "bdxr_classroom.evaluate",
                        str(output / "final.pt"), "--episodes", "5", "--seconds", "15"],
                       cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
