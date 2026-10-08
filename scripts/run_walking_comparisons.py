"""Fine-tune each walking variation from one parent; never chain variations."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.experiments import config_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path, help="Full training checkpoint, not an inference export")
    parser.add_argument("--prefix", default="comparison")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()
    parent = args.parent.resolve()
    config = json.loads(config_path(parent).read_text())
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MUJOCO_GL="egl")
    for variant, changes in [
        ("control", {}),
        ("smoother", {"smoothness": 3 * config["smoothness"]}),
        ("higher_steps", {"swing_height": 0.09}),
    ]:
        name = f"{args.prefix}_{variant}"
        settings = dict(config, **changes)
        pointer = ROOT / "artifacts" / f"{name}.path"
        pointer.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "bdxr_classroom.train",
                   "--name", name, "--steps", "3000", "--minutes", "15",
                   "--result-file", str(pointer),
                   "--resume", str(parent), "--finetune", "--wandb-mode", args.wandb_mode]
        for key in ["num_envs", "seed", "batch_size", "updates", "buffer_size",
                    "actor_hidden", "critic_hidden", "gamma", "task_version",
                    "speed", "smoothness", "command_mode", "speed_min", "speed_max",
                    "yaw_max", "yaw_tracking", "yaw_std", "yaw_weight", "linear_weight",
                    "swing_height", "yaw_bias_weight", "training_terrain_height",
                    "training_terrain_style"]:
            if settings.get(key) is not None:
                command.extend(["--" + key.replace("_", "-"), str(settings[key])])
        command.extend(["--amp", config.get("amp", "off")])
        if config.get("compile"):
            command.append("--compile")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        output = Path(pointer.read_text().strip())
        subprocess.run([sys.executable, "-m", "bdxr_classroom.command_evaluate",
                        str(output / "checkpoints/final.pt"), "--episodes", "5", "--seconds", "15"],
                       cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
