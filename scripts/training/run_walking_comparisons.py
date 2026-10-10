"""Fine-tune each walking variation from one parent; never chain variations."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.experiments import config_path


def environment_differences(parent, child, path=""):
    if isinstance(parent, dict) and isinstance(child, dict):
        return [difference for key in sorted(parent.keys() | child.keys())
                for difference in environment_differences(
                    parent.get(key), child.get(key), path + "/" + key)]
    return [] if parent == child else [
        {"path": path, "parent": parent, "variation": child}]


def comparison_record(parent, output, variant):
    parent_config = config_path(parent)
    child_config = config_path(output)
    config = json.loads(child_config.read_text())
    differences = environment_differences(
        json.loads(parent_config.with_name("environment.json").read_text()),
        json.loads(child_config.with_name("environment.json").read_text()))
    expected = {
        "control": [],
        "smoother": ["/rewards/action_rate_l2/weight"],
        "higher_steps": ["/rewards/foot_swing_height/params/target_height"],
    }[variant]
    with parent.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "parent_checkpoint": str(parent.resolve()), "parent_sha256": digest,
        "same_parent": Path(config["resume"]).resolve() == parent.resolve(),
        "finetune": config["finetune"], "steps": config["steps"], "seed": config["seed"],
        "resolved_environment_differences": differences,
        "expected_difference_paths": expected,
        "exact_expected_differences": [item["path"] for item in differences] == expected,
    }


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
        ("higher_steps", {"swing_height": 0.075}),
    ]:
        name = f"{args.prefix}_{variant}"
        settings = dict(config, **changes)
        pointer = ROOT / "artifacts" / f"{name}.path"
        pointer.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "bdxr_classroom.learning.train",
                   "--name", name, "--steps", "3000", "--minutes", "15",
                   "--result-file", str(pointer),
                   "--resume", str(parent), "--finetune", "--wandb-mode", args.wandb_mode]
        for key in ["num_envs", "seed", "batch_size", "updates", "buffer_size",
                    "actor_hidden", "critic_hidden", "gamma", "task_version",
                    "speed", "smoothness", "command_mode", "speed_min", "speed_max",
                    "yaw_max", "yaw_tracking", "yaw_std", "yaw_weight", "linear_weight",
                    "swing_height", "yaw_bias_weight", "lateral_bias_weight", "training_terrain_height",
                    "training_terrain_style", "training_start_span", "initial_jitter",
                    "reward_scale", "linear_tracking"]:
            if settings.get(key) is not None:
                command.extend(["--" + key.replace("_", "-"), str(settings[key])])
        command.extend(["--amp", config.get("amp", "off")])
        if config.get("compile"):
            command.append("--compile")
        if config.get("mirror_replay"):
            command.append("--mirror-replay")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        output = Path(pointer.read_text().strip())
        record = comparison_record(parent, output, variant)
        (output / "config/matched_comparison.json").write_text(
            json.dumps(record, indent=2) + "\n")
        if not record["same_parent"] or not record["exact_expected_differences"]:
            raise RuntimeError(f"{name}: unexpected comparison configuration; inspect its audit")
        subprocess.run([sys.executable, "-m", "bdxr_classroom.evaluation.command_evaluate",
                        str(output / "checkpoints/final.pt"), "--episodes", "5", "--seconds", "15"],
                       cwd=ROOT, env=env, check=True)
        # A failed policy is still a completed experiment; evaluate the other variants.
        checked = subprocess.run(
            [sys.executable, str(ROOT / "scripts/evaluation/check_command_results.py"),
             str(output / "evaluation/final_commands/summary.json")],
            cwd=ROOT, env=env)
        verdict_path = output / "evaluation/final_commands/selection_check.json"
        if checked.returncode not in (0, 1) or not verdict_path.is_file():
            raise RuntimeError(f"{name}: acceptance checker could not complete")
        verdict = json.loads(verdict_path.read_text())
        if verdict.get("passed") is not (checked.returncode == 0):
            raise RuntimeError(f"{name}: inconsistent acceptance verdict")


if __name__ == "__main__":
    main()
