"""Execute the authored notebook with real local training and explicit scope.

By default this is a short integration test, not a policy-quality evaluation.
The test copy retains failed verdicts and bypasses its classroom stop gate only
so the later cells and backup can be exercised. --full uses the actual budgets.
"""

import argparse
import ast
import hashlib
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


def replace_assignment(source, name, expression):
    """Replace a top-level assignment without depending on line wrapping."""
    lines = source.splitlines(keepends=True)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            lines[node.lineno - 1 : node.end_lineno] = [f"{name} = {expression}\n"]
            return "".join(lines)
    raise ValueError(f"Missing validation assignment: {name}")


def test_copy(
    notebook,
    work,
    reuse_env=None,
    full=False,
    game_port=18765,
    include_hopping=False,
    bundle_url=None,
    exercise_recovery=False,
):
    """Modify only runtime location/budgets in the executed validation copy."""
    if not full:
        notebook.cells[0].source = (
            "**LOCAL INTEGRATION TEST:** Short runs do not establish locomotion quality. "
            "Failed policy checks are retained; the classroom stop gate is bypassed only "
            "in this test copy. Actual Google Colab is untested.\n\n"
            + notebook.cells[0].source
        )
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if source.startswith("# @title Unpack the lesson files"):
            source = replace_assignment(source, "WORK", f"Path({str(work)!r})")
            if bundle_url:
                source = replace_assignment(source, "BUNDLE_URL", repr(bundle_url))
        if reuse_env and source.startswith("# @title Install pinned dependencies"):
            existing = Path(reuse_env).resolve()
            source = f"""# Local validation: reuse pinned dependencies; installation is not tested.
import os, json
from bdxr_classroom.lesson.presentation import read_json
existing = Path({str(existing)!r})
assert (existing / ".venv/bin/python").is_file()
assert read_json(existing / "configs/upstream_revisions.json") == read_json(WORK / "configs/upstream_revisions.json")
vendor = WORK / "vendor"
if not vendor.exists():
    vendor.symlink_to(existing / "vendor", target_is_directory=True)
PYTHON = str(existing / ".venv/bin/python")
runtime_env = dict(os.environ, PYTHONPATH=str(WORK / "src"), MUJOCO_GL="egl", BDXR_RUNTIME="local_notebook_reused_env")
gpu = json.loads(subprocess.check_output([PYTHON, "-c", "import torch,json; print(json.dumps(dict(name=torch.cuda.get_device_name(0))))"], cwd=WORK, env=runtime_env, text=True))
print("Reused pinned environment; GPU:", gpu)
"""
        if not full:
            if source.startswith("LEARNING_CONFIG ="):
                source += "\nLEARNING_CONFIG.update(num_envs=32, batch_size=256, steps=60, compile=False)\n"
            if source.startswith("EXPERIMENT ="):
                source += "\nPAIR_LEARNING.update(steps=60)\n"
            if exercise_recovery:
                source = source.replace('"steps": 3000', '"steps": 60')
                source = source.replace(
                    "and drift_only_failure(INITIAL_BASELINE_CHECK):",
                    "and True:  # Exercise routing only; short runs are not eligible policies.",
                )
                source = source.replace(
                    'if not BASELINE_CHECK["passed"]:\n    raise RuntimeError(',
                    "if False:  # Integration copy only; failed verdict retained.\n"
                    "    raise RuntimeError(",
                )
            else:
                source = source.replace(
                    'if not BASELINE_CHECK["passed"] and drift_only_failure(INITIAL_BASELINE_CHECK):',
                    "if False:  # Integration copy only; skip optional refinement.",
                )
                source = source.replace(
                    'if not BASELINE_CHECK["passed"]:',
                    "if False:  # Integration copy only; failed verdict retained.",
                )
            source = source.replace(
                "evaluate_walking(lab, BASELINE)",
                "evaluate_walking(lab, BASELINE, episodes=1, seconds=2)",
            )
            source = source.replace(
                "evaluate_walking(lab, CONTROL)",
                "evaluate_walking(lab, CONTROL, episodes=1, seconds=2)",
            )
            source = source.replace(
                "evaluate_walking(lab, VARIATION)",
                "evaluate_walking(lab, VARIATION, episodes=1, seconds=2)",
            )
            source = source.replace(
                "walking = WALK_CHECKPOINT or SELECTED_WALK_CHECKPOINT",
                'walking = WALK_CHECKPOINT or str(BASELINE / "checkpoints/final.pt")  # Exploratory smoke policy',
            )
            if include_hopping:
                source = source.replace("TRAIN_HOPPING = False", "TRAIN_HOPPING = True")
                source = source.replace(
                    "for stage in CURRICULUM:\n",
                    "for stage in CURRICULUM:\n"
                    '        stage = {**stage, "learning": {**stage["learning"], "steps": 60}}\n',
                )
                source = source.replace(
                    "select_hopper(lab, HOP_STAGES, SELECTED_WALK_CHECKPOINT)",
                    "select_hopper(lab, HOP_STAGES, SELECTED_WALK_CHECKPOINT, episodes=1, seconds=2)",
                )
        else:
            source = source.replace("USE_WANDB = False", "USE_WANDB = True")
            source = source.replace("TRAIN_HOPPING = False", "TRAIN_HOPPING = True")
        source = source.replace("GAME_PORT = 8765", f"GAME_PORT = {game_port}")
        cell.source = source
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default="/tmp/bdxr-lesson-refactor-check")
    parser.add_argument("--kernel", default="bdxr-validation")
    parser.add_argument("--game-port", type=int, default=18765)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--exercise-recovery",
        action="store_true",
        help="Exercise both recovery branches with short integration budgets.",
    )
    parser.add_argument(
        "--include-hopping",
        action="store_true",
        help="Exercise all optional curriculum stages with short integration budgets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/notebook_refactor_executed.ipynb",
    )
    parser.add_argument("--reuse-env", type=Path)
    parser.add_argument(
        "--bundle-url",
        help="Explicit prepublication test URL; hash verification is still required.",
    )
    args = parser.parse_args()
    path = ROOT / "notebooks/BDXR_Walking_101.ipynb"
    notebook = nbformat.read(path, as_version=4)
    notebook.metadata["validation"] = {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "full_training": args.full,
        "colab_tested": False,
        "work": args.work,
        "dependency_installation_tested": args.reuse_env is None,
        "bundle_url_override": args.bundle_url,
        "optional_hopping_enabled": args.full or args.include_hopping,
        "recovery_branches_exercised": args.exercise_recovery and not args.full,
        "scope": "Local notebook execution; short integration unless full_training is true.",
    }
    test_copy(
        notebook,
        args.work,
        args.reuse_env,
        args.full,
        args.game_port,
        args.include_hopping,
        args.bundle_url,
        args.exercise_recovery,
    )
    notebook.metadata.kernelspec.name = args.kernel
    client = NotebookClient(
        notebook,
        timeout=3600 if args.full else 600,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(ROOT)}},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, args.output)
    print("All notebook cells executed:", args.output)


if __name__ == "__main__":
    main()
