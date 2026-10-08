"""Execute the complete notebook with short real training runs (local test)."""

import argparse
import hashlib
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="/tmp/bdxr-colab-check")
    p.add_argument("--kernel", default="bdxr-validation")
    p.add_argument("--game-port", type=int, default=18765)
    p.add_argument("--full", action="store_true", help="Run the full 512-environment lesson with W&B")
    p.add_argument("--output", type=Path, help="Save the executed notebook without replacing earlier evidence")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root / "notebooks/BDXR_Walking_101.ipynb", as_version=4)
    nb.metadata["validation"] = {
        "source_sha256": hashlib.sha256((root / "notebooks/BDXR_Walking_101.ipynb").read_bytes()).hexdigest(),
        "full_training": args.full, "work": args.work, "colab_tested": False,
        "scope": "Local notebook execution; not actual Google Colab.",
        "quality_gate_enforced": args.full,
    }
    if not args.full:
        nb.cells[0].source = (
            "**PLUMBING SMOKE TEST:** Short training runs are expected to fail policy-quality "
            "checks. Their failed verdicts are retained; recovery and the baseline stop gate "
            "are bypassed only in this executed test copy. This does not validate locomotion.\n\n"
            + nb.cells[0].source)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        source = source.replace(
            'WORK = Path("/content/bdxr_classroom") if Path("/content").exists() else Path.cwd() / "bdxr_classroom_lab"',
            f"WORK = Path({args.work!r})",
        )
        source = source.replace("NUM_ENVS = 1024 if gpu_gib >= 24 else 512",
                                f"NUM_ENVS = {512 if args.full else 32}")
        source = source.replace("BATCH_SIZE = 4096 if gpu_gib >= 24 else 2048",
                                f"BATCH_SIZE = {2048 if args.full else 256}")
        if not args.full:
            source = source.replace("NUM_ENVS = 512", "NUM_ENVS = 32")
            source = source.replace("BATCH_SIZE = 2048", "BATCH_SIZE = 256")
        if args.full:
            source = source.replace("USE_WANDB = False", "USE_WANDB = True")
        else:
            source = source.replace(
                'if not BASELINE_CHECK["passed"]:',
                'if False:  # Smoke harness only: retain the failed verdict, skip recovery/stop.')
            source = source.replace("TRAIN_STEPS = 8000", "TRAIN_STEPS = 60")
            source = source.replace("VARIATION_STEPS = 3000", "VARIATION_STEPS = 60")
            source = source.replace("TRAIN_HOPPING = True", "TRAIN_HOPPING = False")
            source = source.replace('"--steps", 6000', '"--steps", 60')
            source = source.replace('"--steps", 2000', '"--steps", 60')
            source = source.replace('"--episodes", 3', '"--episodes", 1')
            source = source.replace('"--episodes", 5', '"--episodes", 1')
            source = source.replace('"--seconds", 15', '"--seconds", 2')
            source = source.replace('"--compile", ', "")
        source = source.replace("GAME_PORT = 8765", f"GAME_PORT = {args.game_port}")
        cell.source = source
    nb.metadata.kernelspec.name = args.kernel
    client = NotebookClient(nb, timeout=1800 if args.full else 600, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(root)}})
    dest = args.output or root / ("artifacts/notebook_full_executed.ipynb" if args.full
                                 else "artifacts/notebook_smoke_executed.ipynb")
    dest.parent.mkdir(exist_ok=True)
    try:
        client.execute()
    finally:
        nbformat.write(nb, dest)
    print(f"All notebook cells executed successfully: {dest}")


if __name__ == "__main__":
    main()
