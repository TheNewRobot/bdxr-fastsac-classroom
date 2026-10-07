"""Package a measured evaluation and compact policy for the notebook gallery."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.export_policy import export_policy
from bdxr_classroom.plots import plot_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("evaluation", type=Path)
    args = p.parse_args()
    out = ROOT / "examples" / args.name
    out.mkdir(parents=True, exist_ok=True)
    export_policy(args.checkpoint, out / "policy.pt")
    for filename in ["evaluation.json", "motion.png"]:
        shutil.copy(args.evaluation / filename, out / filename)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i",
                    str(args.evaluation / "rollout.mp4"), "-vf", "scale=720:480",
                    "-c:v", "libx264", "-crf", "25", "-preset", "medium",
                    "-an", "-y", str(out / "rollout.mp4")], check=True)
    evaluation = json.loads((out / "evaluation.json").read_text())
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    rows = [json.loads(s) for s in (args.checkpoint.parent / "metrics.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["step"] <= evaluation["training_step"]]
    if rows:
        (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        plot_run(args.checkpoint.parent, evaluation["training_step"], out / "learning_curves.png")
    timing_path = args.checkpoint.parent / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    provenance = {
        "checkpoint": str(args.checkpoint), "training_origin": evaluation["label"],
        "hardware": config.get("hardware", timing.get("hardware", "NVIDIA L40S")),
        "training_seconds": rows[-1]["elapsed_seconds"] if rows else None,
        "setup_seconds": rows[-1]["setup_seconds"] if rows else None,
        "cache_state": "warm simulator cache",
        "timing_note": "Measured wall time; other experiments sometimes shared this GPU.",
        "algorithm": "FastSAC classroom adapter using Holosoma components",
        "physics": "MuJoCo Warp", "colab_tested": False,
        "parent_checkpoint": config.get("resume"), "config": config,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(out)


if __name__ == "__main__":
    main()
