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
from bdxr_classroom.experiments import experiment_root, config_path, metrics_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("evaluation", type=Path)
    p.add_argument("--case", default="forward", help="Case to show from a command-suite evaluation")
    p.add_argument("--caption", default="Selected recorded experiment; inspect its evaluation before drawing conclusions.")
    args = p.parse_args()
    out = ROOT / "assets/lesson" / args.name
    out.mkdir(parents=True, exist_ok=True)
    run = experiment_root(args.checkpoint)
    exports = run / "exports" / args.name
    exports.mkdir(parents=True, exist_ok=True)
    export_policy(args.checkpoint, exports / "policy.pt")
    if (args.evaluation / "summary.json").exists():
        summary = json.loads((args.evaluation / "summary.json").read_text())
        if args.case not in summary["cases"]:
            raise ValueError(f"Missing measured case: {args.case}")
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        evaluation = {"training_step": checkpoint["step"],
                      "label": "Recorded command evaluation; see parent lineage",
                      "video_path": run / "videos" / args.evaluation.name / f"{args.case}.mp4"}
        motion = args.evaluation / f"{args.case}.png"
    else:
        evaluation = json.loads((args.evaluation / "evaluation.json").read_text())
        motion = args.evaluation / "motion.png"
    shutil.copy(motion, out / "motion.png")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i",
                    str(evaluation.get("video_path") or args.evaluation / "rollout.mp4"), "-vf", "scale=720:480",
                    "-c:v", "libx264", "-crf", "25", "-preset", "medium",
                    "-an", "-y", str(out / "rollout.mp4")], check=True)
    config = json.loads(config_path(run).read_text())
    rows = [json.loads(s) for s in metrics_path(run).read_text().splitlines()]
    rows = [r for r in rows if r["step"] <= evaluation["training_step"]]
    if rows:
        plot_run(run, evaluation["training_step"], out / "learning_curves.png")
    timing_path = run / "metrics/timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    provenance = {
        "checkpoint": str(args.checkpoint), "training_origin": evaluation["label"],
        "evaluation": str(args.evaluation), "case": args.case,
        "hardware": config.get("hardware", timing.get("hardware", "NVIDIA L40S")),
        "training_seconds": rows[-1]["elapsed_seconds"] if rows else None,
        "setup_seconds": rows[-1]["setup_seconds"] if rows else None,
        "cache_state": "warm simulator cache",
        "timing_note": "Measured wall time; other experiments sometimes shared this GPU.",
        "algorithm": "FastSAC classroom adapter using Holosoma components",
        "physics": "MuJoCo Warp", "colab_tested": False,
        "parent_checkpoint": config.get("resume"), "config": config,
    }
    (exports / "provenance.json").write_text(json.dumps(provenance, indent=2))
    gallery_path = ROOT / "assets/lesson/gallery.json"
    cards = json.loads(gallery_path.read_text()) if gallery_path.exists() else []
    cards = [c for c in cards if c["id"] != args.name]
    cards.append({"id": args.name, "title": args.name.replace("_", " ").title(),
                  "video": f"{args.name}/rollout.mp4", "motion": f"{args.name}/motion.png",
                  "learning_curve": f"{args.name}/learning_curves.png",
                  "caption": args.caption, "experiment": str(run)})
    gallery_path.write_text(json.dumps(cards, indent=2))
    print(out)


if __name__ == "__main__":
    main()
