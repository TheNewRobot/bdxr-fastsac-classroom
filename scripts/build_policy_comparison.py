"""Compose measured policy videos and a compact comparison figure for the lesson."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.experiments import create_experiment, experiment_root, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", nargs=2, action="append", required=True, metavar=("LABEL", "DIRECTORY"))
    parser.add_argument("--case", default="forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= len(args.evaluation) <= 4:
        parser.error("Provide two to four evaluations")
    records, readers = [], []
    for label, directory in args.evaluation:
        directory = Path(directory)
        report = json.loads((directory / "summary.json").read_text())
        trials = report["cases"][args.case]
        root = experiment_root(report["checkpoint"])
        video = root / "videos" / directory.name / f"{args.case}.mp4"
        records.append({"label": label, "evaluation": str(directory.resolve()),
                        "checkpoint": report["checkpoint"], "trials": trials})
        readers.append(imageio.get_reader(video))
    output = create_experiment("policy_comparison")
    write_manifest(output, {"name": "policy_comparison", "hardware": "CPU media composition", "seed": 1000},
                   {"kind": "Recorded evaluation comparison; no new simulation",
                    "evaluations": [r["evaluation"] for r in records]})
    (output / "config/comparison.json").write_text(json.dumps(
        {"case": args.case, "destination": str(args.output), "sources": records}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#266052", "#727d7b", "#317bba", "#bc7825"]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    counts = [reader.count_frames() for reader in readers]
    fps = readers[0].get_meta_data()["fps"]
    if any(abs(reader.get_meta_data()["fps"] - fps) > .01 for reader in readers):
        raise ValueError("Comparison recordings must have the same frame rate")
    height = 320 if len(readers) == 2 else 640
    try:
        with imageio.get_writer(args.output, fps=fps, codec="libx264", quality=7) as writer:
            last_frames = [None] * len(readers)
            for frame_index in range(max(counts)):
                canvas = Image.new("RGB", (960, height), "#edf1e9")
                draw = ImageDraw.Draw(canvas)
                for index, (reader, record) in enumerate(zip(readers, records)):
                    ended = frame_index >= counts[index]
                    if not ended:
                        last_frames[index] = Image.fromarray(reader.get_data(frame_index)).resize((480, 320))
                    x, y = index % 2 * 480, index // 2 * 320
                    canvas.paste(last_frames[index], (x, y))
                    draw.rectangle((x, y, x + 480, y + 31), fill=colors[index])
                    falls = sum(t["fell"] for t in record["trials"])
                    title = f"{record['label']} | {args.case} | falls {falls}/{len(record['trials'])}"
                    draw.text((x + 9, y + 5), title, font=font, fill="white")
                    if ended:
                        draw.text((x + 9, y + 290), "Recorded trial ended early", font=font, fill="red")
                writer.append_data(np.asarray(canvas))
    finally:
        for reader in readers:
            reader.close()
    metrics = [("action_change_rms", "Action-change RMS", "normalized actions"),
               ("swing_height_p95", "Foot swing (95th percentile)", "m"),
               ("max_cross_track_m", "Maximum lateral deviation", "m")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, (key, title, units) in zip(axes, metrics):
        groups = [[trial[key] for trial in record["trials"]] for record in records]
        axis.bar(range(len(records)), [statistics.mean(g) for g in groups],
                 yerr=[statistics.pstdev(g) for g in groups], color=colors[:len(records)], capsize=4)
        axis.set_xticks(range(len(records)), [r["label"] for r in records], rotation=20, ha="right")
        axis.set(title=title, ylabel=units)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{args.case.capitalize()} command · variation across starting states, not training seeds")
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    (output / "evaluation/outputs.json").write_text(json.dumps(
        {"video": str(args.output), "figure": str(args.output.with_suffix(".png"))}, indent=2))
    print(f"Comparison: {args.output}\nProvenance: {output}")


if __name__ == "__main__":
    main()
