"""Plot measured differences; reward scores across changed tasks are not comparable."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ["slower", "baseline", "control", "faster", "smoother"]
    records = {}
    for folder in sorted((ROOT / "examples").iterdir()):
        if not (folder / "evaluation.json").exists():
            continue
        result = json.loads((folder / "evaluation.json").read_text())
        provenance = json.loads((folder / "provenance.json").read_text())
        records[folder.name] = {
            "mean_speed": result["mean_speed"],
            "target_speed": result["target_speed"],
            "fall_fraction": result["fall_fraction"],
            "training_seconds": provenance["training_seconds"],
            "training_origin": provenance["training_origin"],
            "action_change_rms": np.mean([
                e["steady_action_change_rms"] for e in result["episodes"]]),
            "landed_hops": [e["landed_hops"] for e in result["episodes"]],
            "foot_clearance": np.mean([
                e["steady_peak_min_foot_height"] for e in result["episodes"]]),
        }
    (ROOT / "docs/results.json").write_text(json.dumps(records, indent=2))
    names = [n for n in names if n in records]
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    positions = np.arange(len(names))
    colors = ["#69a5a5", "#25825a", "#a0a0a0", "#e9a45c", "#778bc2"][:len(names)]
    axes[0].bar(positions, [records[n]["mean_speed"] for n in names],
               color=colors, label="Measured speed")
    axes[0].plot(positions, [records[n]["target_speed"] for n in names],
               "ko", markersize=6, label="Requested speed")
    axes[0].set_ylabel("Forward speed (m/s)")
    axes[0].set_title("Does the robot follow the command?")
    axes[0].legend()
    axes[1].bar(positions, [records[n]["action_change_rms"] for n in names],
               color=colors)
    axes[1].set_ylabel("RMS change in normalized action")
    axes[1].set_title("How abruptly do commands change?")
    for ax in axes:
        ax.set_xticks(positions, names)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Same walking parent • one parameter changed per variation",
                 fontsize=13)
    fig.savefig(ROOT / "docs/comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
