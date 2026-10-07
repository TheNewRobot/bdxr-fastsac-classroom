"""Plot measured training logs without a W&B account."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(run_dir, through_step=None, output=None):
    run_dir = Path(run_dir)
    rows = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    if through_step is not None:
        rows = [r for r in rows if r["step"] <= through_step]
    if not rows:
        raise ValueError("No training measurements have been logged yet")
    x = np.array([r["transitions"] for r in rows]) / 1e6
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    for ax, key, title in zip(
        axs, ["episode_return", "episode_seconds", "forward_speed"],
        ["Episode reward", "Episode duration (s)", "Mean forward speed (m/s)"],
    ):
        ax.plot(x, [r[key] for r in rows], color="#247a56", linewidth=2)
        ax.set(xlabel="Simulated transitions (millions)", title=title)
        ax.grid(alpha=0.2)
    axs[2].axhline(rows[-1]["target_speed"], color="#d58b22", linestyle="--", label="Target")
    axs[2].legend()
    destination = Path(output) if output else run_dir / "learning_curves.png"
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--through-step", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(plot_run(args.run_dir, args.through_step, args.output))
