"""Export a compact inference-only policy for the student notebook."""

import argparse
from pathlib import Path
import torch


def export_policy(checkpoint, destination):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    compact = {
        k: state[k] for k in ["actor", "actor_norm", "dimensions", "config", "step"]
    }
    compact["format"] = "bdxr-inference-only-v1"
    compact["source_checkpoint"] = str(checkpoint)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, destination)
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("destination")
    args = p.parse_args()
    print(export_policy(args.checkpoint, args.destination))
