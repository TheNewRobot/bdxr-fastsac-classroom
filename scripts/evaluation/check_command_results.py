"""Command-line entry point for walking checks."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.evaluation.walking_checks import (
    check,
    checkpoint_digest,
    select_checkpoint,
)

__all__ = ["check", "checkpoint_digest", "select_checkpoint"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    result = check(
        json.loads(args.summary.read_text()),
        json.loads((ROOT / "configs/evaluation_targets.json").read_text()),
    )
    (args.summary.parent / "selection_check.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
