"""Command-line entry point for hopping checks."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.evaluation.hopping_checks import check, choose

__all__ = ["check", "choose"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    targets = json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/hopping_targets.json"
        ).read_text()
    )
    result = check(json.loads(args.summary.read_text()), targets)
    args.summary.with_name("hopping_check.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
