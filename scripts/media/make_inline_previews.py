"""Rebuild the short Markdown animations from the credited instructor MP4s.

The manifest records each excerpt's duration, frame rate and display width.
Only presentation files change; recordings and training results stay intact.
"""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/lesson"


def main():
    manifest = ASSETS / "inline/provenance.json"
    records = json.loads(manifest.read_text())
    for record in records:
        source = ASSETS / record["source"]
        filters = f"fps={record['fps']},scale={record['width']}:-1:flags=lanczos"
        if record.get("canvas_height"):
            filters += (
                f",pad={record['width']}:{record['canvas_height']}:"
                "(ow-iw)/2:(oh-ih)/2:color=0xfaf7f0"
            )
        filters += (
            ",split[s0][s1];[s0]palettegen=max_colors=96[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-threads",
                "1",
                "-i",
                str(source),
                "-t",
                str(record["seconds"]),
                "-filter_complex_threads",
                "1",
                "-filter_complex",
                filters,
                "-loop",
                "0",
                str(ASSETS / "inline" / record["file"]),
            ],
            check=True,
        )
        record["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
