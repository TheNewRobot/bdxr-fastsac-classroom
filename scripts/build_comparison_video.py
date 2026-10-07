"""Create a synchronized walking comparison from actual evaluation recordings."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    examples = [
        ("slower", "Slower | target 0.15 m/s"),
        ("baseline", "Baseline | target 0.30 m/s"),
        ("faster", "Faster | target 0.45 m/s"),
        ("smoother", "Smoother | action penalty x3"),
    ]
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    for name, _ in examples:
        command.extend(["-i", str(ROOT / "examples" / name / "rollout.mp4")])
    filters = []
    for i, (_, label) in enumerate(examples):
        filters.append(
            f"[{i}:v]scale=600:400,setsar=1,drawbox=x=0:y=0:w=iw:h=38:"
            f"color=0x143b2e@0.9:t=fill,drawtext=text='{label}':"
            f"fontcolor=white:fontsize=21:x=12:y=8[v{i}]")
    filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|600_0|0_400|600_400:shortest=1[out]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]",
                    "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
                    "-an", "-y", str(ROOT / "docs/walking_comparison.mp4")])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
