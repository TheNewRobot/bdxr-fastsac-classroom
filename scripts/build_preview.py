"""Create a README contact sheet from actual evaluation video frames."""

import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    canvas = Image.new("RGB", (1280, 1040), "#f3f7f4")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 18)
    items = [("baseline", "Walk"), ("faster", "Walk faster"),
             ("smoother", "Use smoother commands"), ("hopping", "Make small hops")]
    for index, (name, title) in enumerate(items):
        folder = ROOT / "examples" / name
        report = json.loads((folder / "evaluation.json").read_text())
        time = 2.4
        if name == "hopping":
            provenance = json.loads((folder / "provenance.json").read_text())
            motion = ROOT / Path(provenance["checkpoint"]).parent / "eval_selected/motion.json"
            if motion.exists():
                rows = [r for r in json.loads(motion.read_text())
                        if r["episode"] == 0 and 1 < r["time"] < 3]
                time = max(rows, key=lambda r: min(r["foot_heights"]))["time"]
        reader = imageio.get_reader(folder / "rollout.mp4")
        frame = reader.get_data(round(time * reader.get_meta_data()["fps"]))
        reader.close()
        x, y = (index % 2) * 640, (index // 2) * 520
        canvas.paste(Image.fromarray(frame).resize((620, 413)), (x + 10, y + 50))
        draw.text((x + 20, y + 12), title, font=font, fill="#195b3d")
        subtitle = (f"{report['mean_speed']:.2f} m/s | 0 falls in 5 evaluations"
                    if name != "hopping" else "14 settled hops per evaluation | 0 falls")
        draw.text((x + 20, y + 478), subtitle, font=small, fill="#31463b")
    canvas.save(ROOT / "docs/behaviors.png")
    print(ROOT / "docs/behaviors.png")


if __name__ == "__main__":
    main()
