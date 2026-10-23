"""Consistent notebook figures without hiding experimental configuration."""

import base64
import html
import json
from pathlib import Path
from statistics import mean

STYLE = """
<style>
.bdx-grid {display:flex;flex-wrap:wrap;gap:16px;margin:12px auto 20px;max-width:1000px}
.bdx-card {flex:1 1 300px;max-width:760px;background:#faf7f0;border:1px solid #d8d1c2;
 border-radius:14px;padding:16px;color:#263b3b;font-family:system-ui,sans-serif}
.bdx-card h4 {margin:0 0 10px;font-size:18px}.bdx-card p {line-height:1.5;margin-bottom:0}
.bdx-card img,.bdx-card video {width:100%;border-radius:8px}
.bdx-card img {max-height:420px;object-fit:contain}
table.bdx-table {border-collapse:collapse;margin:14px 0;font-family:system-ui,sans-serif;
 color:inherit;background:transparent}
table.bdx-table thead,table.bdx-table tbody,table.bdx-table tr,
table.bdx-table th,table.bdx-table td {background:transparent !important;color:inherit !important}
table.bdx-table th,table.bdx-table td {padding:9px 14px;border-bottom:1px solid #8888;text-align:left}
table.bdx-table th {font-weight:600}.bdx-note {color:#516766;font-size:13px}
</style>
"""


def table(rows, headings):
    from IPython.display import HTML, display

    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headings)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    display(
        HTML(
            STYLE
            + f'<table class="bdx-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        )
    )


def media_pair(items, video_height=None):
    """Display local image/video panels with explicit titles and captions."""
    from IPython.display import HTML, display

    items = list(items)
    cards = []
    card_style = ' style="max-width:760px"' if len(items) == 1 else ""
    for title, path, caption in items:
        path = Path(path)
        data = base64.b64encode(path.read_bytes()).decode()
        if path.suffix == ".mp4":
            video_style = (
                f' style="height:{int(video_height)}px;object-fit:contain;background:#18352e"'
                if video_height is not None
                else ""
            )
            media = f'<video{video_style} controls playsinline preload="metadata" src="data:video/mp4;base64,{data}"></video>'
        else:
            mime = (
                "image/svg+xml"
                if path.suffix == ".svg"
                else "image/png"
                if path.suffix == ".png"
                else "image/jpeg"
            )
            media = f'<img src="data:{mime};base64,{data}" alt="{html.escape(title, quote=True)}">'
        cards.append(
            f'<article class="bdx-card"{card_style}><h4>{html.escape(title)}</h4>{media}'
            + (f"<p>{html.escape(caption)}</p>" if caption else "")
            + "</article>"
        )
    display(HTML(STYLE + '<div class="bdx-grid">' + "".join(cards) + "</div>"))


def command_table(summary):
    table(
        [
            (
                case,
                f"{sum(t['fell'] for t in trials)}/{len(trials)}",
                round(mean(t["speed_mae"] for t in trials), 3),
                round(mean(t["yaw_mae"] for t in trials), 3),
                round(mean(t["max_cross_track_m"] for t in trials), 3),
            )
            for case, trials in summary["cases"].items()
        ],
        [
            "Command",
            "Falls",
            "Speed error (m/s)",
            "Yaw error (rad/s)",
            "Mean max drift (m)",
        ],
    )


def read_json(path):
    return json.loads(Path(path).read_text())


def interactive_html(path, height=610):
    """Give a generated standalone interactive figure its own document context."""
    from IPython.display import HTML, display

    source = html.escape(Path(path).read_text(), quote=True)
    display(
        HTML(
            f'<iframe title="Joint kinematics explorer" srcdoc="{source}" '
            f'style="border:0;width:100%;height:{height}px;border-radius:12px"></iframe>'
        )
    )


def reference_comparison(video, reference_image, reference_page, reference_photo=None):
    """Display credited reference imagery beside local inference."""
    import urllib.request

    from IPython.display import HTML, display

    data = base64.b64encode(Path(video).read_bytes()).decode()
    image = html.escape(reference_image, quote=True)
    page = html.escape(reference_page, quote=True)
    try:
        if reference_photo is not None:
            image = (
                "data:image/jpeg;base64,"
                + base64.b64encode(Path(reference_photo).read_bytes()).decode()
            )
        else:
            request = urllib.request.Request(
                reference_image, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                mime = response.headers.get_content_type()
                if not mime.startswith("image/"):
                    raise ValueError("The publisher did not return an image.")
                image = (
                    f"data:{mime};base64," + base64.b64encode(response.read()).decode()
                )
    except (OSError, ValueError):
        pass  # The linked source and image alt text remain useful offline.
    display(
        HTML(
            STYLE
            + f'''<div class="bdx-grid">
    <article class="bdx-card"><h4>Real-world inspiration</h4>
    <a href="{page}" target="_blank" rel="noopener"><img style="aspect-ratio:3/2;object-fit:contain;background:#e8efeb" src="{image}" alt="Disney BDX droids — see the credited D23 article"></a>
    <p>Disney BDX droids. Reference photograph: D23 (Disney); not relicensed.
    <a href="{page}" target="_blank" rel="noopener">Read the source article</a>.</p></article>
    <article class="bdx-card"><h4>Our simulated BDX-R walker</h4>
    <video style="aspect-ratio:3/2;object-fit:contain;background:#e8efeb" controls playsinline preload="metadata" src="data:video/mp4;base64,{data}"></video>
    <p>Actual inference from the approved classroom walking policy. This is the open-source BDX-R model, not Disney's hardware or controller.</p>
    </article></div>'''
        )
    )


def recorded_details(
    folder, caption="Recorded instructor evidence; not your new training run."
):
    """Keep reference measurements accessible without opening a large gallery."""
    from IPython.display import HTML, display

    folder = Path(folder)
    figures = []
    for name in ("motion.png", "learning_curves.png", "frames.png", "finish.png"):
        path = folder / name
        if path.is_file():
            data = base64.b64encode(path.read_bytes()).decode()
            figures.append(
                f'<img style="width:720px;max-width:100%;margin:12px 0;display:block" src="data:image/png;base64,{data}" alt="{html.escape(name)}">'
            )
    if figures:
        display(
            HTML(
                "<details><summary>Instructor measurements (optional)</summary><p>"
                + html.escape(caption)
                + "</p>"
                + "".join(figures)
                + "</details>"
            )
        )


def joint_figures(folder):
    """Show the student's joint inspection without repeating the lesson text."""
    folder = Path(folder)
    media_pair(
        [
            ("Joint pivots", folder / "annotated_robot.png", ""),
            ("Kinematic tree", folder / "kinematic_tree.svg", ""),
        ]
    )


def joint_explorer(path):
    """Use browser-side pose switching in Colab; standard widgets elsewhere."""
    import importlib

    path = Path(path)
    try:
        importlib.import_module("google.colab")
    except ImportError:
        pass
    else:
        from IPython.display import HTML, Javascript, display

        # Colab can handle input entirely in its output frame. A Python widget
        # callback would have to wait while a later training cell holds the kernel.
        markup, script = path.read_text().rsplit("<script>", 1)
        script = script.rsplit("</script>", 1)[0]
        panel = HTML(markup)
        display(panel, Javascript(script))
        return panel

    import ipywidgets as widgets
    from IPython.display import display

    # Read older sweep exports as well as the new frame sidecar.
    sidecar = path.with_suffix(".frames.json")
    if sidecar.exists():
        data = read_json(sidecar)
    else:
        source = path.read_text().split("const data=", 1)[1]
        data, _ = json.JSONDecoder().raw_decode(source)
    angles, positions = data["angles"], data["positions"]
    initial = min(range(len(angles)), key=lambda i: abs(angles[i]))
    image = widgets.Image(
        format="jpeg", layout=widgets.Layout(width="420px", max_width="100%")
    )
    slider = widgets.SelectionSlider(
        options=[(f"{angle:g}°", i) for i, angle in enumerate(angles)],
        value=initial,
        description="Angle",
        continuous_update=False,
        layout=widgets.Layout(width="300px", max_width="100%"),
    )
    coordinates = widgets.HTML()
    trace = widgets.HTML()
    xs, zs = [p[0] for p in positions], [p[2] for p in positions]

    def point(p):
        x = 25 + (p[0] - min(xs)) / (max(xs) - min(xs) or 1) * 220
        z = 150 - (p[2] - min(zs)) / (max(zs) - min(zs) or 1) * 125
        return x, z

    points = " ".join(f"{x:.2f},{z:.2f}" for x, z in map(point, positions))

    def update(change=None):
        index = slider.value
        image.value = base64.b64decode(data["frames"][index])
        coordinates.value = "<b>Foot in torso frame</b><br>" + " · ".join(
            f"{axis}: {value:.3f} m" for axis, value in zip("xyz", positions[index])
        )
        x, z = point(positions[index])
        trace.value = (
            '<svg viewBox="0 0 280 190" width="280" aria-label="Foot x–z path">'
            f'<polyline points="{points}" fill="none" stroke="#25847a" stroke-width="3"/>'
            f'<circle cx="{x}" cy="{z}" r="6" fill="#d78535"/>'
            '<text x="80" y="183" font-size="13">Foot x–z path (m)</text></svg>'
        )

    update()
    slider.observe(update, names="value")
    panel = widgets.Box(
        [image, widgets.VBox([slider, coordinates, trace])],
        layout=widgets.Layout(
            display="flex", flex_flow="row wrap", gap="16px", max_width="900px"
        ),
    )
    display(panel)
    return panel
