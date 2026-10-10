"""Consistent notebook figures without hiding experimental configuration."""

import base64
import html
import json
from pathlib import Path
from statistics import mean

STYLE = """
<style>
.bdx-grid {display:flex;flex-wrap:wrap;gap:18px;margin:12px 0 24px}
.bdx-card {flex:1 1 300px;max-width:760px;background:#faf7f0;border:1px solid #d8d1c2;
 border-radius:14px;padding:16px;color:#263b3b;font-family:system-ui,sans-serif}
.bdx-card h4 {margin:0 0 10px;font-size:18px}.bdx-card p {line-height:1.5;margin-bottom:0}
.bdx-card img,.bdx-card video {width:100%;border-radius:8px}
.bdx-table {border-collapse:collapse;margin:14px 0;font-family:system-ui,sans-serif}
.bdx-table th,.bdx-table td {padding:9px 14px;border-bottom:1px solid #d8d1c2;text-align:left}
.bdx-table th {background:#e8efeb}.bdx-note {color:#516766;font-size:13px}
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


def media_pair(items):
    """Display local image/video panels with explicit titles and captions."""
    from IPython.display import HTML, display

    items = list(items)
    cards = []
    card_style = ' style="max-width:1040px"' if len(items) == 1 else ""
    for title, path, caption in items:
        path = Path(path)
        data = base64.b64encode(path.read_bytes()).decode()
        if path.suffix == ".mp4":
            media = f'<video controls playsinline preload="metadata" src="data:video/mp4;base64,{data}"></video>'
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
            f'<article class="bdx-card"{card_style}><h4>{html.escape(title)}</h4>{media}<p>{html.escape(caption)}</p></article>'
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


def reference_comparison(video, reference_image, reference_page):
    """Load credited reference imagery from its publisher, alongside local inference."""
    import urllib.request

    from IPython.display import HTML, display

    data = base64.b64encode(Path(video).read_bytes()).decode()
    image = html.escape(reference_image, quote=True)
    page = html.escape(reference_page, quote=True)
    try:
        # Fetch at execution time: some notebook frontends block remote <img>
        # requests. The third-party photo is still absent from the source bundle.
        request = urllib.request.Request(
            reference_image, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            mime = response.headers.get_content_type()
            if not mime.startswith("image/"):
                raise ValueError("The publisher did not return an image.")
            image = f"data:{mime};base64," + base64.b64encode(response.read()).decode()
    except (OSError, ValueError):
        pass  # The linked source and image alt text remain useful offline.
    display(
        HTML(
            STYLE
            + f'''<div class="bdx-grid">
    <article class="bdx-card"><h4>Real-world inspiration</h4>
    <a href="{page}" target="_blank" rel="noopener"><img style="aspect-ratio:3/2;object-fit:contain;background:#e8efeb" src="{image}" alt="Disney BDX droids — see the credited D23 article"></a>
    <p>Disney BDX droids. Reference photograph served from D23 (Disney).
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
                f'<img style="max-width:100%;margin:12px 0" src="data:image/png;base64,{data}" alt="{html.escape(name)}">'
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
