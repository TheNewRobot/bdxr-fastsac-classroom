"""Build the portable notebook from readable, chapter-sized source files.

Sections use # %% for Python cells and # %% [markdown] for commented Markdown.
Use Markdown lesson-asset: links for self-contained image attachments.
Only the bootstrap cell receives an embedded archive; authored lesson code stays
readable in notebooks/sections/. No remote/private-repository access is needed.
"""

import argparse
import base64
import io
from pathlib import Path
import re
import zipfile

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def embed_references(cells):
    """Embed teaching images and animated previews in Markdown attachments only.

    No cell is executed and no code output is populated. Animated GIF previews
    use native Markdown image syntax, so clearing outputs cannot remove them.
    Student-generated MP4 players remain outputs of their activity cells.
    """
    asset_root = (ROOT / "assets/lesson").resolve()
    for cell in cells:
        if cell.cell_type == "markdown":
            attachments = {}

            def attach(match):
                relative = match.group(1)
                path = (asset_root / relative).resolve()
                if not path.is_relative_to(asset_root):
                    raise ValueError(f"Reference escapes lesson assets: {relative}")
                mime = {
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".gif": "image/gif",
                }[path.suffix]
                name = relative.replace("/", "-")
                raw = path.read_bytes()
                if path.suffix in {".png", ".jpg"}:
                    # Smaller charts keep the explanation in view. Robot portraits
                    # retain more space; originals remain in the lesson archive.
                    width = (
                        800
                        if path.name
                        in {
                            "robot.png",
                            "annotated_robot.png",
                            "course_preview.png",
                            "kinematic_tree.png",
                        }
                        else 720
                    )
                    if path.suffix == ".jpg":
                        width = 480
                    image = Image.open(io.BytesIO(raw))
                    image.thumbnail((width, 900), Image.Resampling.LANCZOS)
                    if path.suffix == ".jpg":
                        canvas = Image.new("RGB", (480, 320), "#faf7f0")
                        canvas.paste(
                            image, ((480 - image.width) // 2, (320 - image.height) // 2)
                        )
                        image = canvas
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    mime = "image/png"
                    raw = buffer.getvalue()
                data = base64.b64encode(raw).decode()
                attachments[name] = {mime: data}
                return f"(attachment:{name})"

            cell.source = re.sub(r"\(lesson-asset:([^)]+)\)", attach, cell.source)
            if attachments:
                cell.attachments = attachments


def source_bundle():
    # GIF previews already live in Markdown attachments; bundle the original MP4s
    # for student use without duplicating the preview bytes.
    archive_buffer = io.BytesIO()
    files = set()
    for pattern in (
        "src/**/*.py",
        "src/**/*.html",
        "scripts/**/*.py",
        "configs/*.json",
        "requirements*.txt",
        "THIRD_PARTY_NOTICES.md",
        "assets/lesson/**/*",
    ):
        files.update(
            p
            for p in ROOT.glob(pattern)
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".gif"
        )
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo(
                path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return base64.b64encode(archive_buffer.getvalue()).decode()


def section_cells(path):
    parts = re.split(r"^# %%([^\n]*)\n", path.read_text(), flags=re.MULTILINE)
    for index in range(1, len(parts), 2):
        marker, source = parts[index], parts[index + 1].strip()
        if "[markdown]" in marker:
            source = "\n".join(
                line[2:]
                if line.startswith("# ")
                else line[1:]
                if line.startswith("#")
                else line
                for line in source.splitlines()
            )
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            cell = nbformat.v4.new_code_cell(source)
            if "[hidden]" in marker:
                cell.metadata["cellView"] = "form"
                cell.metadata["jupyter"] = {"source_hidden": True}
                cell.metadata["tags"] = ["hide-input"]
            if "[instructor]" in marker:
                raise ValueError(
                    "Teaching examples belong in Markdown, not code cells."
                )
        cell.metadata["lesson_section"] = path.stem
        cell.id = f"{path.stem}-{index // 2:02d}"
        yield cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "notebooks/BDXR_Walking_101.ipynb"
    )
    args = parser.parse_args()
    cells = [
        cell
        for path in sorted((ROOT / "notebooks/sections").glob("*.py"))
        for cell in section_cells(path)
    ]
    if not cells:
        raise RuntimeError("No authored notebook sections found.")
    # Keep illustrations with their explanation, rather than in detached cells.
    merged = []
    for cell in cells:
        if (
            merged
            and cell.cell_type == merged[-1].cell_type == "markdown"
            and (cell.metadata.lesson_section == merged[-1].metadata.lesson_section)
        ):
            merged[-1].source += "\n\n" + cell.source
        else:
            merged.append(cell)
    cells = merged
    embed_references(cells)
    encoded = source_bundle()
    replacements = 0
    for cell in cells:
        if "__SOURCE_BUNDLE__" in cell.source:
            cell.source = cell.source.replace("__SOURCE_BUNDLE__", encoded)
            replacements += 1
    assert replacements == 1, "Exactly one isolated bootstrap archive is required."
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"name": "BDXR_Walking_101.ipynb"},
        },
    )
    if args.output.exists():
        previous = nbformat.read(args.output, as_version=4)
        # Retain the user's local kernel selection when regenerating the lesson.
        for key in ("kernelspec", "vscode"):
            if key in previous.metadata:
                notebook.metadata[key] = previous.metadata[key]
    nbformat.validate(notebook)
    nbformat.write(notebook, args.output)
    print(
        f"Wrote {args.output}: {len(cells)} cells, {args.output.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
