"""Build the portable notebook from readable, chapter-sized source files.

Sections use # %% for Python cells and # %% [markdown] for commented Markdown.
Mark read-only instructor display cells [instructor] to save their outputs.
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
import sys
from unittest.mock import patch

import nbformat
from PIL import Image
from IPython.core.formatters import DisplayFormatter

ROOT = Path(__file__).resolve().parents[2]


def embed_references(cells):
    """Attach prose illustrations and render only explicitly marked reference cells.

    Student code is never executed by the builder. Reference cells are authored
    local-media displays, with saved outputs that can also be regenerated after
    setup. Markdown attachments survive clearing notebook outputs.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from bdxr_classroom.lesson.presentation import (
        media_pair,
        read_json,
        recorded_details,
    )

    scope = dict(
        WORK=ROOT,
        media_pair=media_pair,
        read_json=read_json,
        recorded_details=recorded_details,
    )
    formatter = DisplayFormatter()
    asset_root = (ROOT / "assets/lesson").resolve()
    for cell in cells:
        if cell.cell_type == "markdown":
            attachments = {}

            def attach(match):
                relative = match.group(1)
                path = (asset_root / relative).resolve()
                if not path.is_relative_to(asset_root):
                    raise ValueError(f"Reference escapes lesson assets: {relative}")
                mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
                name = relative.replace("/", "-")
                raw = path.read_bytes()
                if path.suffix == ".png":
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
                    image = Image.open(io.BytesIO(raw))
                    image.thumbnail((width, 900), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    raw = buffer.getvalue()
                data = base64.b64encode(raw).decode()
                attachments[name] = {mime: data}
                return f"(attachment:{name})"

            cell.source = re.sub(r"\(lesson-asset:([^)]+)\)", attach, cell.source)
            if attachments:
                cell.attachments = attachments
        elif cell.metadata.get("instructor_reference"):
            outputs = []

            def save_display(*objects, **kwargs):
                for obj in objects:
                    data, metadata = formatter.format(obj)
                    outputs.append(
                        nbformat.v4.new_output(
                            "display_data", data=data, metadata=metadata
                        )
                    )

            with patch("IPython.display.display", save_display):
                exec(compile(cell.source, cell.id, "exec"), scope)
            if not outputs:
                raise ValueError(f"Empty instructor reference: {cell.id}")
            cell.outputs = outputs
            cell.metadata["jupyter"] = {"source_hidden": True}
            cell.metadata["tags"] = ["instructor-reference", "hide-input"]


def source_bundle():
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
            if p.is_file() and "__pycache__" not in p.parts
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
                cell.metadata["instructor_reference"] = True
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
