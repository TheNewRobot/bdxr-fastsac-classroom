"""Build the portable notebook from readable, chapter-sized source files.

Sections use # %% for Python cells and # %% [markdown] for commented Markdown.
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

ROOT = Path(__file__).resolve().parents[2]


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
    nbformat.validate(notebook)
    nbformat.write(notebook, args.output)
    print(
        f"Wrote {args.output}: {len(cells)} cells, {args.output.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()
