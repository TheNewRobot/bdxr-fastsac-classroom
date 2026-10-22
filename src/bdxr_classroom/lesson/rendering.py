"""Select and verify NVIDIA EGL without installing or replacing system drivers."""

import json
import subprocess
from pathlib import Path

PROBE = """
import json
import mujoco
from OpenGL import GL

model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(mujoco.MjData(model))
    image = renderer.render()
    def text(key):
        value = GL.glGetString(key)
        return value.decode() if value else "unavailable"
    print(json.dumps({
        "vendor": text(GL.GL_VENDOR),
        "renderer": text(GL.GL_RENDERER),
        "version": text(GL.GL_VERSION),
        "frame_shape": list(image.shape),
    }))
"""


def probe_renderer(python, environment):
    """Probe in a fresh process: EGL selection must precede graphics imports."""
    result = subprocess.run(
        [str(python), "-c", PROBE],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or "Renderer probe failed.")
    return json.loads(result.stdout)


def configure_rendering(work, python, environment):
    """Return verified rendering settings; preserve the existing backend on failure.

    Colab can expose CUDA while omitting NVIDIA's EGL vendor registration.
    MuJoCo's official Colab tutorial supplies that registration. Here it stays
    inside the lesson folder, selected through GLVND's documented override.
    Only fresh child processes use the returned environment.
    """
    work = Path(work).resolve()
    folder = work / ".runtime"
    folder.mkdir(parents=True, exist_ok=True)
    original = dict(environment, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    selected = original
    report = {"before": None, "after": None, "nvidia_verified": False, "errors": []}
    try:
        report["before"] = probe_renderer(python, original)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["errors"].append(str(error))

    before = report["before"] or {}
    if "NVIDIA" in before.get("vendor", "").upper():
        report["after"] = before
        report["nvidia_verified"] = True
    else:
        manifest = folder / "nvidia_egl.json"
        manifest.write_text(
            json.dumps(
                {
                    "file_format_version": "1.0.0",
                    "ICD": {"library_path": "libEGL_nvidia.so.0"},
                },
                indent=2,
            )
            + "\n"
        )
        candidate = dict(original, __EGL_VENDOR_LIBRARY_FILENAMES=str(manifest))
        try:
            after = probe_renderer(python, candidate)
            if "NVIDIA" not in after.get("vendor", "").upper():
                raise RuntimeError(f"NVIDIA EGL probe returned {after['renderer']}.")
            selected = candidate
            report["after"] = after
            report["nvidia_verified"] = True
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            report["errors"].append(str(error))
            report["after"] = report["before"]

    (folder / "rendering.json").write_text(json.dumps(report, indent=2) + "\n")
    runtime_path = work / "runtime_info.json"
    if runtime_path.is_file():
        runtime = json.loads(runtime_path.read_text())
        runtime["rendering"] = report
        runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    info = report["after"] or {}
    print("Graphics vendor:", info.get("vendor", "unavailable"))
    print("Graphics renderer:", info.get("renderer", "unavailable"))
    if report["nvidia_verified"]:
        print("NVIDIA rendering verified. New recording processes use this backend.")
    else:
        print(
            "NVIDIA rendering could not be verified; the previous backend is retained. "
            f"Video generation may be slow. Details: {folder / 'rendering.json'}"
        )
    return selected
