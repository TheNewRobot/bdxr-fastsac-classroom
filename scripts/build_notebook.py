"""Build the uploadable lesson with a small embedded source-code bundle."""

import base64
import io
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


def main():
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in ["src/**/*.py", "configs/*.json", "scripts/setup.py",
                        "requirements.txt", "requirements.lock.txt", "THIRD_PARTY_NOTICES.md",
                        "examples/**/*", "docs/comparison.png"]:
            for path in ROOT.glob(pattern):
                if path.is_file():
                    z.write(path, path.relative_to(ROOT))
    encoded = base64.b64encode(bundle.getvalue()).decode()
    md("""# Teach a little droid to walk
### 163A / 263A · Robotics 101 · BDX-R + FastSAC

**Your challenge:** help BDX-R stay balanced, move forward, and change its walking behavior.

We will **watch → train → compare → explain**. You will produce your own videos and measurements.

**Project credits:** [BDX-R](https://github.com/BDX-R/BDX-R-MjLab) supplies the robot and starting task; [Holosoma / FastSAC](https://younggyo.me/fastsac-humanoid/) supplies the learning components; mjlab and MuJoCo Warp simulate the physics.

**Validation status:** the GPU simulation and training adapter have been exercised locally on an NVIDIA L40S. Actual Google Colab execution is still pending. Training time on your assigned GPU can differ.
""")
    preview = ROOT / "artifacts/robot_preview.png"
    if preview.exists():
        cell = nbf.v4.new_markdown_cell("![Our complete BDX-R robot](attachment:robot.png)\n\n*Rendered locally from the BDX-R model. This image is not a trained walking result.*")
        cell["attachments"] = {"robot.png": {"image/png": base64.b64encode(preview.read_bytes()).decode()}}
        cells.append(cell)
    md("""## 1 · Watch the goal

Play the upstream simulation demonstration. Notice the feet, knees, and body.

**Before we start:** What must the robot control to move without falling?

This is a **credited upstream demonstration**, not a result trained by our notebook.
""")
    code("""from IPython.display import Video, display
display(Video(url="https://github.com/user-attachments/assets/af664888-200a-4969-a248-1ef59d8b259c",
              width=720))
""")
    md("""## 2 · Prepare your workspace

In Colab, select **Runtime → Change runtime type → GPU**.

The next cells unpack this lesson's source code and install the robot and simulator. No repository publication is required: the small code bundle is included in this notebook. Upstream robot assets are downloaded from pinned revisions.

Installation and first-time GPU compilation take additional time. We measure those separately from training.
""")
    code(f'''# @title Unpack the lesson
import base64, io, os, sys, zipfile
from pathlib import Path

WORK = Path("/content/bdxr_classroom") if Path("/content").exists() else Path.cwd() / "bdxr_classroom_lab"
WORK.mkdir(parents=True, exist_ok=True)
SOURCE_BUNDLE = "{encoded}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_BUNDLE))) as archive:
    archive.extractall(WORK)
print("Lesson files:", WORK)
''')
    cells[-1]["metadata"]["cellView"] = "form"
    if (ROOT / "examples/baseline").exists():
        md("""### Recorded classroom examples

These videos are **real policies trained for this lesson on the instructor's L40S GPU**. They are included so you can inspect the behaviors before training your own robot.

The baseline is trained from scratch; variations are labeled in their records. The small saved `policy.pt` files are **inference-only**—they are not full training checkpoints.
""")
        code("""# @title Watch the recorded examples
import html, json
from IPython.display import HTML, Image

cards = []
for name, title in [("baseline", "Walking"), ("slower", "Slower walking"),
                    ("faster", "Faster walking"), ("smoother", "Smoother walking"),
                    ("hopping", "Small hops"), ("hopping_taller", "Higher hop target")]:
    folder = WORK / "examples" / name
    if not (folder / "rollout.mp4").exists():
        continue
    report = json.loads((folder / "evaluation.json").read_text())
    config = json.loads((folder / "provenance.json").read_text())["config"]
    video = base64.b64encode((folder / "rollout.mp4").read_bytes()).decode()
    setting = (f"Foot-height target {config['hop_height'] * 100:.0f} cm"
               if config["task_version"] == "hop"
               else f"Speed target {config['speed']:.2f} m/s; action penalty ×{config['smoothness']}")
    caption = f"{setting} · {report['label']} · falls in {report['fall_fraction']:.0%} of evaluated episodes"
    cards.append(f'<div style="flex:1 1 310px;max-width:460px;padding:12px;border:1px solid #d7e7dc;border-radius:12px">'
                 f'<h4>{html.escape(title)}</h4><video controls loop playsinline width="100%" '
                 f'src="data:video/mp4;base64,{video}"></video><p>{html.escape(caption)}</p></div>')
display(HTML('<div style="display:flex;flex-wrap:wrap;gap:12px">' + ''.join(cards) + '</div>'))
if (WORK / "docs/comparison.png").exists():
    display(Image(filename=str(WORK / "docs/comparison.png"), width=900))
    print("Instructor measurements. Control = unchanged settings with the same extra training budget.")
""")
        cells[-1]["metadata"]["cellView"] = "form"
    code("""# @title Install and check the simulator
import subprocess, time

setup_start = time.monotonic()
print("Installing the lesson environment. This may take several minutes.")
with (WORK / "installation.log").open("w") as log:
    result = subprocess.run([sys.executable, str(WORK / "scripts/setup.py")],
                            cwd=WORK, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print((WORK / "installation.log").read_text()[-6000:])
    raise RuntimeError("Installation failed; see installation.log above.")
PYTHON = WORK / ".venv/bin/python"
RUN_ENV = dict(os.environ, PYTHONPATH=str(WORK / "src"), MUJOCO_GL="egl",
               BDXR_RUNTIME="google-colab" if Path("/content").exists() else "local")
print(f"Installation/checks: {(time.monotonic() - setup_start) / 60:.1f} minutes")
""")
    md("""## 3 · Choose one baseline

An **observation** tells the policy about the robot. An **action** requests joint positions; motor controllers turn those targets into forces. A **reward** scores the resulting behavior.

The baseline asks for forward walking at **0.3 m/s**. Keep this setting for the first run.
Keep the seed unchanged for your first comparison so the runs are easier to compare.

W&B is optional for students. The notebook always saves local curves and videos.
""")
    code("""TARGET_SPEED = 0.3  # @param {type:"number"}
TRAIN_MINUTES = 15  # @param {type:"number"}
USE_WANDB = False  # @param {type:"boolean"}
TRAIN_SEED = 7  # @param {type:"integer"}

gpu_info = subprocess.check_output(
    [str(PYTHON), "-c", "import torch; p=torch.cuda.get_device_properties(0); print(p.total_memory//(1024**3))"],
    cwd=WORK, env=RUN_ENV, text=True)
gpu_gib = int(gpu_info.strip())
# A fixed, measured classroom preset; larger batches are an instructor extension.
NUM_ENVS = 512
BATCH_SIZE = 2048
if USE_WANDB and not RUN_ENV.get("WANDB_API_KEY"):
    import getpass, netrc
    try:
        existing_login = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        existing_login = None
    if not existing_login:
        RUN_ENV["WANDB_API_KEY"] = getpass.getpass("Your W&B API key (hidden): ")
WANDB_MODE = "online" if USE_WANDB else "disabled"
print(f"{NUM_ENVS} parallel robots; up to {TRAIN_MINUTES} training minutes.")
""")
    md("""## 4 · Teach it to walk

Many simulated robots collect experience at once. FastSAC stores that experience and uses it to improve the policy.

**Watch for:** increasing episode duration, forward motion, and fewer falls. A high reward by itself is not proof of walking.

The run saves intermediate policies and a continuation checkpoint. If you interrupt it, wait for checkpoint saving to finish.
""")
    code("""def run_module(module, *args):
    command = [str(PYTHON), "-u", "-m", module, *map(str, args)]
    log_path = WORK / "last_command.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=WORK, env=RUN_ENV,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("step=") or "Traceback" in line or "Error" in line:
                    print(line.rstrip())
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if returncode:
        print(log_path.read_text()[-6000:])
        raise RuntimeError(f"Command failed; full log: {log_path}")

RUN_NAME = "walk_" + time.strftime("%H%M%S")
run_module("bdxr_classroom.train", "--name", RUN_NAME,
           "--num-envs", NUM_ENVS, "--batch-size", BATCH_SIZE,
           "--steps", 6000, "--minutes", TRAIN_MINUTES,
           "--speed", TARGET_SPEED, "--seed", TRAIN_SEED,
           "--compile", "--wandb-mode", WANDB_MODE)
BASELINE = WORK / "runs" / RUN_NAME
print("Saved results:", BASELINE)
""")
    md("""## 5 · Before and after

We evaluate the policy with exploration disabled. The first episode is shown in the video; the report includes **all** evaluated episodes.

An initial policy may fall quickly. A final policy may still need more training—describe what you actually observe.
""")
    code("""run_module("bdxr_classroom.evaluate", BASELINE / "initial.pt", "--episodes", 3)
run_module("bdxr_classroom.evaluate", BASELINE / "final.pt", "--episodes", 5)
print("BEFORE: initial policy")
display(Video(str(BASELINE / "eval_initial/rollout.mp4"), embed=True, width=640))
print("AFTER: trained policy")
display(Video(str(BASELINE / "eval_final/rollout.mp4"), embed=True, width=640))
""")
    md("""## 6 · Measure what changed

- **Learning curves:** how reward, episode duration, and speed evolve.
- **Motion plots:** achieved speed and left/right knee angles.
- **Evaluation report:** falls and speed across several starting orientations.

The knee-angle plot connects this learning experiment to kinematics: changing joint configurations produces the observed leg motion.
""")
    code("""import json
from IPython.display import Image

run_module("bdxr_classroom.plots", BASELINE)
display(Image(filename=str(BASELINE / "learning_curves.png")))
display(Image(filename=str(BASELINE / "eval_final/motion.png")))
evaluation = json.loads((BASELINE / "eval_final/evaluation.json").read_text())
print("Target speed:", evaluation["target_speed"])
print("Mean achieved speed:", round(evaluation["mean_speed"], 3))
print("Fraction of episodes with a fall:", evaluation["fall_fraction"])
print("Measured training time:", json.loads((BASELINE / "timing.json").read_text()))
""")
    md("""## 7 · Change one thing

Choose one experiment and write a prediction.

| Experiment | Change | Keep fixed |
|---|---|---|
| Slower walking | Target speed | Reward weights and evaluation procedure |
| Faster walking | Target speed | Reward weights and evaluation procedure |
| Smoother walking | Penalty on abrupt action changes | Target speed and evaluation procedure |

This cell **fine-tunes** the baseline. It is not a second independent training-from-scratch run. The replay buffer starts fresh because the task has changed.

For a fairer experiment, also continue the unchanged baseline for the same number of steps. That is a **control**: it helps separate a setting's effect from extra training. The instructor results include this comparison.
""")
    code("""EXPERIMENT = "smoother"  # @param ["smoother", "slower", "faster"]
PREDICTION = "The leg motion will become less abrupt."  # @param {type:"string"}
VARIATION_MINUTES = 5  # @param {type:"number"}

speed_offset = {"slower": -0.15, "faster": 0.15, "smoother": 0.0}[EXPERIMENT]
variation_speed = max(0.05, TARGET_SPEED + speed_offset)
smoothness = 3.0 if EXPERIMENT == "smoother" else 1.0
VAR_NAME = EXPERIMENT + "_" + time.strftime("%H%M%S")
print("Prediction:", PREDICTION)
run_module("bdxr_classroom.train", "--name", VAR_NAME,
           "--num-envs", NUM_ENVS, "--batch-size", BATCH_SIZE,
           "--steps", 2000, "--minutes", VARIATION_MINUTES,
           "--speed", variation_speed, "--smoothness", smoothness, "--seed", TRAIN_SEED,
           "--resume", BASELINE / "final.pt", "--finetune",
           "--compile", "--wandb-mode", WANDB_MODE)
VARIATION = WORK / "runs" / VAR_NAME
run_module("bdxr_classroom.evaluate", VARIATION / "final.pt", "--episodes", 5)
display(Video(str(VARIATION / "eval_final/rollout.mp4"), embed=True, width=640))
display(Image(filename=str(VARIATION / "eval_final/motion.png")))
""")
    md("""## 8 · Explain your result

1. What changed in the video? Describe a specific motion.
2. Which measured result supports your observation?
3. Did the robot meet its target without falling?
4. Did the experiment match your prediction?

""")
    md("""### Optional extension · Small hops

The hopping task adds a **rhythm signal** and rewards coordinated foot clearance and landing. It still learns its motor commands with SAC.

Our tested route has two stages: learn tiny hops, then fine-tune with a higher foot and body-height target. The target height is an objective, not a guarantee of achieved jump height.

The recorded gallery also shows a follow-up changing only the foot-height target from 7 to 10 cm. Measured foot clearance increases; this does not mean the body jumps 10 cm.

Leave this switched off for the short class demo and use the recorded hopping example. Switch it on for an extended lab. We exclude the initial drop onto the floor when counting hops, and require an upright landing.
""")
    code("""# @title Optional: train the hopping extension
TRAIN_HOPPING = False  # @param {type:"boolean"}
HOP_RESULTS = None
if TRAIN_HOPPING:
    HOP_ENVS = 1024 if gpu_gib >= 8 else 512
    HOP_BATCH = 4096 if gpu_gib >= 8 else 2048
    tiny_name = "tiny_hops_" + time.strftime("%H%M%S")
    run_module("bdxr_classroom.train", "--name", tiny_name,
               "--task-version", "hop", "--speed", 0,
               "--num-envs", HOP_ENVS, "--batch-size", HOP_BATCH,
               "--seed", 42,
               "--steps", 4000, "--minutes", 10, "--compile", "--wandb-mode", WANDB_MODE)
    tiny_path = WORK / "runs" / tiny_name
    hop_name = "hops_" + time.strftime("%H%M%S")
    run_module("bdxr_classroom.train", "--name", hop_name,
               "--task-version", "hop", "--speed", 0,
               "--hop-height", 0.07, "--hop-body-weight", 6,
               "--num-envs", HOP_ENVS, "--batch-size", HOP_BATCH,
               "--seed", 42,
               "--steps", 3000, "--minutes", 5,
               "--resume", tiny_path / "final.pt", "--finetune",
               "--compile", "--wandb-mode", WANDB_MODE)
    HOP_RESULTS = WORK / "runs" / hop_name
    run_module("bdxr_classroom.evaluate", HOP_RESULTS / "final.pt", "--episodes", 5)
    display(Video(str(HOP_RESULTS / "eval_final/rollout.mp4"), embed=True, width=640))
    display(Image(filename=str(HOP_RESULTS / "eval_final/motion.png")))
    print(json.loads((HOP_RESULTS / "eval_final/evaluation.json").read_text()))
else:
    print("Optional training skipped. The recorded examples remain available above.")
""")
    md("""## 9 · The important ideas

| Concept | Meaning in this lesson |
|---|---|
| State / observation | Joint positions and velocities, body motion, and the requested speed |
| Policy / actor | A neural network choosing ten leg-joint position commands |
| Physics engine | MuJoCo Warp calculates motion and contact forces |
| Reward | Encourages speed tracking, upright posture, and useful foot motion; penalizes unwanted motion |
| Critic | Learns to estimate the future value of actions |
| Replay buffer | Reuses experience collected by simulated robots |
| Exploration | Tries different actions during training |
| Evaluation | Tests the learned behavior without exploratory action noise |

**SAC** means *Soft Actor-Critic*. It learns an actor and a critic and includes an entropy term encouraging exploration. FastSAC adapts this approach for large parallel batches.

**Kinematics and dynamics:** joint angles describe configuration; their evolution describes motion. Forces, gravity, and contacts determine whether that motion is physically achievable.

**The lesson:** the robot learns the objective we implement. We must inspect its behavior, not just its score. Five evaluations test one trained policy from different starting states; they are not five independent training runs.
""")
    md("""## 10 · Save your work

The ZIP includes your videos, plots, configurations, and compact training checkpoints. The larger `resume.pt` file includes replay data and remains in the run folder; copy it to persistent storage separately if you want to continue training later.
""")
    code("""import zipfile
download_path = WORK / "my_bdxr_results.zip"
with zipfile.ZipFile(download_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for folder in [BASELINE, VARIATION] + ([HOP_RESULTS] if HOP_RESULTS else []):
        for file in folder.rglob("*"):
            if file.is_file() and "wandb" not in file.parts and file.name != "resume.pt":
                archive.write(file, file.relative_to(WORK))
print("Saved:", download_path)
try:
    from google.colab import files
    files.download(str(download_path))
except ImportError:
    print("Outside Colab: retrieve the ZIP from the path above.")
""")
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "BDXR_Walking_101.ipynb"},
        "accelerator": "GPU",
    })
    nbf.validate(nb)
    dest = ROOT / "notebooks/BDXR_Walking_101.ipynb"
    nbf.write(nb, dest)
    print(f"Wrote {dest}: {len(cells)} cells, {dest.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
