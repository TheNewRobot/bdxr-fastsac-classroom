"""Build the portable lesson with curated media, source, and the playable game."""

import base64
import io
from pathlib import Path
import zipfile
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text, hidden=False):
        cell = nbf.v4.new_code_cell(text)
        if hidden:
            cell.metadata["cellView"] = "form"
        cells.append(cell)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for pattern in ["src/**/*.py", "src/**/*.html", "configs/*.json",
                        "requirements*.txt", "scripts/setup.py", "THIRD_PARTY_NOTICES.md",
                        "assets/lesson/**/*"]:
            for path in ROOT.glob(pattern):
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT))
    encoded = base64.b64encode(bundle.getvalue()).decode()
    md("""# Teach a little droid to walk
### 163A / 263A · BDX-R + FastSAC

**Watch → train → measure → change one thing → play.**

Train a low-level robot controller, compare its motion, then load your checkpoint into a small desert obstacle course.

**Credits:** [BDX-R](https://github.com/BDX-R/BDX-R-MjLab) supplies the complete robot; [Holosoma / FastSAC](https://younggyo.me/fastsac-humanoid/) supplies the learning components; mjlab and MuJoCo Warp simulate physics.

**Testing:** local L40S testing is recorded in the repository. Actual Google Colab execution remains pending. A 15-minute training budget is not a guarantee for every GPU, seed, or behavior.
""")
    md("""## 1 · Set up

Select **Runtime → Change runtime type → GPU**. This notebook includes the lesson source and selected media; it downloads pinned upstream robot assets.
""")
    code(f'''# @title Unpack the lesson
import base64, io, zipfile, json, time, os, subprocess
from pathlib import Path
from IPython.display import display, Video, Image, HTML, IFrame
WORK = Path("/content/bdxr_classroom") if Path("/content").exists() else Path.cwd() / "bdxr_classroom_lab"
WORK.mkdir(parents=True, exist_ok=True)
SOURCE_BUNDLE = "{encoded}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_BUNDLE))) as archive:
    archive.extractall(WORK)
print("Lesson files:", WORK)
''', hidden=True)
    md("""### Watch recorded experiments

These are actual instructor recordings. Captions distinguish preliminary experiments from selected command policies. A video that looks attractive does not by itself establish straightness or command-following.
""")
    code('''# @title Recorded videos and real learning curves
import html
gallery = json.loads((WORK / "assets/lesson/gallery.json").read_text())
cards = []
legacy_cards = []
for item in gallery:
    folder = WORK / "assets/lesson"
    video = base64.b64encode((folder / item["video"]).read_bytes()).decode()
    curve = base64.b64encode((folder / item["learning_curve"]).read_bytes()).decode() if item.get("learning_curve") else None
    motion = base64.b64encode((folder / item["motion"]).read_bytes()).decode() if item.get("motion") else None
    plots = "".join(f'<img width="100%" src="data:image/png;base64,{data}">' for data in [curve, motion] if data)
    destination = legacy_cards if item["video"].startswith("legacy/") else cards
    destination.append(f'<article style="flex:1 1 320px;max-width:480px;padding:12px;border:1px solid #b9cebe;border-radius:12px">'
                 f'<h4>{html.escape(item["title"])}</h4><video controls loop playsinline width="100%" src="data:video/mp4;base64,{video}"></video>'
                 f'<p>{html.escape(item["caption"])}</p><details><summary>Measured curves and motion</summary>{plots}</details></article>')
display(HTML('<div style="display:flex;flex-wrap:wrap;gap:12px">' + "".join(cards) + "</div>"))
display(HTML('<details><summary>Earlier experiments: useful failures and preliminary behaviors</summary>'
             '<div style="display:flex;flex-wrap:wrap;gap:12px">' + "".join(legacy_cards) + "</div></details>"))
''', hidden=True)
    code('''# @title Install pinned dependencies
setup_start = time.monotonic()
setup_log = WORK / "setup.log"
with setup_log.open("w") as log:
    result = subprocess.run(["python", "scripts/setup.py"], cwd=WORK, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print(setup_log.read_text()[-6000:])
    raise RuntimeError("Setup failed; see setup.log")
PYTHON = str(WORK / ".venv/bin/python")
runtime_env = dict(os.environ, PYTHONPATH=str(WORK / "src"), MUJOCO_GL="egl",
                   BDXR_RUNTIME="colab" if Path("/content").exists() else "local_notebook")
gpu = subprocess.check_output([PYTHON, "-c", "import torch,json; print(json.dumps(dict(name=torch.cuda.get_device_name(0),gib=torch.cuda.get_device_properties(0).total_memory/2**30)))"],
                             cwd=WORK, env=runtime_env, text=True)
gpu = json.loads(gpu)
print("GPU:", gpu)
print("Setup seconds:", round(time.monotonic() - setup_start))
''', hidden=True)
    md("""## 2 · What does the robot learn?

| Part | Meaning |
|---|---|
| Observation | Angular velocity, gravity direction, joint positions/velocities, previous actions, and requested speed/yaw rate |
| Action | Ten leg-joint position commands; motor controllers generate forces |
| Reward | Encourages command tracking and balance; penalizes unwanted motion |
| Actor | Chooses actions from proprioception and commands |
| Critic | Uses additional simulator information during training to estimate future value |
| Replay | Reuses collected experience to improve the networks |

**The actor does not receive true base linear velocity, world position, absolute heading, or a terrain map.** The critic may use privileged observations. Only the actor controls the robot during evaluation and play.

Positive yaw-rate commands turn left; negative commands turn right. Zero yaw asks for no rotation—it is not an external heading-correction controller.

SAC learns actions with high estimated future reward while encouraging exploration. Replay lets it learn from earlier experience. We use SAC here; the older reference notebook used PPO.
""")
    code('''# @title How the controller learns and plays
display(HTML((WORK / "assets/lesson/controller.svg").read_text()))
''', hidden=True)
    code('''# @title Training settings
NUM_ENVS = 512
BATCH_SIZE = 2048
TRAIN_SEED = 7
TRAIN_MINUTES = 15  # @param {type:"number"}
TRAIN_STEPS = 10000  # @param {type:"integer"}
USE_WANDB = False  # @param {type:"boolean"}
WANDB_MODE = "online" if USE_WANDB else "disabled"
if USE_WANDB and not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
    import getpass
    runtime_env["WANDB_API_KEY"] = getpass.getpass("Your W&B API key (not saved in this notebook): ")

def run_module(module, *arguments):
    command = [PYTHON, "-u", "-m", module] + [str(value) for value in arguments]
    log_path = WORK / "last_command.log"
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(command, cwd=WORK, env=runtime_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                if line.startswith(("step=", "Experiment:", "forward:", "backward:", "sequence:")) or "Error" in line:
                    print(line.rstrip())
            result = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if result:
        print(log_path.read_text()[-6000:])
        raise RuntimeError("Command failed; inspect last_command.log")

def train(name, *arguments):
    pointer = WORK / "last_experiment.txt"
    run_module("bdxr_classroom.train", "--name", name,
               "--num-envs", NUM_ENVS, "--batch-size", BATCH_SIZE, "--seed", TRAIN_SEED,
               "--compile", "--wandb-mode", WANDB_MODE, "--result-file", pointer, *arguments)
    return Path(pointer.read_text().strip())

def show_motion(run, checkpoint="final"):
    run_module("bdxr_classroom.evaluate", run / "checkpoints" / (checkpoint + ".pt"), "--episodes", 1)
    report = json.loads((run / "evaluation" / ("eval_" + checkpoint) / "evaluation.json").read_text())
    display(Video(report["video_path"], embed=True, width=640))
''')
    md("""## 3 · Train the command-walking baseline

Simulated robots practice forward/backward speeds, stops, and turns. Many environments collect experience in parallel.

Predict: what would a robot do if speed tracking mattered but turning error barely mattered?
""")
    code('''BASELINE = train("command_walking_baseline",
                 "--command-mode", "balanced", "--yaw-tracking", "rate", "--yaw-std", 0.2,
                 "--steps", TRAIN_STEPS, "--minutes", TRAIN_MINUTES)
print("Your experiment:", BASELINE)
print("Before learning:")
show_motion(BASELINE, "initial")
print("After learning:")
show_motion(BASELINE)
run_module("bdxr_classroom.plots", BASELINE)
display(Image(filename=str(BASELINE / "metrics/learning_curves.png")))
''')
    md("""## 4 · Check what actually works

We test the same policy at several speeds and turn rates. The plots show measured speed/yaw rate and world trajectories.

**Position and heading are evaluation measurements, not actor inputs.** Check drift as well as falls. Repeated starting states test one trained policy; they are not independent training runs.
""")
    code('''run_module("bdxr_classroom.command_evaluate", BASELINE / "checkpoints/final.pt",
           "--cases", "slow", "forward", "fast", "backward", "stop", "turn_left", "turn_right", "left", "right", "sequence",
           "--episodes", 5, "--seconds", 15)
for case in ["forward", "sequence"]:
    display(Video(str(BASELINE / "videos/final_commands" / (case + ".mp4")), embed=True, width=640))
    display(Image(filename=str(BASELINE / "evaluation/final_commands" / (case + ".png"))))
print("Measured timing:", json.loads((BASELINE / "metrics/timing.json").read_text()))
''')
    md("""## 5 · Change one thing

Commanding the robot to walk faster is different from training a new style.
Choose a reward-setting experiment and fine-tune from the same baseline.

| Experiment | Only changed behavior setting |
|---|---|
| Smoother | Action-change penalty ×3 |
| Higher steps | Swing-foot target from 6 to 9 cm |
| Control | No changed setting; same extra training |

Keep the speed and yaw command range. A prettier gait that loses backward walking is not an unqualified improvement.
The unchanged control receives the same extra training budget, so we can distinguish a reward change from simply training longer.
""")
    code('''EXPERIMENT = "smoother"  # @param ["smoother", "higher_steps", "control"]
PREDICTION = "The motor commands will change less abruptly."  # @param {type:"string"}
VARIATION_STEPS = 3000  # @param {type:"integer"}
changed = ["--smoothness", 3] if EXPERIMENT == "smoother" else (["--swing-height", 0.09] if EXPERIMENT == "higher_steps" else [])
example = WORK / "assets/lesson/style_comparison"
print(json.loads((example / "provenance.json").read_text())["caption"])
display(Video(str(example / "gyro_style_comparison.mp4"), embed=True, width=720))
display(Image(filename=str(example / "gyro_style_comparison.png")))
print("Prediction:", PREDICTION)
VARIATION = train(EXPERIMENT, "--command-mode", "balanced", "--yaw-tracking", "rate",
                  "--yaw-std", 0.2, "--steps", VARIATION_STEPS, "--minutes", 10,
                  "--resume", BASELINE / "checkpoints/final.pt", "--finetune", *changed)
CONTROL = VARIATION if EXPERIMENT == "control" else train(
    "unchanged_control", "--command-mode", "balanced", "--yaw-tracking", "rate",
    "--yaw-std", 0.2, "--steps", VARIATION_STEPS, "--minutes", 10,
    "--resume", BASELINE / "checkpoints/final.pt", "--finetune")
from statistics import mean
comparisons = [("control", CONTROL)] + ([] if CONTROL == VARIATION else [(EXPERIMENT, VARIATION)])
for label, experiment in comparisons:
    run_module("bdxr_classroom.command_evaluate", experiment / "checkpoints/final.pt",
               "--cases", "slow", "forward", "fast", "backward", "stop", "turn_left", "turn_right", "left", "right", "sequence",
               "--episodes", 5, "--seconds", 15)
    report = json.loads((experiment / "evaluation/final_commands/summary.json").read_text())
    forward = report["cases"]["forward"]
    updates = json.loads((experiment / "metrics/timing.json").read_text())["step"]
    print(f"{label}: {updates} updates | action-change RMS = {mean(r['action_change_rms'] for r in forward):.3f}"
          f" | foot swing = {mean(r['swing_height_p95'] for r in forward):.3f} m"
          f" | cross-track = {mean(r['max_cross_track_m'] for r in forward):.3f} m")
    display(Video(str(experiment / "videos/final_commands/sequence.mp4"), embed=True, width=640))
''')
    md("""## 6 · Teach small hops

Hopping adds a rhythm command and rewards foot clearance and landing. Feet rising is not the same as the body jumping that distance.

This is an extended lab: first learn small stationary hops, then practice moving hops. Inspect both stages. Flat-floor success does not prove the robot can clear the game obstacles.
""")
    code('''TRAIN_HOPPING = True  # @param {type:"boolean"}
TINY_HOPS = HOP_RESULTS = None
if TRAIN_HOPPING:
    TINY_HOPS = train("tiny_hops", "--task-version", "hop", "--speed", 0,
                     "--hop-height", 0.04, "--steps", 4000, "--minutes", 10)
    HOP_RESULTS = train("moving_hops", "--task-version", "hop", "--speed", 0,
                        "--command-mode", "balanced", "--speed-min", -0.1, "--speed-max", 0.3,
                        "--yaw-max", 0.3, "--yaw-tracking", "rate", "--yaw-std", 0.2,
                        "--hop-height", 0.07, "--hop-body-weight", 6, "--linear-weight", 6,
                        "--steps", 6000, "--minutes", 15,
                        "--resume", TINY_HOPS / "checkpoints/final.pt", "--finetune")
    show_motion(HOP_RESULTS)
else:
    print("Hopping training skipped. You can load an existing compatible checkpoint in the game.")
''')
    md("""## 7 · Explain your results

1. Which commands work, and which fail?
2. Does zero yaw produce straight walking? Point to the trajectory.
3. Did your parameter change do what you predicted?
4. Why might a hopping policy fail when switched in mid-stride?

**Kinematics:** joint configurations and their trajectories describe motion.
**Dynamics:** forces, gravity, and contacts determine whether that motion works.
**SAC:** the actor learns actions, critics estimate future value, and entropy encourages exploration.

Save your recordings and checkpoint before disconnecting the runtime.
""")
    code('''def export_results():
    destination = WORK / "my_bdxr_results.zip"
    folders = list(dict.fromkeys([BASELINE, VARIATION, CONTROL] +
                                 [run for run in [TINY_HOPS, HOP_RESULTS] if run]))
    folders += list((WORK / "experiments").glob("*_desert_droid_game"))
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in folders:
            for file in folder.rglob("*"):
                if file.is_file() and "wandb" not in file.parts and file.name != "resume.pt":
                    archive.write(file, file.relative_to(WORK))
    print("Saved:", destination)
    try:
        from google.colab import files
        files.download(str(destination))
    except ImportError:
        print("Retrieve this ZIP locally.")
    return destination

export_results()
print("Full replay checkpoints remain in experiments/*/checkpoints/resume.pt; copy them to persistent storage to resume training.")
''')
    md("""## 8 · Play: Desert Droid Run

Load your learned controller into a desert outpost with **three obstacles, rough ground, and a raised finish beacon**. The course is designed around roughly 45 seconds of normal-paced traversal; policy ability and player skill determine the actual result.

Use speed and yaw controls to steer. If you loaded a hopping policy, switch modes before obstacles and watch the transition state. There is no navigation controller.

**Fall → return to the start. Time and falls accumulate. Reach the finish upright → win.** The scoreboard measures simulation time. A new game resets the score.
""")
    code('''# @title Launch your playable validation course
WALK_CHECKPOINT = str(BASELINE / "checkpoints/final.pt")  # @param {type:"string"}
HOP_CHECKPOINT = str(HOP_RESULTS / "checkpoints/final.pt") if HOP_RESULTS else ""  # @param {type:"string"}
GAME_PORT = 8765
if "game_process" in globals() and game_process.poll() is None:
    game_process.terminate()
    game_process.wait(timeout=30)
command = [PYTHON, "-u", "-m", "bdxr_classroom.game_server",
           "--walking", WALK_CHECKPOINT, "--port", str(GAME_PORT)]
if HOP_CHECKPOINT:
    command += ["--hopping", HOP_CHECKPOINT]
game_log = (WORK / "game_server.log").open("w")
game_process = subprocess.Popen(command, cwd=WORK, env=runtime_env,
                                stdout=game_log, stderr=subprocess.STDOUT)
import urllib.request
for _ in range(120):
    if game_process.poll() is not None:
        raise RuntimeError((WORK / "game_server.log").read_text()[-4000:])
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{GAME_PORT}/api/state", timeout=1) as response:
            status = json.load(response)
        if status.get("ready"):
            break
        if status.get("error"):
            raise RuntimeError(status["error"])
    except OSError:
        pass
    time.sleep(0.5)
else:
    raise RuntimeError("Game initialization timed out; inspect game_server.log")
try:
    from google.colab import output
    output.serve_kernel_port_as_iframe(GAME_PORT, height=950)
except ImportError:
    display(IFrame(f"http://127.0.0.1:{GAME_PORT}", width="100%", height=950))
print("Game logs and recordings:", status["experiment"])
print("After playing, pause the game and run export_results() again to include the gameplay files.")
''')
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": "BDXR_Walking_101.ipynb"}})
    nbf.validate(nb)
    dest = ROOT / "notebooks/BDXR_Walking_101.ipynb"
    nbf.write(nb, dest)
    print(f"Wrote {dest}: {len(cells)} cells, {dest.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
