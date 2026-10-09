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
        cell = nbf.v4.new_markdown_cell(text)
        cell.id = f"lesson-{len(cells):02d}"
        cells.append(cell)
    def code(text, hidden=False):
        cell = nbf.v4.new_code_cell(text)
        cell.id = f"lesson-{len(cells):02d}"
        if hidden:
            cell.metadata["cellView"] = "form"
        cells.append(cell)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        files = set()
        for pattern in ["src/**/*.py", "src/**/*.html", "configs/*.json",
                        "requirements*.txt", "scripts/setup.py", "scripts/check_command_results.py",
                        "scripts/check_hopping_results.py", "scripts/measure_gait.py",
                        "scripts/validate_transitions.py", "scripts/validate_policy_pair.py",
                        "THIRD_PARTY_NOTICES.md",
                        "assets/lesson/**/*"]:
            for path in ROOT.glob(pattern):
                if path.is_file():
                    files.add(path)
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
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
    wandb_link = (f'<p><a href="{html.escape(item["wandb_gallery"], quote=True)}" target="_blank">'
                  'Open the recorded W&B run</a></p>') if item.get("wandb_gallery") else ""
    destination = legacy_cards if item["video"].startswith("legacy/") else cards
    destination.append(f'<article style="flex:1 1 320px;max-width:480px;padding:12px;border:1px solid #b9cebe;border-radius:12px">'
                 f'<h4>{html.escape(item["title"])}</h4><video controls loop playsinline width="100%" src="data:video/mp4;base64,{video}"></video>'
                 f'<p>{html.escape(item["caption"])}</p>{wandb_link}<details><summary>Measured curves and motion</summary>{plots}</details></article>')
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

The critic evaluates a proposed action, not just the current pose. This
implementation uses two Q estimates to help limit overly optimistic predictions.
The actor learns to choose actions that the critics rate highly.

During training, the actor samples actions. An **entropy bonus** encourages it
to keep exploring instead of always repeating one choice. During evaluation
and gameplay, we use its deterministic action without that sampling noise.

Simulator velocity scores speed tracking and sustained sideways drift during
training. Those measurements do not become actor inputs.

We train 512 simulated robots in parallel. One training step advances every robot,
collecting 512 experiences. Replaying those experiences produces separate neural-network
updates. The learning-curve horizontal axis counts experiences, not seconds.

**Kinematics connection:** hip, knee, and ankle angles determine where each foot
is relative to the body. The policy chooses joint targets; the simulator then
determines the resulting motion and contacts. A foot-height reward asks for an
outcome—it does not directly prescribe the joint angles or guarantee a gait.
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
COMMAND_OPTIONS = ["--command-mode", "balanced", "--yaw-tracking", "imu",
                   "--yaw-std", 0.2, "--yaw-bias-weight", 6,
                   "--lateral-bias-weight", 2, "--mirror-replay"]
if USE_WANDB and not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
    import getpass
    runtime_env["WANDB_API_KEY"] = getpass.getpass("Your W&B API key (not saved in this notebook): ")

import sys
sys.path.insert(0, str(WORK / "scripts"))
from check_command_results import checkpoint_digest, select_checkpoint

def run_module(module, *arguments):
    command = [PYTHON, "-u", "-m", module] + [str(value) for value in arguments]
    log_path = WORK / "last_command.log"
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(command, cwd=WORK, env=runtime_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                if line.startswith(("step=", "Experiment:", "forward:", "backward:", "sequence:")) or "Error" in line or "View run" in line:
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

def show_motion(run, checkpoint="final", speed=None):
    suffix = "" if speed is None else f"_speed_{speed:g}"
    output = run / "evaluation" / ("eval_" + checkpoint + suffix)
    options = [] if speed is None else ["--speed", speed]
    run_module("bdxr_classroom.evaluate", run / "checkpoints" / (checkpoint + ".pt"),
               "--episodes", 1, "--output", output, *options)
    report = json.loads((output / "evaluation.json").read_text())
    display(Video(report["video_path"], embed=True, width=640))

def summarize_commands(run, expected_digest):
    import sys
    from statistics import mean
    sys.path.insert(0, str(WORK / "scripts"))
    from check_command_results import check
    report = json.loads((run / "evaluation/final_commands/summary.json").read_text())
    targets = json.loads((WORK / "configs/evaluation_targets.json").read_text())
    verdict = check(report, targets)
    checkpoint = run / "checkpoints/final.pt"
    if checkpoint_digest(checkpoint) != expected_digest:
        raise RuntimeError("Checkpoint changed during evaluation; evaluate a stable saved file.")
    verdict.update(checkpoint=str(checkpoint), checkpoint_sha256=expected_digest)
    (run / "evaluation/final_commands/selection_check.json").write_text(json.dumps(verdict, indent=2))
    rows = []
    for case, trials in report["cases"].items():
        rows.append(f"<tr><td>{html.escape(case)}</td><td>{sum(t['fell'] for t in trials)}/{len(trials)}</td>"
                    f"<td>{mean(t['speed_mae'] for t in trials):.3f}</td>"
                    f"<td>{mean(t['yaw_mae'] for t in trials):.3f}</td>"
                    f"<td>{mean(t['max_cross_track_m'] for t in trials):.3f}</td></tr>")
    display(HTML("<table><tr><th>Command</th><th>Falls</th><th>Speed error (m/s)</th>"
                 "<th>Turn-rate error (rad/s)</th><th>Mean max lateral deviation (m)</th></tr>"
                 + "".join(rows) + "</table>"))
    print("Passed the declared command checks." if verdict["passed"] else
          "Needs more work: " + "; ".join(verdict["failures"]))
    return verdict
''')
    md("""## 3 · Train the command-walking baseline

Simulated robots practice forward/backward speeds, stops, and turns. Many environments collect experience in parallel.

The training recipe also mirrors left/right experience and rewards low sustained turning error and sideways drift.
Turning rate is estimated from the gyro and gravity direction; there is no compass or absolute-heading input.

Predict: what would a robot do if speed tracking mattered but turning error barely mattered?
""")
    code('''BASELINE = train("command_walking_baseline",
                 *COMMAND_OPTIONS,
                 "--steps", TRAIN_STEPS, "--minutes", TRAIN_MINUTES)
INITIAL_BASELINE = BASELINE
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

Lateral deviation is a straight-line check only. A curved path is expected when the command asks for a turn.

If the first checkpoint misses the checks, try one unchanged continuation.
Style experiments start only after the walking baseline passes.
""")
    code('''def evaluate_walking_run(run):
    digest_before = checkpoint_digest(run / "checkpoints/final.pt")
    run_module("bdxr_classroom.command_evaluate", run / "checkpoints/final.pt",
               "--cases", "slow", "forward", "fast", "backward", "stop",
               "turn_left", "turn_right", "left", "right", "sequence",
               "--episodes", 5, "--seconds", 15)
    return summarize_commands(run, digest_before)

BASELINE_CHECK = evaluate_walking_run(BASELINE)
if not BASELINE_CHECK["passed"]:
    print("The first checkpoint needs more practice. Continuing with unchanged behavior settings.")
    BASELINE = train("command_walking_recovery", *COMMAND_OPTIONS,
                     "--steps", 3000, "--minutes", 10,
                     "--resume", BASELINE / "checkpoints/final.pt", "--finetune")
    BASELINE_CHECK = evaluate_walking_run(BASELINE)
if not BASELINE_CHECK["passed"]:
    raise RuntimeError("The walking baseline still needs work. Inspect its reports before starting style experiments.")
if BASELINE != INITIAL_BASELINE:
    run_module("bdxr_classroom.plots", BASELINE)
    display(Image(filename=str(BASELINE / "metrics/learning_curves.png")))
for case in ["forward", "sequence"]:
    display(Video(str(BASELINE / "videos/final_commands" / (case + ".mp4")), embed=True, width=640))
    display(Image(filename=str(BASELINE / "evaluation/final_commands" / (case + ".png"))))
print("Measured timing:", json.loads((BASELINE / "metrics/timing.json").read_text()))
''')
    md("""## 5 · Change one thing

Commanding the robot to walk faster is different from training a new style.
Choose a reward-setting experiment and train a fresh policy from scratch.
The control starts from the same random seed and uses the same training budget.

| Experiment | Only changed behavior setting |
|---|---|
| Higher target | Swing-foot marker target from 6 to 10 cm |
| Quick steps | Reward airborne durations between 0.06 and 0.16 s, weight 4 |
| Level feet | Add a reward for keeping the soles level, weight 1.5 |
| Smoother | Action-change penalty ×5 |
| Control | No changed setting |

Keep the speed and yaw command range. A prettier gait that loses backward walking is not an unqualified improvement.
These are experiment hypotheses, not a promise that every setting passes.
Predict the result, then check falls, command tracking, cadence, sole clearance,
and foot angle. Changing reward weights changes the score itself, so a higher
return alone cannot establish a better gait.

Each run has its own time budget; running a variant and its control takes two
training runs. If either reaches the time limit early, their step counts may
differ: report that limitation before attributing a difference to the reward.
One successful seed is an example, not proof of reliability across seeds.

The recorded comparison below is an earlier fine-tuning demonstration, not
the result of the new scratch-training experiment. It also illustrates that
raising a reward target need not raise the feet.
""")
    code('''EXPERIMENT = "higher_target"  # @param ["higher_target", "quick_steps", "level_feet", "smoother", "control"]
PREDICTION = "A higher target will lift the feet farther."  # @param {type:"string"}
VARIATION_STEPS = 10000  # @param {type:"integer"}
VARIATION_MINUTES = 15  # @param {type:"number"}
changes = {
    "higher_target": ["--swing-height", 0.10],
    "quick_steps": ["--air-time-min", 0.06, "--air-time-max", 0.16, "--air-time-weight", 4],
    "level_feet": ["--level-feet-weight", 1.5],
    "smoother": ["--smoothness", 5],
    "control": [],
}
changed = changes[EXPERIMENT]
example = WORK / "assets/lesson/style_comparison"
style_provenance = json.loads((example / "provenance.json").read_text())
print(style_provenance["caption"])
display(HTML(f'<a href="{html.escape(style_provenance["wandb_gallery"], quote=True)}" target="_blank">Open the recorded W&B comparison</a>'))
display(Video(str(example / "walking_style_comparison.mp4"), embed=True, width=720))
display(Image(filename=str(example / "walking_style_comparison.png")))
print("Prediction:", PREDICTION)
VARIATION = train(EXPERIMENT, *COMMAND_OPTIONS,
                  "--steps", VARIATION_STEPS, "--minutes", VARIATION_MINUTES, *changed)
CONTROL = VARIATION if EXPERIMENT == "control" else train(
    "scratch_control", *COMMAND_OPTIONS,
    "--steps", VARIATION_STEPS, "--minutes", VARIATION_MINUTES)
from statistics import mean
comparisons = [("control", CONTROL)] + ([] if CONTROL == VARIATION else [(EXPERIMENT, VARIATION)])
walking_candidates = [BASELINE_CHECK]
for label, experiment in comparisons:
    digest_before = checkpoint_digest(experiment / "checkpoints/final.pt")
    run_module("bdxr_classroom.command_evaluate", experiment / "checkpoints/final.pt",
               "--cases", "slow", "forward", "fast", "backward", "stop", "turn_left", "turn_right", "left", "right", "sequence",
               "--episodes", 5, "--seconds", 15)
    report = json.loads((experiment / "evaluation/final_commands/summary.json").read_text())
    walking_candidates.append(summarize_commands(experiment, digest_before))
    forward = report["cases"]["forward"]
    steps = json.loads((experiment / "metrics/timing.json").read_text())["step"]
    subprocess.run([PYTHON, str(WORK / "scripts/measure_gait.py"),
                    str(experiment / "evaluation/final_commands")],
                   cwd=WORK, env=runtime_env, check=True)
    gait = json.loads((experiment / "evaluation/final_commands/gait_measurements.json").read_text())["forward"]["trials"]
    tilt_values = [r["swing_sole_tilt_deg"] for r in gait if r["swing_sole_tilt_deg"] is not None]
    tilt_text = f"{mean(tilt_values):.1f} degrees" if tilt_values else "no measured swing"
    if gait:
        print(f"{label}: {steps} parallel simulation steps | cadence = {mean(r['foot_landings_per_second'] for r in gait):.2f} foot landings/s"
              f" | sole clearance = {mean(r['sole_clearance_p95_m'] for r in gait):.3f} m"
              f" | swing-foot tilt = {tilt_text}"
              f" | cross-track = {mean(r['max_cross_track_m'] for r in forward):.3f} m")
    else:
        print(label, "ended too early for gait measurements; inspect the failed command checks.")
    run_module("bdxr_classroom.plots", experiment)
    display(Image(filename=str(experiment / "metrics/learning_curves.png")))
    display(Video(str(experiment / "videos/final_commands/sequence.mp4"), embed=True, width=640))
SELECTED_WALK_CHECKPOINT = select_checkpoint(walking_candidates)
WALKING_VALIDATED = SELECTED_WALK_CHECKPOINT is not None
if WALKING_VALIDATED:
    print("Walking checkpoint selected for the game:", SELECTED_WALK_CHECKPOINT)
else:
    SELECTED_WALK_CHECKPOINT = str(BASELINE / "checkpoints/final.pt")
    print("No walking checkpoint passed. The baseline remains available for exploratory play, not as a validated controller.")
''')
    md("""## 6 · Teach small hops

Hopping adds a rhythm command and rewards foot clearance and landing. Feet rising is not the same as the body jumping that distance.

A marker above the foot can rise while the heel stays close to the floor.
Check the lowest sole clearance and contact state before claiming that a hop can clear an obstacle.

This extended lab uses a curriculum: stationary hops → moving hops → clearance over small physical obstacles.
Walking and hopping have separate time budgets. The recorded higher, level-foot
hopper used about 19.5 minutes of training across six stages on a local L40S,
plus setup and evaluation. This is instructor timing, not a Colab guarantee.

For a hop, measure both feet: one raised foot can be a step. We use joint
kinematics and the simulated foot geometry to measure sole clearance and tilt.
The instructor comparison counts clear flights lasting at least 80 ms with
both soles reaching at least 2.5 cm above the ground. Check the video too:
more height can also mean harder landings.

The first terrain stage requests 4.5 cm sole clearance and practices on
25 mm terrain from varied starting positions. This exposes the policy to more
foot placements; it does not give the actor a terrain map. Inspect each stage;
flat-floor success does not prove obstacle capability. The optional natural-hop
extension then requests 6.5 cm bilateral clearance, restores terrain practice,
and rewards level feet during flight so the ankles remain free during push-off.

Compare the final checkpoint with saved earlier checkpoints. Each must
complete five trials per command without falling, land at least five hops per trial,
and reach at least 2.5 cm sole clearance. Forward commands must move forward;
stop must stay slow. The exact screen is saved with the results. Passing it does
not replace tests of turning, mode switches, or the full course.

Repeated tests can differ. Keep every result: a later success does not erase
an earlier failure. The notebook matches reports to the exact saved weights
and selects only checkpoints that passed all their recorded screens.

Before the game, test repeated walk/hop switches on flat ground and on a small
step. Passing the hopping screen alone is not enough. A failed pair is not
loaded automatically; its videos and reports stay in your results.
""")
    code('''TRAIN_HOPPING = True  # @param {type:"boolean"}
NATURAL_HOPS = True  # @param {type:"boolean"}
TINY_HOPS = MOVING_HOPS = HOP_RESULTS = None
HOP_STAGES = []
SELECTED_HOP_CHECKPOINT = None
if TRAIN_HOPPING:
    TINY_HOPS = train("tiny_hops", "--task-version", "hop", "--speed", 0,
                     "--hop-height", 0.04, "--steps", 4000, "--minutes", 10)
    show_motion(TINY_HOPS)
    MOVING_HOPS = train("moving_hops", "--task-version", "hop", "--speed", 0,
                        "--command-mode", "balanced", "--speed-min", -0.1, "--speed-max", 0.3,
                        "--yaw-max", 0.3, "--yaw-tracking", "rate", "--yaw-std", 0.2,
                        "--hop-height", 0.07, "--hop-body-weight", 6, "--linear-weight", 6,
                        "--steps", 6000, "--minutes", 15,
                        "--resume", TINY_HOPS / "checkpoints/final.pt", "--finetune")
    show_motion(MOVING_HOPS, speed=0.15)
    HOP_RESULTS = train("terrain_hops", "--task-version", "hop", "--speed", 0.3,
                        "--command-mode", "balanced", "--speed-min", -0.1, "--speed-max", 0.3,
                        "--yaw-max", 0.3, "--yaw-tracking", "rate", "--yaw-std", 0.2,
                        "--hop-height", 0.045, "--hop-body-weight", 6, "--linear-weight", 6,
                        "--hop-clearance", "sole", "--hop-height-reward", "flight", "--mirror-replay",
                        "--training-terrain-height", 0.025, "--training-terrain-style", "mixed",
                        "--training-start-span", 1.5,
                        "--steps", 2000, "--minutes", 10,
                        "--resume", MOVING_HOPS / "checkpoints/final.pt", "--finetune")
    HOP_STAGES = [TINY_HOPS, MOVING_HOPS, HOP_RESULTS]
    if NATURAL_HOPS:
        NATURAL_OPTIONS = [
            "--task-version", "hop", "--command-mode", "balanced",
            "--speed-min", -0.1, "--speed-max", 0.3, "--yaw-max", 0.3,
            "--yaw-tracking", "rate", "--yaw-std", 0.2,
            "--hop-height", 0.065, "--hop-body-weight", 6, "--linear-weight", 6,
            "--hop-clearance", "sole", "--hop-height-reward", "flight_both",
            "--level-feet-weight", 3, "--mirror-replay",
        ]
        LEVEL_HOPS = train("level_hops", *NATURAL_OPTIONS,
                           "--steps", 5000, "--minutes", 12,
                           "--resume", HOP_RESULTS / "checkpoints/final.pt", "--finetune")
        TERRAIN_OPTIONS = [
            "--training-terrain-height", 0.025, "--training-terrain-style", "mixed",
            "--training-start-span", 1.5,
        ]
        LEVEL_TERRAIN = train("level_terrain_hops", *NATURAL_OPTIONS, *TERRAIN_OPTIONS,
                              "--steps", 3000, "--minutes", 10,
                              "--resume", LEVEL_HOPS / "checkpoints/final.pt", "--finetune")
        HOP_RESULTS = train("natural_hops", *NATURAL_OPTIONS, *TERRAIN_OPTIONS,
                            "--level-feet-flight-only", "--steps", 1000, "--minutes", 6,
                            "--resume", LEVEL_TERRAIN / "checkpoints/final.pt", "--finetune")
        HOP_STAGES += [LEVEL_HOPS, LEVEL_TERRAIN, HOP_RESULTS]
    import sys, hashlib
    sys.path.insert(0, str(WORK / "scripts"))
    from check_hopping_results import choose
    from statistics import mean
    history = HOP_RESULTS / "evaluation/hopping_history"
    history.mkdir(exist_ok=True)
    current_checkpoints = {}
    final_step = json.loads((HOP_RESULTS / "metrics/timing.json").read_text())["step"]
    filenames = ["final.pt"] + [
        path.name for path in sorted((HOP_RESULTS / "checkpoints").glob("step_*.pt"), reverse=True)
        if int(path.stem.split("_")[1]) != final_step]
    for filename in filenames:
        checkpoint = HOP_RESULTS / "checkpoints" / filename
        if not checkpoint.exists():
            print("Checkpoint not reached before the training time limit:", filename)
            continue
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        current_checkpoints[str(checkpoint)] = digest
        folder = f"{checkpoint.stem}_commands_mixed_0.015_{time.time_ns()}"
        output = HOP_RESULTS / "evaluation" / folder
        run_module("bdxr_classroom.command_evaluate", checkpoint,
                   "--cases", "slow", "forward", "stop", "--episodes", 5, "--seconds", 15,
                   "--terrain-height", 0.015, "--terrain-style", "mixed", "--output", output)
        with checkpoint.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise RuntimeError("Checkpoint changed during evaluation; retry with stable saved weights.")
        report = json.loads((output / "summary.json").read_text())
        report.update(checkpoint_sha256=digest, evaluation_directory=str(output),
                      evaluated_at_ns=time.time_ns())
        (output / "summary.json").write_text(json.dumps(report, indent=2))
        (history / (folder + ".json")).write_text(json.dumps(report, indent=2))
        for case, trials in report["cases"].items():
            print(f"{filename}, {case}: falls {sum(t['fell'] for t in trials)}/{len(trials)}, "
                     f"mean landed hops {mean(t['landed_hops'] for t in trials):.1f}, "
                     f"mean peak sole clearance {100*mean(t['peak_simultaneous_foot_clearance'] for t in trials):.1f} cm, "
                     f"mean speed {mean(t['mean_speed'] for t in trials):.2f} m/s")
    hop_reports = [json.loads(file.read_text()) for file in sorted(history.glob("*.json"))]
    hop_reports.sort(key=lambda report: report.get("evaluated_at_ns", 0))
    selection = choose(hop_reports, json.loads((WORK / "configs/hopping_targets.json").read_text()),
                       current_checkpoints)
    (HOP_RESULTS / "evaluation/checkpoint_selection.json").write_text(json.dumps(selection, indent=2))
    for result in selection["candidates"]:
        print(Path(result["checkpoint"]).name, "Passed" if result["passed"] else "Needs work",
              f"across {result['evaluation_count']} recorded evaluations:",
              "; ".join(result["failures"]))
    # Try passing candidates in order; a failed handover is not a usable pair.
    for candidate in selection["candidates"]:
        if not WALKING_VALIDATED:
            print("A passing walking checkpoint is needed before selecting a walking/hopping pair.")
            break
        if not candidate["passed"]:
            continue
        selected = Path(candidate["checkpoint"])
        print("Testing", selected.name, "with walking on flat ground and a 20 mm step...")
        pair_output = HOP_RESULTS / "evaluation"
        pair_log = pair_output / ("pair_check_" + str(time.time_ns()) + ".log")
        with pair_log.open("w") as log:
            checked = subprocess.run(
                [PYTHON, str(WORK / "scripts/validate_policy_pair.py"),
                 SELECTED_WALK_CHECKPOINT, str(selected),
                 "--output", str(pair_output)], cwd=WORK, env=runtime_env,
                stdout=log, stderr=subprocess.STDOUT)
        print(pair_log.read_text()[-3000:])
        if checked.returncode == 0:
            SELECTED_HOP_CHECKPOINT = str(selected)
            break
    if SELECTED_HOP_CHECKPOINT:
        selected = Path(SELECTED_HOP_CHECKPOINT)
        print("Selected for further testing in the game:", selected.name)
        show_motion(HOP_RESULTS, selected.stem)
        print("Body rise, sole clearance, and knee angles are different quantities.")
        display(Image(filename=str(HOP_RESULTS / "evaluation" / ("eval_" + selected.stem) / "motion.png")))
        latest = next(report for report in reversed(hop_reports)
                      if report["checkpoint_sha256"] == current_checkpoints[str(selected)])
        display(Video(str(HOP_RESULTS / "videos" / Path(latest["evaluation_directory"]).name / "slow.mp4"),
                      embed=True, width=640))
    else:
        print("No hopping pair passed all screens. The game will start with walking only; inspect the failures.")
    run_module("bdxr_classroom.plots", HOP_RESULTS)
    display(Image(filename=str(HOP_RESULTS / "metrics/learning_curves.png")))
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
    code('''INCLUDE_REPLAY_IN_EXPORT = True  # @param {type:"boolean"}
# Replay makes the backup larger, but preserves the collected training experience.
def export_results(include_replay=INCLUDE_REPLAY_IN_EXPORT):
    import time
    started = time.monotonic()
    destination = WORK / ("my_bdxr_training_backup.zip" if include_replay else "my_bdxr_results.zip")
    temporary = destination.with_suffix(".tmp.zip")
    folders = list(dict.fromkeys([INITIAL_BASELINE, BASELINE, VARIATION, CONTROL] +
                                 HOP_STAGES))
    folders += list((WORK / "experiments").glob("*_command_walking_recovery"))
    folders += list((WORK / "experiments").glob("*_desert_droid_game"))
    folders += list((WORK / "experiments").glob("*_transition_validation"))
    folders = list(dict.fromkeys(folders))
    print("Creating backup. Replay files can take several minutes to save.", flush=True)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, folder in enumerate(folders, 1):
            print(f"Saving experiment {index}/{len(folders)}: {folder.name}", flush=True)
            for file in folder.rglob("*"):
                if file.is_file() and "wandb" not in file.parts and (include_replay or file.name != "resume.pt"):
                    archive.write(file, file.relative_to(WORK))
    temporary.replace(destination)
    print(f"Saved in {time.monotonic() - started:.1f} seconds:", destination)
    try:
        from google.colab import files
        files.download(str(destination))
    except ImportError:
        print("Retrieve this ZIP locally.")
    return destination

export_results()
if INCLUDE_REPLAY_IN_EXPORT:
    print("To resume: unpack the backup into WORK, then train with --resume experiments/<run>/checkpoints/resume.pt.")
    print("Networks, optimizers, normalization and replay are restored; simulated episodes restart.")
else:
    print("Replay was omitted. To start a new fine-tuning run, unpack into WORK and use --resume experiments/<run>/checkpoints/final.pt --finetune.")
    print("This reuses trained networks with a new replay buffer, rather than restoring collected experience.")
''')
    md("""## 8 · Play: Desert Droid Run

Load your learned controller into a desert outpost with **three obstacles, rough ground, and a raised finish beacon**. The course is designed around roughly 45 seconds of normal-paced traversal; policy ability and player skill determine the actual result.

The classroom obstacles are 17.5–20 mm high, followed by uneven ground and an 18 cm ascent. These dimensions were calibrated using recorded controller trials.

Use speed and yaw controls to steer. If you loaded a hopping policy, switch modes before obstacles and watch the transition state. There is no navigation controller.

Wait until both feet visibly clear the obstacle and land before returning to
walking. Mode changes slow the robot down, so leave space for the handover.

The default walking checkpoint is the first passing baseline, control, or
variation. An unchanged continuation can improve a baseline that missed a check.

**Fall → return to the start. Time and falls accumulate. Cross the three obstacles and rough patch, then reach the finish upright → win.** The four checkpoints prevent shortcuts. The scoreboard measures simulation time. A new game resets the score.
""")
    code('''# @title Launch your playable validation course
# Optional paths: leave blank to use the checkpoints selected above.
WALK_CHECKPOINT = ""  # @param {type:"string"}
HOP_CHECKPOINT = ""  # @param {type:"string"}
WALK_CHECKPOINT = WALK_CHECKPOINT or SELECTED_WALK_CHECKPOINT
HOP_CHECKPOINT = HOP_CHECKPOINT or SELECTED_HOP_CHECKPOINT or ""
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
print("After playing, run the next cell to pause, watch your recording, and save the backup.")
''')
    code('''# @title After playing: watch and save your run
with urllib.request.urlopen(f"http://127.0.0.1:{GAME_PORT}/api/state", timeout=5) as response:
    saved_game = json.load(response)
if not saved_game.get("paused", True):
    request = urllib.request.Request(
        f"http://127.0.0.1:{GAME_PORT}/api/command",
        data=json.dumps({"action": "pause"}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=5).close()
    for _ in range(100):
        with urllib.request.urlopen(f"http://127.0.0.1:{GAME_PORT}/api/state", timeout=5) as response:
            saved_game = json.load(response)
        if saved_game.get("paused"):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Pause the game before exporting its recording.")
recordings = sorted((WORK / saved_game["experiment"] / "videos").glob("gameplay_*.mp4"))
for recording in recordings:
    display(Video(str(recording), embed=True, width=640))
if not recordings:
    print("No gameplay recorded yet. Drive the droid, then run this cell again.")
export_results()
''', hidden=True)
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
