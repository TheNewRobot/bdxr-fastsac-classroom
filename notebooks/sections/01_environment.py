# %% [markdown]
# # Teach a little droid to walk
# **163A / 263A · Robot kinematics and reinforcement learning**
#
# Build a simulation, inspect the robot's joints, train a walking controller,
# then change one reward and see how its gait changes.
#
# **Today's questions:** What does the robot observe? What does its action mean?
# How does a reward influence movement? How do we check whether it learned?
#
# **Our route:** environment → kinematics → walking problem → SAC training
# → a short walking check → one reward experiment → drive your droid.
#
# Examples are embedded in the text. Executable cells produce your own results.
# Local timings are not Colab guarantees; we will print your actual steps and times.

# %% [markdown]
# ## 1 · Meet the robot and create the environment
#
# The real-world demonstration is Disney's BDX droid. Our classroom uses the
# separately developed **BDX-R open-source model**, simulated with **MuJoCo Warp**
# through mjlab. We train with **FastSAC**; this is not Disney's controller.
#

# %% [markdown]
# | Real-world inspiration | Our simulated walker |
# |:---:|:---:|
# | ![Disney BDX droid — D23 reference photograph](lesson-asset:reference/disney_bdx.jpg) | ![Recorded instructor walking example](lesson-asset:classroom_walking/baseline.gif) |
#
# Head gestures are cosmetic animation; leg motion comes from simulation.
#
# Photo: [D23 / Disney](https://d23.com/bdx-droids-disneys-perfect-marriage-of-creativity-and-technological-achievement/), not relicensed. Simulation: the open-source BDX-R model with our recorded walking policy.

# %% [markdown]
# **Before running anything:** which joints do you expect to be most important
# for moving a foot forward? Which might help keep the body upright?
#
# Credits: [BDX-R model](https://github.com/BDX-R/BDX-R-MjLab),
# [FastSAC project](https://younggyo.me/fastsac-humanoid/),
# [Holosoma](https://github.com/amazon-far/holosoma), and
# [mjlab](https://github.com/mujocolab/mjlab). Pinned revisions and license notices
# are included with the lesson files.
# %% [markdown]
# ### Install and configure the environment
#
# Select **Runtime → Change runtime type → L4 GPU**. **L4 is required for class.**
# Setup is separate from training.
# An environment defines the robot, ground, actuators, observations, rewards,
# and episode endings. We first inspect one robot, then train many copies.

# %% [hidden]
# @title Unpack the lesson files
import hashlib
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

# Download a small helper and verify it before running any downloaded code.
BUNDLE_URL = "__BUNDLE_URL__"
BUNDLE_SHA256 = "__BUNDLE_SHA256__"
BOOTSTRAP_SHA256 = "__BOOTSTRAP_SHA256__"
WORK = (
    Path("/content/bdxr_classroom") if Path("/content").exists()
    else Path.cwd() / "bdxr_classroom_lab"
) / BUNDLE_SHA256[:12]
helper = Path(tempfile.mkdtemp(prefix="bdxr_setup_")) / "lesson_download.py"
fetch = "from urllib.request import urlretrieve; import sys; urlretrieve(*sys.argv[1:])"
subprocess.run(
    [sys.executable, "-c", fetch, BUNDLE_URL.rsplit("/", 1)[0] + "/lesson_download.py", str(helper)],
    check=True, timeout=60,
)
if hashlib.sha256(helper.read_bytes()).hexdigest() != BOOTSTRAP_SHA256:
    raise RuntimeError("Setup helper verification failed. Rerun this cell.")
runpy.run_path(str(helper))["prepare_lesson"](BUNDLE_URL, BUNDLE_SHA256, WORK)

# %% [hidden]
# @title Install pinned dependencies
if "WORK" not in globals() or not (WORK / "scripts/setup/setup.py").is_file():
    raise RuntimeError("Run the unpack cell first.")

from bdxr_classroom.lesson.setup import install_runtime

PYTHON, runtime_env = install_runtime(WORK)

# %% [hidden]
# @title Initialize the notebook's simulator connection
import json
import os
from bdxr_classroom.lesson.presentation import media_pair, read_json
from bdxr_classroom.lesson.runtime import Lesson
from bdxr_classroom.lesson.simulator import SimulatorCells

if "simulator" in globals():
    simulator.close()
simulator = SimulatorCells(PYTHON, WORK, runtime_env)
simulator.register(get_ipython())
lab = Lesson(WORK, PYTHON, runtime_env)
(WORK / "figures").mkdir(exist_ok=True)

# %% [markdown]
# ### Define the walking task
#
# These settings define the classroom walking task. Positive forward speed asks for
# forward motion; negative speed asks for backward motion. Positive yaw rate
# requests a left turn. Zero yaw requests no rotation, without a compass or
# an external heading-correction controller.
#
# The yaw- and lateral-bias weights discourage **sustained** turning error
# and sideways drift. They do not add a heading-correction controller.

# | Environment setting | Meaning |
# |---|---|
# | Physics step: 5 ms | Integrate forces and contacts |
# | Control step: 20 ms | Ask the actor for new joint targets |
# | Episode length: 10 s | Maximum duration before another practice reset |
# | Speed range: −0.15 to +0.35 m/s | Reverse through brisk forward walking |
# | Turn-rate range: ±0.45 rad/s | Right and left turns; zero requests straight motion |
#
# `balanced` samples stopping, straight movement, turning, and combined commands.
# Small reset perturbations expose the learner to more than one starting posture.
# These define the task; they are separate from SAC's network and replay settings.

# %%
ENV_CONFIG = {
    "task_version": "v2",  # Ten leg-joint targets; the complete robot remains physical.
    "command_mode": "balanced",  # Practice forward/backward movement, stopping, and turns.
    "speed_min": -0.15,  # m/s, backward command limit.
    "speed_max": 0.35,  # m/s, forward command limit.
    "yaw_max": 0.45,  # rad/s, magnitude of the turning command.
    "yaw_tracking": "imu",  # Estimate turning rate from gyro and gravity direction.
    "yaw_std": 0.20,  # rad/s, width of the turning reward.
    "yaw_bias_weight": 10.0,  # Weight for reducing sustained turning error.
    "lateral_bias_weight": 2.0,  # Weight for reducing sustained sideways drift.
    "initial_jitter": 0.03,  # Radians: practice small reset perturbations.
}
(WORK / "environment_settings.json").write_text(json.dumps(ENV_CONFIG, indent=2))

# %% [markdown]
# Cells marked `%%robot` run visible Python in the pinned simulator environment.
# Variables persist between these cells. Ordinary cells display figures and
# organize experiments in the notebook.

# %%
%%robot
import json
from pathlib import Path
from bdxr_classroom.simulation.environment import make_env
from bdxr_classroom.lesson.simulator import setup_log

settings = json.loads(Path("environment_settings.json").read_text())
with setup_log("artifacts/logs/environment_setup.log"):
    env = make_env(num_envs=1, seed=7, render=True, **settings)
observations, _ = env.reset(seed=7)
print("Physics timestep:", env.cfg.sim.mujoco.timestep, "seconds")
print("Policy/control timestep:", env.step_dt, "seconds")
print("Episode time limit:", env.cfg.episode_length_s, "seconds")

# %%
%%robot
from bdxr_classroom.lesson.kinematics import save_pose

save_pose(env, "figures/robot.png")
print("Robot joints:", env.scene["robot"].joint_names)

# %%
from IPython.display import display, Image

display(Image(filename=str(WORK / "figures/robot.png"), width=720))

# %% [markdown]
# The simulator advances physics every **5 ms**; the policy chooses new targets
# every **20 ms**. Training episodes last up to **10 s** and also end on the
# configured fall conditions. A reset starts another attempt.
#
# **Check your understanding:** why can physics need more updates than the policy?
