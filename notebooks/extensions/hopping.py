# %% [markdown]
# # Teach a little droid to hop
# **Optional robotics lesson · BDX-R, FastSAC, and MuJoCo Warp**
#
# Train one hopping policy, watch its takeoff and landing, then click **Jump**
# in a small practice arena. No walking checkpoint or policy switching is needed.
#
# ## 1 · Meet the hopping task and set up
#
# | Instructor example · one requested hop, then settle |
# |:---:|
# | ![One hopper taking off and landing](lesson-asset:hopping_practice/jump.gif) |
#
# Look for **both feet leaving the ground** and a controlled landing. The GIF
# loops automatically; [open the playable example](lesson-asset:hopping_practice/jump.mp4)
# if it stays still. This is a recorded reference policy; your own results appear later.
# Choose a GPU runtime and run the setup cells below. Timings depend on your GPU.

# %% [setup]

# %% [markdown]
# ## 2 · Define a hop
#
# The actor receives body rotation, gravity direction, joint positions and
# velocities, previous actions, movement commands, and a **two-number phase cue**.
# That is **49 inputs**; the critic receives **64**, including training-only
# simulator measurements. True linear velocity, foot contacts and terrain are
# not inputs to the deployed actor.
#
# The **10 outputs** specify leg-joint targets. Position servos produce forces;
# the controller does not place the robot in midair.
#
# $$q_i^{\mathrm{target}} = q_i^{\mathrm{default}} + s_i a_i$$
# $$\mathrm{phase}(t) = [\sin(2\pi t),\;\cos(2\pi t)]$$
#
# During training the phase repeats each second. In the demo, Jump advances one
# cycle; between requests it holds the stance cue. **Physics and the same actor
# keep running** while the robot settles. This behavior must be tested, not assumed.
#
# Rewards encourage upright posture, both soles clearing the ground, level feet
# in flight, and landing at the right point in the cycle. Episodes also end on
# falls or low body height. A tall foot marker alone is not a successful hop.

# %%
import json
from bdxr_classroom.lesson.runtime import Lesson
from bdxr_classroom.lesson.simulator import SimulatorCells

lab = Lesson(WORK, PYTHON, runtime_env)
HOP_RECIPE = json.loads((WORK / "configs/stationary_hopping.json").read_text())
HOP_ENV = HOP_RECIPE["environment"]
if "simulator" in globals():
    simulator.close()
simulator = SimulatorCells(PYTHON, WORK, runtime_env)
simulator.register(get_ipython())

# %%
%%robot
import json
from pathlib import Path
from bdxr_classroom.simulation.environment import make_env
from bdxr_classroom.lesson.simulator import setup_log

with setup_log("artifacts/logs/hopping_environment.log"):
    recipe = json.loads((Path.cwd() / "configs/stationary_hopping.json").read_text())
    env = make_env(num_envs=1, **recipe["environment"])
observations, _ = env.reset(seed=7)
for group in ("actor", "critic"):
    print(group, "inputs:", observations[group].shape[-1])
print("Leg actions:", env.action_manager.total_action_dim)
print("Joint target scales:", env.cfg.actions["joint_pos"].scale)
print("Physics step:", env.cfg.sim.mujoco.timestep, "s; policy step:", env.step_dt, "s")
env.close()

# %% [markdown]
# ## 3 · Train with SAC
#
# SAC learns from replayed experience: an actor proposes actions and a critic
# estimates their return. Exploration helps it discover takeoff and landing.
# A rising reward curve alone does not prove that both feet leave the ground.
#
# **One run from scratch.** Keep the movement command at zero and learn hops on
# flat ground. No terrain curriculum or walking policy is required.
#
# | Objective | What it encourages |
# |---|---|
# | Whole-sole clearance | Both feet lift toward a 5 cm target |
# | Flight bonus | Actual bilateral flight during the requested phase |
# | Body height and level feet | Coordinated takeoff and controlled landing |
#
# One fresh **6,000-step** run took **9.9 minutes on a shared local L40S**
# (10.1 minutes including startup). Its three recorded hops cleared 5.6–5.7 cm
# and settled without idle foot lifts. This is one seed, not a Colab timing or
# success guarantee. The earlier eight-stage recipe failed this same check.

# %%
from bdxr_classroom.lesson.presentation import table

table([
    ("Sole-clearance target (m)", HOP_ENV["hop_height"]),
    ("Bilateral-flight bonus", HOP_ENV["hop_flight_weight"]),
    ("Level-feet reward", HOP_ENV["level_feet_weight"]),
    ("Body-lift reward", HOP_ENV["hop_body_weight"]),
    ("Overall reward scale", HOP_ENV["reward_scale"]),
], ["Reward parameter", "Value"])
LEARNING_CONFIG = {
    **HOP_RECIPE["learning"],
    "seed": 7,
    "steps": 6000,
    "num_envs": 512,       # Robots collecting experience in parallel.
    "batch_size": 2048,   # Replay samples per learning update.
    "wandb_mode": "disabled",
}

# %%
TRAIN_HOPPING = True  # @param {type:"boolean"}
HOP_RUN = None
if TRAIN_HOPPING:
    HOP_RUN = lab.train("stationary_hops", HOP_ENV, LEARNING_CONFIG)
else:
    print("Enable TRAIN_HOPPING to train your own hopper from scratch.")

# %% [markdown]
# ## 4 · Watch and check your hopper
#
# Test your final weights with **three individual Jump requests**. Watch both feet,
# the landing, and the stance between requests. The table reports actual sole
# clearance, flight duration, and falls. Failed attempts remain visible.
# A passing attempt needs one flight lasting at least **0.08 s**, at least
# **2.5 cm** of sole clearance, and a supported landing without falling.
# We then check two seconds of quiet stance, allowing one small balance correction.
# Do not infer success from reward alone.

# %%
from IPython.display import Image, display
from bdxr_classroom.lesson.hop_practice import check_and_watch_hops

HOP_CHECK = None
if HOP_RUN:
    display(Image(filename=str(lab.learning_curve(HOP_RUN)), width=640))
    HOP_CHECK = check_and_watch_hops(lab, HOP_RUN)
else:
    print("Train your hopper above to generate its video and measurements.")

# %% [markdown]
# **Discuss:** did both feet leave the ground, and did the robot settle afterward?
# Clearance and flight duration describe the hop; stable landings make it useful.
# These short flat-ground trials do not establish reliable obstacle jumping.
#
# ## 5 · Click Jump
#
# Wait for **Ready**, then click **Jump** once. Let the robot land before another
# request. **Reset** starts a new attempt. This uses only your trained hopper.
# The arena opens only after the hopping check passes. If it failed, review the
# recorded behavior and training results; the notebook will not launch a failed policy.
# On remote VS Code, forward port **8766** if the panel is blank.

# %%
from bdxr_classroom.lesson.hop_practice import launch_hop_practice

HOP_PORT = 8766  # @param {type:"integer"}
HOP_DEMO = None
if HOP_CHECK and HOP_CHECK["passed"]:
    HOP_DEMO = launch_hop_practice(lab, HOP_RUN, port=HOP_PORT, assessment=HOP_CHECK)
elif HOP_RUN:
    print("The hopping check failed. Review the video and measurements above.")
else:
    print("Train a hopping policy before opening the practice arena.")

# %% [markdown]
# ## 6 · Optional recording and saving
#
# Enable **Record** in the panel, click Jump, then turn Record off. Enable the
# switch below to replay that clip. Recording and downloads are optional.

# %%
from bdxr_classroom.lesson.hop_practice import watch_hop_recording

WATCH_RECORDING = False  # @param {type:"boolean"}
if HOP_DEMO:
    replay = watch_hop_recording(lab, HOP_PORT, enabled=WATCH_RECORDING)

# %%
from bdxr_classroom.lesson.exports import optional_download

SAVE_RESULTS = False  # @param {type:"boolean"}
INCLUDE_REPLAY = False  # @param {type:"boolean"}
if HOP_RUN:
    backup = optional_download(WORK, lab.runs, SAVE_RESULTS, INCLUDE_REPLAY)

# %% [markdown]
# **Keep:** your hopping video, measured clearance and flight duration, and one
# observation about the landing. Report unsuccessful attempts honestly.
#
# [The separate walking class](https://colab.research.google.com/github/TheNewRobot/bdxr-fastsac-classroom/blob/main/notebooks/BDXR_Walking_101.ipynb)
# teaches command-following locomotion; it is not a prerequisite for this lesson.
