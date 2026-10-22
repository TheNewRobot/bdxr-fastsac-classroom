# %% [markdown]
# # Optional after-class extension: learn to hop
#
# Select a GPU runtime and run setup before enabling training.
# This extension is separate from the completed walking lesson.
#
# A hop lifts **both feet** before a controlled landing. This recorded hopper
# reached about 5.9 cm bilateral clearance and 0.16 s of clear flight; its feet
# still tilt about 9°. Foot-marker height alone does not prove obstacle clearance.

# %% [markdown]
# | Earlier vs updated hopping | Same motion at half speed |
# |:---:|:---:|
# | ![Recorded hopping comparison](lesson-asset:inline/hopping.gif) | ![Recorded hopping comparison at half speed](lesson-asset:inline/hopping_slow.gif) |
#
# Watch both feet leave the ground, then inspect the landing.

# %% [markdown]
# ### A curriculum adds one skill at a time
#
# **Rhythm → clearance → foot control → landing.** The goal is not to make every
# row taller. Some stages return to flat ground to isolate a new skill before
# adding obstacles again. Each stage starts from its predecessor's networks.
#
# | Phase | Stages, in order | What changes—and why |
# |---|---|---|
# | **1 · Find a rhythm** | 1. Tiny hops → 2. Moving hops | Start on flat ground; then add speed and turning commands. |
# | **2 · Clear the ground** | 3. Terrain hops | Measure the whole sole in flight and introduce 25 mm terrain. |
# | **3 · Control both feet** | 4. Level hops → 5. Level terrain hops | Reward simultaneous clearance and flatter soles on flat ground; then restore terrain. |
# | **4 · Refine the landing** | 6. Natural hops → 7. Block hops → 8. Soft landing hops | Level feet during flight only, practise steps, then reduce body-lift reward. |
#
# The last phases keep the same 6.5 cm foot target while changing **how** the
# robot takes off, clears terrain, and lands. The eight recorded stages took
# about **23 minutes** on an L40S, excluding setup and evaluation.
#
# | Recorded hopping measurements |
# |:---:|
# | ![Measured instructor hopping](lesson-asset:course_hops/motion.png) |
#
# ### Your optional activity · train and validate hopping
#
# Enable the toggle below to generate your own policies and videos. Leaving it
# off does not train or download an instructor checkpoint.

# %% [setup]

# %%
import json
import os
from pathlib import Path
from bdxr_classroom.lesson.runtime import Lesson
from bdxr_classroom.lesson.hopping import load_curriculum, train_stage

lab = Lesson(WORK, PYTHON, runtime_env)
CURRICULUM = load_curriculum(WORK)
LEARNING_CONFIG = {
    "seed": 7, "num_envs": 512, "batch_size": 2048, "buffer_size": 512,
    "actor_hidden": 256, "critic_hidden": 384, "gamma": 0.97,
    "updates": 4, "compile": True, "wandb_mode": "disabled",
}

# %%
TRAIN_HOPPING = False  # @param {type:"boolean"}
HOP_STAGES = []
if TRAIN_HOPPING:
    previous = None
    for stage in CURRICULUM:
        print("Training stage:", stage["name"])
        previous = train_stage(lab, stage, LEARNING_CONFIG, previous)
        HOP_STAGES.append(previous)

# %% [markdown]
# Choose a stage to inspect after training. Later stages are not automatically
# better. Look for both feet leaving the ground and a controlled landing;
# height of a foot marker alone is not proof of obstacle clearance.
# These recordings are observations, not certification of a hopping controller.

# %%
from IPython.display import display, Video

STAGE_TO_WATCH = 6  # @param {type:"integer"}
if HOP_STAGES:
    if not 1 <= STAGE_TO_WATCH <= len(HOP_STAGES):
        raise ValueError("Choose a stage number between 1 and 8.")
    chosen = HOP_STAGES[STAGE_TO_WATCH - 1]
    display(Video(str(lab.record(chosen)), embed=True, width=640))
else:
    print("Enable hopping training above to produce your own policies.")

# %%
from bdxr_classroom.lesson.exports import export_results

if HOP_STAGES:
    print("Saved:", export_results(WORK, lab.runs, include_replay=True))
