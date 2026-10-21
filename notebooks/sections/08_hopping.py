# %% [markdown]
# ## 8 · Optional extension: learn to hop
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

# %%
from bdxr_classroom.lesson.hopping import load_curriculum, train_stage, validate_pair

CURRICULUM = load_curriculum(WORK)

# %%
from IPython.display import display, Video

TRAIN_HOPPING = False  # @param {type:"boolean"}
HOP_STAGES = []
SELECTED_HOP_CHECKPOINT = None
if TRAIN_HOPPING:
    previous = None
    for stage in CURRICULUM:
        print("Training stage:", stage["name"])
        previous = train_stage(lab, stage, LEARNING_CONFIG, previous)
        HOP_STAGES.append(previous)
        display(Video(str(lab.record(previous)), embed=True, width=640))

# %% [markdown]
# ### Select evidence, not just the latest checkpoint
#
# Saved earlier checkpoints can outperform the final checkpoint. Screen the exact
# weights on terrain, then test repeated walking/hopping handovers on flat ground
# and a step. Every recorded failure stays in the history; a later success does
# not silently erase it. Passing these checks does not establish full-course skill.

# %%
from IPython.display import display, Image

if HOP_STAGES:
    SELECTED_HOP_CHECKPOINT = validate_pair(
        lab, HOP_STAGES[-1], SELECTED_WALK_CHECKPOINT
    )
    display(Image(filename=str(lab.learning_curve(HOP_STAGES[-1])), width=900))
    print("Selected hopping checkpoint:", SELECTED_HOP_CHECKPOINT)
else:
    print(
        "Hopping training skipped. An existing compatible checkpoint can be loaded in the game."
    )

# %% [markdown]
# **Discuss:** why might two good policies still fail when switching between them?
# Consider command speed, phase, and abrupt changes in joint targets.
