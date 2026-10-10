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
# ### Why use a curriculum?
#
# Start with an easier task, then add movement and physical terrain. The actor
# gets a rhythm command for hopping, not a terrain map. Each stage starts from
# the preceding stage's learned networks with fresh replay.
#
# The complete eight-stage recorded lineage took about **23 minutes of training**
# on an L40S, excluding setup and evaluation. This is an extended activity,
# separate from the short walking exercise. Skipping it leaves walking available
# for exploratory play; it does not provide a validated hopping controller.

# %% [markdown]
# ![Measured instructor hopping](lesson-asset:course_hops/motion.png)
#
# Recorded measurements above; training stages below. Each stage continues its predecessor.

# %% [markdown]
# | Stage | Steps | Requested foot height (m) | Terrain height (m) |
# |---|---:|---:|---:|
# | tiny_hops | 4,000 | 0.04 | 0 |
# | moving_hops | 6,000 | 0.07 | 0 |
# | terrain_hops | 2,000 | 0.045 | 0.025 |
# | level_hops | 5,000 | 0.065 | 0 |
# | level_terrain_hops | 3,000 | 0.065 | 0.025 |
# | natural_hops | 1,000 | 0.065 | 0.025 |
# | block_hops | 2,000 | 0.065 | 0.025 |
# | soft_landing_hops | 1,000 | 0.065 | 0.025 |
#
# ### Your optional activity · train and validate hopping
#
# Enable the toggle below to generate your own policies and videos. Leaving it
# off does not train or download an instructor checkpoint.

# %%
from bdxr_classroom.lesson.hopping import load_curriculum, train_stage, validate_pair

CURRICULUM = load_curriculum(WORK)

# %%
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
