# %% [markdown]
# ## 8 · Optional extension: learn to hop
#
# Walking alternates support. A hop has an interval when **both feet leave the
# ground**, followed by a controlled landing. A foot marker moving upward does
# not prove the whole sole cleared an obstacle.
#
# Compare bilateral sole clearance, flight duration, foot tilt, and landing.
# The recorded course hopper averaged about 5.9 cm bilateral peak clearance and
# 0.16 s clear flight. Its feet still tilt about 9 degrees; they are not perfectly
# parallel to the floor. Sampled touchdown-force peaks were about 30% lower than
# the earlier hopper, but 20 ms samples are not exact substep impact forces.

# %% [hidden] [instructor]
media_pair(
    [
        (
            "Earlier and updated hopping",
            WORK / "assets/lesson/course_hops/comparison.mp4",
            "Recorded instructor policies at the same forward command.",
        ),
        (
            "The same motion at half speed",
            WORK / "assets/lesson/course_hops/half_speed.mp4",
            "Inspect bilateral takeoff, flight, and landing.",
        ),
    ]
)

recorded_details(
    WORK / "assets/lesson/course_hops",
    "Measured instructor hopping. The learning curve covers only the final 1,000-step stage, not all eight stages.",
)

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
# The video above is an instructor reference, available without training.
# The table below is the recorded curriculum; every stage starts from its listed predecessor.

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
