# %% [markdown]
# ## 7 · Change one thing and train again
#
# Commanding a trained robot to walk faster is different from **changing a reward
# and training a different gait**. Here we hold the command range fixed and alter
# one interpretable behavior setting. Train the variant and an unchanged control
# from scratch with the same seed and step budget.
#
# | Experiment | Changed reward setting | What to inspect |
# |---|---|---|
# | Level feet | Sole-level reward weight 0.75 | Foot tilt |
# | Quick steps | Airtime window 0.06–0.16 s, weight 4 | Step cadence |
# | Crouched walking | Torso-height target 0.23 m, fixed weight 4 | Body height and foot clearance |
#
# Quick steps changes one reward's window and weight together; it is not a
# one-scalar ablation. All three recorded variants passed the full command suite,
# with one successful training seed per style. That is an example, not proof that
# every seed will succeed.

# %% [markdown]
# ### A recorded example to compare with your prediction
#
# This instructor comparison is shown at equal commands. Local variant training
# took 14m54s (level feet), 15m1s (quick steps), or 11m13s (crouched).
# The unchanged seed-8 control missed command checks; it is labeled as a matched
# comparison, not another validated baseline. The selected crouched checkpoint
# is the 9,000-step result; training longer did not improve its acceptance checks.

# %% [markdown]
# #### Instructor example · Flatter feet
#
# | Matched-command example | Measured gait differences |
# |:---:|:---:|
# | ![Recorded flatter-foot comparison at matched commands](lesson-asset:inline/level_feet.gif) | ![Measured gait differences](lesson-asset:reference/level_feet_focus.png) |

# %% [markdown]
# #### Instructor example · Quicker steps
#
# | Matched-command example | Measured gait differences |
# |:---:|:---:|
# | ![Recorded quick-step comparison at matched commands](lesson-asset:inline/quick_steps.gif) | ![Measured gait differences](lesson-asset:reference/quick_steps_focus.png) |

# %% [markdown]
# #### Instructor example · Crouched walking
#
# | Matched-command example | Measured gait differences |
# |:---:|:---:|
# | ![Recorded crouched comparison at matched commands](lesson-asset:inline/crouched.gif) | ![Measured gait differences](lesson-asset:reference/crouched_focus.png) |

# %% [markdown]
# ### Your turn
#
# Predict a measurable change. Train a matched control and variant, then compare
# your videos, measurements, and command checks.

# %%
EXPERIMENT = "level_feet"  # @param ["level_feet", "quick_steps", "crouched"]
PREDICTION = "The feet will stay flatter during stepping."  # @param {type:"string"}
CHANGES = {
    "level_feet": {"level_feet_weight": 0.75},
    "quick_steps": {"air_time_min": 0.06, "air_time_max": 0.16, "air_time_weight": 4.0},
    "crouched": {"walking_height": 0.23},
}
VARIANT_ENV = {**ENV_CONFIG, **CHANGES[EXPERIMENT]}
PAIR_LEARNING = {
    **LEARNING_CONFIG,
    "seed": 8 if EXPERIMENT == "crouched" else 7,
    "steps": 9000 if EXPERIMENT == "crouched" else 10000,
}
print("Prediction:", PREDICTION)
print("Only changed behavior setting:", CHANGES[EXPERIMENT])

# %%
# Neither call has a parent checkpoint: both start from scratch.
CONTROL = lab.train("scratch_control", ENV_CONFIG, PAIR_LEARNING)
VARIATION = lab.train(EXPERIMENT, VARIANT_ENV, PAIR_LEARNING)

# %%
CONTROL_SUMMARY, CONTROL_CHECK = evaluate_walking(lab, CONTROL)
VARIATION_SUMMARY, VARIATION_CHECK = evaluate_walking(lab, VARIATION)
command_table(CONTROL_SUMMARY)
command_table(VARIATION_SUMMARY)
print("Control passes:", CONTROL_CHECK["passed"])
print("Variation passes:", VARIATION_CHECK["passed"])

# %%
# Same commanded speed makes the visual comparison meaningful.
control_video = lab.record(
    CONTROL, speed=0.30, expected_sha256=CONTROL_CHECK["checkpoint_sha256"]
)
variation_video = lab.record(
    VARIATION, speed=0.30, expected_sha256=VARIATION_CHECK["checkpoint_sha256"]
)
media_pair(
    [
        (
            "Your unchanged control",
            control_video,
            "Fresh training at the matched seed and budget.",
        ),
        ("Your variation", variation_video, f"Changed setting: {CHANGES[EXPERIMENT]}"),
    ]
)

# %%
from bdxr_classroom.lesson.gait import compare_gaits

compare_gaits(lab, CONTROL, VARIATION)
media_pair(
    [
        (
            "Control learning curve",
            lab.learning_curve(CONTROL),
            "Read the trend; do not compare reward magnitude alone.",
        ),
        (
            "Variation learning curve",
            lab.learning_curve(VARIATION),
            "The reward definition changed between experiments.",
        ),
    ]
)
SELECTED_WALK_CHECKPOINT = selected_walking(
    [BASELINE_CHECK, CONTROL_CHECK, VARIATION_CHECK]
)

# %% [markdown]
# **Explain your result:** did the gait change as predicted, and did command
# following remain good? A different-looking policy that falls is not a successful
# variant. Check actual step counts: a time limit can stop one run earlier and
# spoil the equal-budget comparison. Preserve failures in your report.
