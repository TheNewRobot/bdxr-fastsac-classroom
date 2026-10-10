# %% [markdown]
# ## 6 · Train your walking controller
#
# This call starts a fresh actor and critics using the two configurations above.
# Checkpoints, exact settings, timings, and real learning metrics are saved in
# a timestamped experiment folder. The helper only runs the training program
# and manages its output paths; the learning settings remain visible here.

# %% [markdown]
# ### Instructor example · a real fresh-training run
#
# | Learning: episode return | Independent check: lateral drift |
# |:---:|:---:|
# | ![Fresh seed-7 walking learning curve](lesson-asset:reference/scratch_learning.png) | ![Measured lateral drift during forward walking](lesson-asset:reference/scratch_forward.png) |
#
# Seed 7, trained from scratch. Rising return suggests learning; the drift plot
# checks straightness. Each line is one trial. World position is evaluation-only.
#
# ### Your activity · train, record, and evaluate
#
# Train, record your before/after videos, and check command following. Compare
# your results with the examples above.

# %%
BASELINE = lab.train("command_walking_baseline", ENV_CONFIG, LEARNING_CONFIG)
INITIAL_BASELINE = BASELINE
print("Saved experiment:", BASELINE)

# %%
# Both recordings use checkpoints from this training run.
before = lab.record(BASELINE, checkpoint="initial")
after = lab.record(BASELINE, checkpoint="final")
media_pair(
    [
        ("Before learning", before, "Initial randomly initialized actor."),
        (
            "After learning",
            after,
            "Your saved final checkpoint—not an instructor recording.",
        ),
    ]
)

# %%
display(Image(filename=str(lab.learning_curve(BASELINE)), width=720))
print("Actual timing:", read_json(BASELINE / "metrics/timing.json"))

# %% [markdown]
# ### Does it do what we asked?
#
# Evaluate slow/normal/fast forward walking, backward walking, stopping, turns,
# and a command sequence. Use five perturbed starting states for each command.
# These are **evaluation trials of one trained policy**, not five training seeds.
#
# The fixed acceptance limits check speed error, yaw-rate error, straight-line
# drift, complete 15-second trials, and zero falls. Curved paths are expected
# when the command requests a turn. World position is measured for evaluation;
# it is not an actor input.

# %%
from bdxr_classroom.lesson.assessment import evaluate_walking, selected_walking
from bdxr_classroom.lesson.presentation import command_table

BASELINE_SUMMARY, BASELINE_CHECK = evaluate_walking(lab, BASELINE)
command_table(BASELINE_SUMMARY)
print("Passed:", BASELINE_CHECK["passed"])
print("Failures:", BASELINE_CHECK["failures"])

# %% [markdown]
# If the first checkpoint misses the checks, allow one additional training stage
# with the **same behavior settings**. This fine-tunes saved networks using fresh
# replay; it is not an exact replay-buffer resume. Preserve and inspect both runs.

# %%
if not BASELINE_CHECK["passed"]:
    recovery = {**LEARNING_CONFIG, "steps": 3000, "minutes": 10}
    BASELINE = lab.train(
        "command_walking_recovery",
        ENV_CONFIG,
        recovery,
        parent=BASELINE / "checkpoints/final.pt",
    )
    BASELINE_SUMMARY, BASELINE_CHECK = evaluate_walking(lab, BASELINE)
    command_table(BASELINE_SUMMARY)
if not BASELINE_CHECK["passed"]:
    raise RuntimeError(
        "Inspect the failed walking checks before starting style experiments."
    )
SELECTED_WALK_CHECKPOINT = selected_walking([BASELINE_CHECK])

# %%
# Keep the robot video separate from the smaller diagnostic chart.
display(Video(str(BASELINE / "videos/final_commands/forward.mp4"), embed=True, width=640))
display(Image(filename=str(BASELINE / "evaluation/final_commands/forward.png"), width=720))

# %% [markdown]
# **Interpret:** can you point to evidence for straight walking, beyond the video?
# If reward rises while the controller still falls, is training finished?
