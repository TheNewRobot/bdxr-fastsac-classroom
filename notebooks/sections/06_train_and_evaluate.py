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
            "After the first training attempt",
            after,
            "This attempt still needs the command checks below.",
        ),
    ]
)

# %%
from IPython.display import display, Image

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
# These checks can take several minutes; progress identifies each command.
# Measurements are saved without rendering unused videos. We record the selected
# policy below.

# %%
from bdxr_classroom.lesson.assessment import (
    check_recent_walking,
    drift_only_failure,
    evaluate_walking,
    selected_walking,
)
from bdxr_classroom.lesson.presentation import command_table

BASELINE_SUMMARY, BASELINE_CHECK = evaluate_walking(lab, BASELINE)
INITIAL_BASELINE_CHECK = BASELINE_CHECK
command_table(BASELINE_SUMMARY)
print("Passed:", BASELINE_CHECK["passed"])
print("Failures:", BASELINE_CHECK["failures"])

# %% [markdown]
# A later update can make a policy worse. If the final checkpoint fails, check
# up to two recent saved snapshots using the **same full command suite**. This
# adds evaluation time, not training. Each result stays in its own folder.
#
# If none passes, allow one additional training stage
# with the **same behavior settings**. This fine-tunes saved networks using fresh
# replay; it is not an exact replay-buffer resume. Preserve and inspect both runs.
# A recovered policy has its own folder and video; it does not replace the first attempt.

# %%
BASELINE_SUMMARY, BASELINE_CHECK = check_recent_walking(
    lab, BASELINE, BASELINE_SUMMARY, BASELINE_CHECK
)
if not BASELINE_CHECK["passed"]:
    recovery = {**LEARNING_CONFIG, "steps": 3000, "minutes": 10}
    BASELINE = lab.train(
        "command_walking_recovery",
        ENV_CONFIG,
        recovery,
        parent=BASELINE / "checkpoints/final.pt",
    )
    BASELINE_SUMMARY, BASELINE_CHECK = evaluate_walking(lab, BASELINE)
    BASELINE_SUMMARY, BASELINE_CHECK = check_recent_walking(
        lab, BASELINE, BASELINE_SUMMARY, BASELINE_CHECK
    )
    command_table(BASELINE_SUMMARY)

# %% [markdown]
# If no snapshot passes and the original final policy failed **only drift**,
# try one last **3,000-step recovery** with yaw-bias
# reward weight **10** (the baseline uses 6). This discourages sustained turning
# without adding a heading controller. It starts again from your original
# attempt; all failed experiments and their timings remain saved.
# Falls or poor command following need more basic learning, not this refinement.

# %%
# @title Additional recovery only if needed
if not BASELINE_CHECK["passed"] and drift_only_failure(INITIAL_BASELINE_CHECK):
    recovery_env = {**ENV_CONFIG, "yaw_bias_weight": 10.0}
    recovery_learning = {**LEARNING_CONFIG, "steps": 3000, "minutes": 10}
    BASELINE = lab.train(
        "command_walking_yaw_recovery",
        recovery_env,
        recovery_learning,
        parent=INITIAL_BASELINE / "checkpoints/final.pt",
    )
    BASELINE_SUMMARY, BASELINE_CHECK = evaluate_walking(lab, BASELINE)
    BASELINE_SUMMARY, BASELINE_CHECK = check_recent_walking(
        lab, BASELINE, BASELINE_SUMMARY, BASELINE_CHECK
    )
    command_table(BASELINE_SUMMARY)
# %%
SELECTED_WALK_CHECKPOINT = selected_walking([BASELINE_CHECK])
if not BASELINE_CHECK["passed"]:
    raise RuntimeError(
        "No checked snapshot passed. Keep the reports and inspect the failed "
        "commands before trying another seed or a longer training budget."
    )
print("Selected walking checkpoint:", SELECTED_WALK_CHECKPOINT)
print("Passed command checks:", BASELINE_CHECK["passed"])

# %%
from IPython.display import display, Image, Video

# Keep the robot video separate from the smaller diagnostic chart.
selected_evaluation = Path(BASELINE_CHECK["evaluation_directory"])
selected_video = lab.record(
    BASELINE,
    checkpoint=Path(BASELINE_CHECK["checkpoint"]).stem,
    speed=0.30,
    expected_sha256=BASELINE_CHECK["checkpoint_sha256"],
)
print("Selected-policy evaluation:", selected_evaluation)
display(Video(str(selected_video), embed=True, width=640))
display(Image(filename=str(selected_evaluation / "forward.png"), width=720))

# %% [markdown]
# The video is one new rollout of the checked weights; the plot covers the
# independent command-test starts.
#
# **Interpret:** can you point to evidence for straight walking, beyond the video?
# If reward rises while the controller still falls, is training finished?
