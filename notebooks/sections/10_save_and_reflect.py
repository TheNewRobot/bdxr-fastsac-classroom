# %% [markdown]
# ## 10 · Save and explain your results
#
# Pause the game before exporting. The backup includes your training stages,
# configurations, checkpoints, curves, and recorded gameplay. Including replay
# makes the file larger but preserves collected experience for continuation.

# %%
recordings = pause_and_recordings(lab, GAME_PORT)
for video in recordings:
    display(Video(str(video), embed=True, width=720))
if not recordings:
    print("No gameplay video yet. Drive the droid, then rerun this cell.")

# %%
from bdxr_classroom.lesson.exports import export_results

INCLUDE_REPLAY = True  # @param {type:"boolean"}
backup = export_results(WORK, lab.runs, include_replay=INCLUDE_REPLAY)
print("Saved:", backup)
try:
    from google.colab import files

    files.download(str(backup))
except ImportError:
    print("Retrieve this ZIP from the local lesson folder.")

# %% [markdown]
# To continue later, unpack the backup into the lesson folder. The training CLI's
# `--resume <run>/checkpoints/resume.pt` restores saved networks, optimizers,
# normalization, and replay; simulated episodes restart. Branching from an earlier
# `step_*.pt` with `--finetune` starts new replay and is a different operation.
#
# **Your short lab report:**
#
# 1. Explain how one joint angle changes foot position using the kinematic study.
# 2. Identify the actor's observations and explain why it cannot use world heading.
# 3. Show your baseline video and evidence for command tracking and straightness.
# 4. State your changed reward, prediction, seed, actual training budget, and result.
# 5. Explain one failure or limitation—including whether both runs passed the checks.
#
# Kinematics describes the motion geometry. Dynamics determines whether forces
# and contacts can realize it. Reinforcement learning searches for actions that
# earn reward; videos and independent measurements tell us what it actually learned.
