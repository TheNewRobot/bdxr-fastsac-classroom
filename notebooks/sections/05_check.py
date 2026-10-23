# %% [markdown]
# ## 5 · Briefly check the walking
#
# Run **three 10-second forward trials** from slightly different starting poses.
# We measure falls, speed error, turning error, and sideways drift. The numerical
# tolerances stay unchanged; this short check does not test every command or terrain.
# It targets under a minute of wall time and prints the actual duration.
#
# World position measures drift for evaluation only—it never enters the actor.
# A steady-looking video and a drift measurement answer different questions.

# %%
from bdxr_classroom.lesson.brief import assess_walking

BASELINE_CHECK = assess_walking(lab, BASELINE)

# %% [markdown]
# **Interpret:** “stayed upright” and “followed the command” are distinct results.
# Record shortcomings honestly. This cell does not select another checkpoint,
# retrain the robot, or launch more checks automatically.
