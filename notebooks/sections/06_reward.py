# %% [markdown]
# ## 6 · Change one reward and compare
#
# Predict what happens when we reward flatter feet. Change **one scalar reward
# weight**, then train a new policy from scratch. Your first run is the unchanged
# control: keep the seed, commands, and learning budget identical.
#
# | Instructor example at equal commands |
# |:---:|
# | ![Fresh baseline and flatter-foot comparison, with check outcomes](lesson-asset:classroom_walking/reward_comparison.gif) |
#
# The preview loops automatically; it has no Play button.
# If it stays still, [open the comparison video](lesson-asset:classroom_walking/reward_comparison.mp4)
# and press ▶ in the video player.
#
# Only `level_feet_weight` changes: **0 → 0.75**. Both trained from scratch for
# 12,000 steps. Measured swing-foot tilt fell from **9.7° to 5.2°**, while foot
# landings increased from **4.11 to 4.52 per second**. Both passed the brief
# walking check.
#
# **Watch the feet:** the change is subtle—flatter feet and slightly quicker
# steps. This example shows how one reward influences movement, rather than a
# dramatically different gait. Your measurements may differ.

# %%
PREDICTION = "The feet will stay flatter during stepping."  # @param {type:"string"}
LEVEL_FEET_WEIGHT = 0.75  # @param {type:"number"}
VARIANT_ENV = {**ENV_CONFIG, "level_feet_weight": LEVEL_FEET_WEIGHT}
BASELINE_STEPS = read_json(BASELINE / "metrics/timing.json")["step"]
VARIANT_LEARNING = {**LEARNING_CONFIG, "steps": BASELINE_STEPS}
print("My prediction:", PREDICTION)

# %%
# No parent checkpoint: the variant starts from random networks too.
VARIATION = lab.train("level_feet", VARIANT_ENV, VARIANT_LEARNING)
variation_video = lab.record(VARIATION, speed=0.30)

# %%
media_pair([
    ("Your baseline", baseline_video, ""),
    ("Your reward variation", variation_video, ""),
])

# %%
VARIATION_CHECK = assess_walking(lab, VARIATION)

# %%
from bdxr_classroom.lesson.gait import compare_gaits

compare_gaits(lab, BASELINE, VARIATION,
              assessments=[BASELINE_CHECK, VARIATION_CHECK])

# %% [markdown]
# **Explain:** did foot tilt change as predicted, while walking remained useful?
# The variant targets the baseline's actual completed step count, including when
# the first run hit its time limit. Check that both runs finished that budget.
# Changing a reward changes
# the return scale, so a larger return alone does not establish improvement.
