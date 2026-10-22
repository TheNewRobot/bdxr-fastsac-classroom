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
# Both trained from scratch for 12,000 steps. Swing-foot tilt fell from **8.7° to
# 5.6°**, but the variant missed the speed target. Your measurements may differ.

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
