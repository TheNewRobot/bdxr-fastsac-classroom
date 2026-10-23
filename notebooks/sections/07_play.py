# %% [markdown]
# ## 7 · Drive your trained robot and save
#
# Finish by driving **your own walking controller** through Desert Droid Run.
# Steer between the markers, slow down for narrow passages and uneven ground,
# then reach the raised finish. Required obstacles use walking; no hopper is needed.
# You command speed and turn rate. The learned actor supplies the leg actions.
#
# | The walking course |
# |:---:|
# | ![Desert walking course](lesson-asset:course_preview.png) |
#
# Click **Resume**, then hold a direction button or **W/S** to move and **A/D**
# to turn. Release to request stopping. **Slow / Normal / Fast** select speed;
# they do not start movement. Actual motion and requested commands are shown
# separately. Falling keeps accumulated time and falls; Pause freezes simulation.
# On remote VS Code, forward port 8765 if the panel is blank.
#
# **Choose a checked policy:** `auto` uses a passing run, preferring baseline if
# both pass. The summary reuses your existing checks—no extra training or
# evaluation. A failed selection will not launch the game. Passing the short
# forward check does not guarantee successful turning or obstacle traversal.

# %%
from bdxr_classroom.lesson.policy_choice import choose_walking_policy

POLICY = "auto"  # @param ["auto", "baseline", "variation"]
walking = choose_walking_policy(
    {"baseline": BASELINE, "variation": VARIATION},
    {"baseline": BASELINE_CHECK, "variation": VARIATION_CHECK},
    preference=POLICY,
)
SELECTED_WALK_CHECKPOINT = walking

# %%
from bdxr_classroom.lesson.play import launch_game, display_game

GAME_PORT = 8765
if walking:
    status = launch_game(lab, walking, port=GAME_PORT, course="configs/walking_course.json")
    display_game(GAME_PORT)
else:
    print("Choose a policy with a passing check in the preceding cell.")

# %% [markdown]
# ### Optional · watch and keep your result
#
# To keep a clip, open **Recording · optional** in the game and start recording.
# Enable **WATCH_RECORDING** below to pause and watch your latest clip.
# Saving is optional: the ZIP contains configurations, measurements,
# checkpoints, and gameplay. Include replay only for a larger resumable backup.

# %%
from bdxr_classroom.lesson.play import optional_recording

WATCH_RECORDING = False  # @param {type:"boolean"}
replay = optional_recording(lab, GAME_PORT, WATCH_RECORDING)

# %%
from bdxr_classroom.lesson.exports import optional_download

SAVE_RESULTS = False  # @param {type:"boolean"}
INCLUDE_REPLAY = False  # @param {type:"boolean"}
backup = optional_download(WORK, lab.runs, SAVE_RESULTS, INCLUDE_REPLAY)

# %% [markdown]
# **Submit:** your walking video, short check, and reward prediction/comparison.
# State actual steps/timings and one limitation. A gameplay clip is optional.
#
# After class: [the separate hopping extension](https://colab.research.google.com/github/TheNewRobot/bdxr-fastsac-classroom/blob/main/notebooks/BDXR_Hopping_Extra.ipynb)
# explores another behavior. It is not required to finish this lesson or drive the course.
