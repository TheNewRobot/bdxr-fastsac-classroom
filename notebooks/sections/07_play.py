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
# Click inside the panel: **W/S** moves and **A/D** turns. Release keys to stop;
# use the sliders for finer speed control. Falling resets the robot but keeps
# accumulated time and falls. Pause stops simulation time.
# On remote VS Code, forward port 8765 if the panel is blank.

# %%
from bdxr_classroom.lesson.play import launch_game, display_game, pause_and_recordings

POLICY = "baseline"  # @param ["baseline", "variation"]
GAME_PORT = 8765
chosen_run = BASELINE if POLICY == "baseline" else VARIATION
walking = str(chosen_run / "checkpoints/final.pt")
status = launch_game(lab, walking, port=GAME_PORT, course="configs/walking_course.json")
display_game(GAME_PORT)

# %% [markdown]
# ### Keep your result
#
# Play first, then pause and run these cells to show your latest recording and
# export configurations, measurements, checkpoints, and gameplay. Replay state
# preserves collected experience for continuation; it increases the download size.

# %%
from IPython.display import display, Video

recordings = pause_and_recordings(lab, GAME_PORT)
if recordings:
    display(Video(str(recordings[-1]), embed=True, width=640))
else:
    print("Drive the droid first, then rerun this cell to see your recording.")

# %%
from bdxr_classroom.lesson.exports import export_results

INCLUDE_REPLAY = True  # @param {type:"boolean"}
backup = export_results(WORK, lab.runs, include_replay=INCLUDE_REPLAY)
print("Saved:", backup)
try:
    from google.colab import files
    files.download(str(backup))
except ImportError:
    print("Retrieve the ZIP from your local lesson folder.")

# %% [markdown]
# **Submit:** your walking video, short check, reward prediction/comparison,
# and a gameplay clip. State actual steps/timings and one limitation.
#
# After class: [the separate hopping extension](https://colab.research.google.com/github/TheNewRobot/bdxr-fastsac-classroom/blob/main/notebooks/BDXR_Hopping_Extra.ipynb)
# explores another behavior. It is not required to finish this lesson or drive the course.
