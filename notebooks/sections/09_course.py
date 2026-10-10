# %% [markdown]
# ## 9 · Play: Desert Droid Run
#
# Drive through **three obstacles**, cross the rough patch, and reach the raised
# finish beacon. You provide speed, turn-rate, and mode commands; the trained
# controller supplies leg actions. There is no navigation planner.
#
# The middle step is 25 mm high. The selected hopper crossed 3/3 tested starts;
# the walking controller crossed 0/6 at normal/fast commands. This supports hopping
# for these controllers and trials, not a claim that every walker must fail.
# Recorded visually steered runs finished without falls in about 50–53 simulated
# seconds. Fixed command replays can miss the finish; players need to steer.

# %% [markdown]
# ### Instructor example · the course
#
# ![Desert Droid Run terrain](lesson-asset:course_preview.png)
#
# Locate the three obstacles, rough ground, and raised finish before watching the run.
# You will steer your own trained controller through this same world.

# %% [markdown]
# | Why hopping matters | Course preview |
# |:---:|:---:|
# | ![Walking versus hopping over the obstacle](lesson-asset:inline/obstacle.gif) | ![First eight seconds of the recorded instructor run](lesson-asset:inline/course.gif) |
#
# Left: the same step at two walking speeds and with hopping. Right: a visually
# steered run’s first eight seconds. These are recorded examples.

# %% [markdown]
# ### Your live interactive game
#
# Run the next cell to open the live control panel here. Keep the runtime connected.
# Leave paths blank to use your selected policies, or enter compatible checkpoints.
#
# **Click inside:** W/S moves, A/D turns, Space switches walk/hop. Release keys to
# stop, or use the on-screen buttons. Hopping needs a trained hopping checkpoint.
# Switch before the obstacle and allow the robot to settle after landing.
#
# **Fall → reset; time accumulates.** Reach the raised finish upright to win.
# On remote VS Code, forward port 8765 if the panel is blank. No plugin is needed.
#
# %%
# @title Start the interactive game with your trained controller
from bdxr_classroom.lesson.play import launch_game, display_game, pause_and_recordings

WALK_CHECKPOINT = ""  # @param {type:"string"}
HOP_CHECKPOINT = ""  # @param {type:"string"}
GAME_PORT = 8765
walking = WALK_CHECKPOINT or SELECTED_WALK_CHECKPOINT
hopping = HOP_CHECKPOINT or SELECTED_HOP_CHECKPOINT
status = launch_game(lab, walking, hopping, port=GAME_PORT)
display_game(GAME_PORT)
print("Game recordings and score:", status["experiment"])
