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
# ### Instructor example · the course and a complete run
#
# ![Desert Droid Run terrain](lesson-asset:course_preview.png)
#
# Locate the three obstacles, rough ground, and raised finish before watching the run.
# You will steer your own trained controller through this same world.

# %% [hidden] [instructor]
media_pair(
    [
        (
            "Why hopping matters here",
            WORK / "assets/lesson/obstacle_comparison/comparison.mp4",
            "Same obstacle: two walking commands versus hopping.",
        ),
        (
            "One complete instructor run",
            WORK / "assets/lesson/course25_run/rollout.mp4",
            "Visually steered with pauses. Simulation time is not real-time operator performance.",
        ),
    ],
    video_height=260,
)


# %% [markdown]
# ### Your live interactive game · run the next cell
#
# Running this cell starts a live simulator and embeds its control panel below
# the cell. The videos above are demonstrations; this panel responds to you.
# No browser plugin is required. Keep the notebook runtime connected.
#
# **Click inside the game:** hold **W/S** to move, **A/D** to turn, and press
# **Space** to switch walking/hopping. Release movement keys to stop. The panel
# also has on-screen controls. Hopping requires a trained hopping checkpoint.
#
# On a remote VS Code session, forward port **8765** using the Ports panel if
# the embedded panel is blank, then open its forwarded address in your browser.
#
# Leave paths blank to use the policies selected from your experiments. You can
# also upload compatible checkpoints using Colab’s Files panel and enter their
# paths. Walking-only play
# is available if hopping was skipped, but this does not establish that your
# walker can cross the course.
#
# Switch to hopping before an obstacle and allow room for the handover. Return
# to walking after the feet clear the obstacle and land. Watch the active mode
# and transition state in the interface.
#
# **Fall → reset to the start; time and falls accumulate.** Clear the route's
# checkpoints and reach the finish upright to win. **New game** resets the score.

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
