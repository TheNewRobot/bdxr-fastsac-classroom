# Teaching the walking lesson

The core has seven sections: environment, kinematics, walking problem, SAC
training, a brief walking check, one reward experiment, and the walking game.
Students train two policies from scratch. The first run is the unchanged control;
the second changes only the sole-level reward weight. Hopping is a separate
[after-class notebook](../notebooks/BDXR_Hopping_Extra.ipynb).

Keep the environment and learning configurations visible. Explain sensor
measurements versus commands, the 47 actor inputs, 62 critic inputs, and ten leg
actions. Privileged simulator measurements help the critic and rewards; they do
not enter the deployed actor or a hidden path controller.

Show the student's walking video immediately after training. The short check
uses three 10-second forward trials and prints its actual wall time. It retains
the previous numerical tolerances but does not certify turning or terrain skill.
No recovery training or checkpoint search runs automatically. Discuss balance,
tracking, and drift separately instead of treating an upright robot as perfect.

The reward comparison reuses the saved assessment trajectories for gait
measurements. Its target step count matches the baseline's completed steps;
check both actual budgets before attributing a change to the reward. A higher
return after changing reward weights is not itself evidence of a better gait.

End with students steering their own policy through the walking course. W/S and
A/D control movement; sliders allow finer commands. Falling resets the robot,
while elapsed simulation time and fall count accumulate. Pause before exporting.
No hopping policy is required for the core course.

Allow for two training runs, setup, discussion, rendering, and gameplay. The
15-minute training limit is per run, not the duration of the whole lesson. A
slower runtime may stop before the requested step count. See the current
[validation scope](progress.md) before advertising a classroom duration.
