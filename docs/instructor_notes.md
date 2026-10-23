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

End with students steering their own policy through the walking course. Hold
direction buttons or W/S/A/D to move; release to request stopping. Speed presets
choose the pace, and the panel shows measured speed and turn rate separately.
Falling resets the robot while simulation time and falls accumulate. Recording
and exporting are optional. No hopping policy is required for the core course.

Allow for two training runs, setup, discussion, rendering, and gameplay. The
22-minute training limit is per run, not the duration of the whole lesson.
Use a Colab GPU runtime and retain the 12,000-step target. Setup accepts T4;
a user-reported run took 1,024 seconds with the earlier recipe. The revised
quiet-standing recipe produced a failed baseline and a successful variation in
the latest uploaded T4 run. See the current
[validation scope](progress.md) before advertising a classroom duration.

## The independent hopping class

Students train one stationary hopping policy from scratch for 6,000 steps, then
record three individual hops and test a Jump button on flat ground. They do not
upload a walking checkpoint, switch policies, or navigate an obstacle course.
The actor has 49 inputs (including two phase components), the critic 64, and the
policy controls ten leg actions. Jump advances one phase cycle; the stance cue
is held between requests while physics and the actor continue running.

Use the measured flight, whole-sole clearance, and landing results to discuss
success. The short check retains unsuccessful attempts; training curves alone
do not establish hopping. Recording, replay and export are optional. This demo
tests flat-ground hops, not obstacle clearance or arbitrary terrain.

The recipe in `configs/stationary_hopping.json` explicitly rewards bilateral
flight. One local seed passed three hops with quiet landings; the previous
eight-stage recipe failed the same check. Do not present either reward curves
or the reference GIF as proof that a student run succeeded.
