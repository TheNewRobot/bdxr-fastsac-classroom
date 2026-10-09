# Implementation and testing scope

The repository includes pinned robot/simulator dependencies, FastSAC training,
command evaluation, resumable checkpoints, W&B logging, and the playable course.
The actor uses proprioception and commands; the critic may use privileged
training observations. GPU tests cover that separation.

The lesson contains scratch-trained walking variations and an extended hopping
curriculum. It checks command tracking, drift, falls, hopping, and repeated
walk/hop handovers before selecting controllers. Course scoring requires three
obstacles, the rough patch, and the raised finish. Recorded fixed-command
failures remain distinct from visually steered successes.

An earlier full notebook version completed fresh local training, evaluation,
and export. Local browser checks exercised policy loading, keyboard/buttons,
focus-loss stopping, mode changes, and backup integrity. Those checks do not
establish actual Google Colab compatibility or performance.

Current validation records identify the notebook hash, hardware, settings,
and whether a run was a short integration test or full training. They live in
ignored experiment/artifact folders. Curated results and limitations accompany
the media in `assets/lesson/`; raw development history stays out of Git.

**Actual Google Colab execution remains untested.** Training times are measured
on the local L40S and exclude setup/evaluation unless stated otherwise.
