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

The refactored lesson has ten readable chapter sources, with explicit environment
and learning settings, an actual-model joint tree, an interactive knee sweep,
and a visible network/experience-collection example. The previous release notebook completed
all 43 code cells locally with short training runs and reused pinned dependencies.
Its 103 bundled files match the repository byte for byte.
The integration copy records failed short-run policy verdicts and bypasses its
classroom stop gate so later game/export cells can be exercised; this is not
evidence that those short runs learned to walk.

The previous release’s 64-test suite passed, including simulator checks with the selected
walking checkpoint. The 29 protected checkpoint/configuration files retain their
original hashes. All nine retained distribution videos decode successfully;
compression preserved their resolution, frame count, and playback speed.
Local browser inspection verified the reference photo, annotated model, joint
tree, knee slider, and SAC diagram. The game controls also passed a Chromium
transport-fixture check, including checkpoint suggestions and focus-loss stopping.

The internal presentation layer passed 50 command trials for each of the five
selected walkers, plus the hopper's 15 terrain trials and repeated flat/on-step
handovers. A visually steered course run finished in 53.14 simulated seconds,
with four checkpoints and no falls. The earlier larger idle gesture destabilized
the quick-step policy; its failure remains recorded. The checked, smaller gesture
is enabled by the internal inference configuration and is not a student topic.

An additional local integration execution exercised all eight optional hopping
stages at short budgets. Its backup contained all eleven training runs' resumable
checkpoints and passed integrity, duplicate-entry, and byte-for-byte checks.
These short runs test the workflow; selected-policy quality comes from the
separate physical evaluations.

The presentation update embeds 13 Markdown figures and saves eight instructor
video players. A no-kernel HTML rendering loaded all 18 displayed images and all
eight videos; playback was checked. All 43 code cells passed a local short
integration run. Student activities ship without outputs, and three focused
checks protect the separation between instructor references and student results.
The final archive contains 114 files matching the repository byte for byte.

**Actual Google Colab execution remains untested.** Training times are measured
on the local L40S and exclude setup/evaluation unless stated otherwise.

The layout pass bounds charts to 720 pixels and robot/course figures to 800.
Section 6 uses a focused return curve and time-based tracking plots, avoiding
a flattened equal-scale path panel. The course video row has matched heights.
A real local game embedded in an iframe responded to Forward and Stop; this
checks interactive controls locally, not the Colab proxy or a complete course run.
