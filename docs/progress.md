# Implementation and testing scope

The lesson covers robot kinematics, explicit environment settings, FastSAC,
matched scratch-training experiments, hopping, and an interactive course.
The actor receives proprioception and commands; the critic may use privileged
training observations. Approved policy weights and acceptance thresholds are retained.

## Colab reliability audit · October 2026

The submitted L4 notebook required a manual ZIP-upload workaround after setup
hung. Its saved output also contains a missing `Video` import. The original
hanging cell's traceback and the manually uploaded ZIP were not supplied.
The notebook's embedded source matches the previously published lesson.

The L4 baseline completed all 10,000 steps in 689.7 seconds. It had no falls in
50 command trials but failed the forward/fast drift checks. A 3,000-step recovery
(~189 seconds) passed the full suite. Displaying that selected result then failed
on the missing import; the earlier video showed the first training attempt.
This evidence does not establish identical results across GPUs or seeds.

Setup now downloads a versioned public release, verifies its checksum, and reports
progress with bounded retries. It no longer queries the Colab notebook bridge.
Incomplete setup can be rerun; fonts come from the installed Matplotlib package.
Recordings report their checkpoint path, step and digest. Evaluations report case
progress, retain prior results, and keep the existing acceptance thresholds.
Backups include runtime versions and diagnostic logs.

All 86 local tests passed, including GPU environment, checkpoint identity,
interrupted setup, download failures, and game reset/finish contracts. The right-knee
slider was checked against actual right-foot model coordinates and exercised in a
browser. Its camera shows the moving leg and highlights the hinge. Static left-leg
illustrations remain separate and labeled.

A fresh local environment installed successfully, including repair of an interrupted
Git checkout. Existing package caches were available. A local integration copy ran
all 37 code cells and eight optional hopping stages at 60-step test budgets; its
backup contained eleven resumable runs. This establishes workflow execution, not
learned policy quality. Additional full-budget reproduction checks are in progress.

**The revised workflow has not run in an authenticated Colab GPU runtime.** The
submitted L4 output is evidence about the prior workflow. New GPU tests use a local
L40S; concurrent jobs and cached installation affect measured wall-clock times.

## Teaching media and earlier results

Figures and eight silent animations use public PNG/GIF URLs inside Markdown.
All code cells ship without outputs. A previous public Colab browser check loaded
all 22 images and confirmed the opening animation advances without a runtime.
Local rendering alone did not establish compatibility with Colab's renderer.

The selected course recording finished in 53.56 simulated seconds with visual
steering and no falls. On the isolated 25 mm step, walking crossed 0/6 trials and
hopping 3/3. Fixed command replays sometimes missed the route; these are separate
from the successful steered recording. See the results guide for selection limits.

The visible level gaze, neck bounce and idle glances are renderer gestures;
existing physical head targets are retained. Tests verify rendering leaves simulator
state and actor observations unchanged. The approved walker passed 50 command
trials with physical head motion enabled, with render-only gestures, and with
both disabled. This does not imply every newly trained policy passes.

Raw checks, failures, checkpoints and execution artifacts remain gitignored.
Curated lesson media retain their source and W&B provenance beside the recordings.
