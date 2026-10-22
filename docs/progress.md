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
progress with bounded retries. A trickling response cannot keep a buffered read
open indefinitely; a regression test checks this timeout case. Setup no longer
queries the Colab notebook bridge.
Incomplete setup can be rerun; fonts come from the installed Matplotlib package.
Recordings report their checkpoint path, step and digest. Evaluations report case
progress, retain prior results, and keep the existing acceptance thresholds.
Checks save trajectory measurements without rendering unused videos. The selected
video is a separate rollout of the assessed weights, verified by checksum before
recording. Backups include runtime versions and diagnostic logs.

All 98 local tests passed, including GPU environment, checkpoint identity,
interrupted setup, download failures, and game reset/finish contracts. The right-knee
slider was checked against actual right-foot model coordinates and exercised in a
browser. Its camera shows the moving leg and highlights the hinge. Static left-leg
illustrations remain separate and labeled.

A fresh local environment installed from the public release, using existing package
caches. Interrupted Git checkout recovery was tested separately. The bootstrap
release ran all 37 code cells and eight optional hopping stages at 60-step budgets;
its backup contained eleven resumable runs and setup diagnostics. A metadata-free
copy of the revised notebook also downloaded successfully and preserved results on
rerun. Short integration runs establish workflow execution, not learned policy quality.
An additional local run exercised both recovery branches: all 38 code cells
completed and the backup contained 13 resumable runs. That check reused installed
dependencies and used short budgets; it did not retest installation or policy quality.

One fresh seed-7 run reached its 9,000-step checkpoint in 579 seconds; that snapshot
passed all 50 command trials twice, including recording through the notebook helper.
It also passed all 50 trials from additional seed-2000 starting states. The evaluator
now accepts an explicit `--seed`; its classroom default remains 1000.
Its final 10,000-step weights and a 3,000-step recovery failed drift limits. The
notebook now checks up to two earlier snapshots before additional training, retaining
all reports and displaying the exact selected checkpoint. A second seed-7 run,
its recovery, and a separate optimizer/replay continuation missed drift limits.
The earlier level-feet recipe with yaw-bias weight 6 produced flatter stepping
but failed drift limits; it is not a qualified replacement.
A shared-GPU seed-8 run stopped at 7,399 steps after 901.4 seconds and failed motion
checks. Completing its budget in a separate resumed run still failed command checks.

The full-budget notebook validation also needed recovery. Its 10,000-step baseline
took 847 seconds, followed by 172 seconds for a 3,000-step recovery with unchanged
behavior settings. The notebook selected the recovery's 2,000-step snapshot:
it passed 50 command trials and 50 additional seed-2000 starts. The later checkpoint
failed drift limits. About 17 minutes of training were spent, excluding setup and
evaluation; a 15-minute successful result is not guaranteed.

An additional recovery option uses yaw-bias reward weight 10 while keeping the
original baseline and first recovery unchanged. It qualified two failed training
parents: seed 7 at 3,000 recovery steps (176 seconds), and seed 42 at its saved
2,000-step snapshot. The latter's final weights still failed. The seed-7 recovery
also passed 50 additional trials with evaluation seed 2000.
This bounded extra attempt runs only after the existing checks and recovery fail,
and only when the original final policy failed drift checks alone. It starts from
that original attempt and retains the same acceptance limits. Applying the same
refinement to an undertrained seed-8 parent did not fix its falls or command errors;
the notebook therefore excludes such parents from this targeted refinement.

Two local simulator probes used identical seeded initial states and fixed actions,
without learner updates. First-step differences were about 3e-8 metres in root
position and 2e-7 in actor observations; later reset timing diverged. This demonstrates
local environment nondeterminism, not the precise cause of the submitted L4 drift.
Matching a seed therefore does not establish identical training trajectories.

A matched seed-7 level-feet recipe uses yaw-bias weight 10 in both fresh runs,
with sole-level weight 0 versus 0.75. Both 10,000-step final policies passed
50 command trials and 50 additional seed-2000 starts. Mean swing-foot tilt
decreased from 9.85° to 4.58°, with similar clearance. Training took 1,017 and
753 seconds on the shared L40S. The level-feet activity now uses this recipe
with a 25-minute wall-clock cap to avoid truncating its fixed step budget.
The original walking baseline and other style presets are unchanged. This
is one matched training pair, not proof that every new run will qualify:
a separate control repetition qualified at 8,000 steps but failed at 10,000.

A second fresh matched pair, executed through the revised notebook activity,
also passed 50 command trials and 50 additional starts per policy. Training took
578 seconds for the control and 524 seconds for the variation. Swing-foot tilt
decreased from 8.53° to 4.94°; 95th-percentile clearance increased from 24 to 43 mm.
These are repeated local results, not a guarantee across seeds or GPU types.

A split full-budget local execution completed setup, baseline/recovery, the matched
gait activity, all eight hopping stages, game startup, and backup export. The
baseline was trained in the first part and reused by verified checkpoint hash
when the revised gait activity resumed. All 32 cells in that continuation completed
without exceptions. The exported backup contains 12 resumable training runs.

Hopping used 24,000 steps and 1,256 seconds of training. The final stage missed
one slow-motion trial; an extra fine-tune and a final-stage seed variation introduced
falls. The earlier **natural-hops** stage passed all 15 terrain-screen trials and
all ten flat/on-step handover trials with the newly trained walker. The notebook
now checks the last three curriculum stages, newest first, and shows the selected
weights' verified video and that stage's curve. Training settings and acceptance
criteria are unchanged, and failed checks remain in the history. The fresh pair
also passed local browser control and mode-switching checks. This does not
establish full-course traversal.

The updated selection flow also completed a 38-cell short-budget integration
with both recovery branches and all hopping stages. Its backup contains 13
resumable runs. The selected-hopper video branch was added during that test and
then executed separately with the real qualified weights, hash check, video,
and curve. Short-budget execution establishes integration, not policy quality.

**The revised workflow has not run in an authenticated Colab GPU runtime.** The
submitted L4 output is evidence about the prior workflow. New GPU tests use a local
L40S; concurrent jobs and cached installation affect measured wall-clock times.

## Teaching media and earlier results

Figures and eight silent animations use public PNG/GIF URLs inside Markdown.
All code cells ship without outputs. The updated public notebook loaded
all 22 images in Colab and its opening animation advanced without a runtime.
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
