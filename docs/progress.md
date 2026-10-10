# Implementation and testing scope

The lesson covers the actual robot's kinematics, environment settings, FastSAC,
matched scratch-training experiments, hopping, and the interactive course.
The actor receives proprioception and commands; the critic may receive privileged
training observations. Selected policy weights and training settings are unchanged.

Teaching figures and eight silent example animations live in Markdown attachments.
All code cells ship without outputs. The hopping explanation groups the existing
eight stages into four phases; it does not change their configuration. Joint
annotations use model anchors, independently checked against rendered markers.
Related plots and examples share rows; standalone figures are centered.

Local checks passed 70 tests, including GPU environment and game contracts.
All 29 protected checkpoint/configuration hashes match their originals. The nine
curated MP4s decode successfully. Browser inspection loaded all 22 Markdown images
with outputs cleared and confirmed that the opening animation advances. The live
local game responded to Forward, Stop, pause, and the overview camera.

The local integration copy executed all 37 code cells, including all eight
hopping stages at 60-step test budgets. Its backup contains eleven resumable
runs with no duplicate entries. This checks workflow and export, not learned
policy quality; the test copy retains failed short-run verdicts. Dependencies
were reused. Final media were checked separately, and the 119 bundled files
match the repository byte for byte.

Refreshed recordings include five forward starts for eight policies/control
recordings, plus a second camera view of the locked walker: 45 trials without
falls. On the isolated 25 mm step, walking crossed 0/6 trials and hopping 3/3.
The new visually steered course finished in 53.06 simulated seconds with all four
checkpoints and no falls. A fixed replay missed the finish; that failure remains
separate from the successful visually steered recording.

**The level head gaze is cosmetic.** It changes only the renderer's CPU state;
physical head targets and leg control are retained. Tested physical pose offsets
reduced locomotion quality and were rejected. Direct checks found no change to
simulation state or actor observations during rendering. A validated physical
neutral-pose correction remains unfinished.

Raw checks, failed trials, and execution artifacts stay in ignored folders.
Curated recordings retain their source and W&B provenance beside the media.
**Actual Google Colab execution remains untested.** Reported training timings
come from the local L40S and exclude setup/evaluation unless stated otherwise.
