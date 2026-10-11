# Implementation and testing scope

The lesson covers the actual robot's kinematics, environment settings, FastSAC,
matched scratch-training experiments, hopping, and the interactive course.
The actor receives proprioception and commands; the critic may receive privileged
training observations. Selected policy weights and training settings are unchanged.

Teaching figures and eight silent example animations use public PNG/GIF URLs
directly in Markdown. The former Jupyter attachments rendered locally but
appeared broken in Colab; local rendering did not establish Colab compatibility.
All code cells ship without outputs. The hopping explanation groups the existing
eight stages into four phases; it does not change their configuration. Joint
annotations use model anchors, independently checked against rendered markers.
The visible left leg is highlighted; schematic collars show its hinge axes.
All five projected centers agree with rendered markers to within 1 pixel.
Related plots and examples share rows; standalone figures are centered.

Local checks passed 75 tests, including GPU environment and game contracts.
All 29 protected checkpoint/configuration hashes match their originals. The nine
curated MP4s decode successfully. The earlier browser check loaded all 22 attachments locally,
but did not test Colab's Markdown renderer. After replacing attachments with
public URLs, an actual Colab browser check loaded all 22 images and confirmed
the opening animation advances without a connected runtime (revision c6d0ff6). The live
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
The new visually steered course finished in 53.56 simulated seconds with all four
checkpoints and no falls. Two fixed replays missed the route; an earlier manual
attempt fell after passing the finish. These remain separate from the successful
visually steered recording.

The single-file archive now lives in notebook metadata, removing encoded data
from displayed code. Loading a renamed local notebook passed; the Colab bridge
was checked with a simulated frontend response, not a real Colab session.

**The level head gaze and visible gestures are cosmetic.** They change only
the renderer's CPU state; physical head targets and leg control are retained. Moving commands add a gentle
neck bounce; idle commands blend into small glances and fidgets. A render check
measured more than 1 cm of vertical head travel while gaze remained within
1.1 degrees of horizontal. Tested physical pose offsets
reduced locomotion quality and were rejected. Direct checks found no change to
simulation state or actor observations during rendering. A validated physical
neutral-pose correction remains unfinished.

Raw checks, failed trials, and execution artifacts stay in ignored folders.
Curated recordings retain their source and W&B provenance beside the media.
**Actual Google Colab execution remains untested.** Reported training timings
come from the local L40S and exclude setup/evaluation unless stated otherwise.
