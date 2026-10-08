# Implementation status

- Pinned robot, simulator and FastSAC components are installed and exercised.
- Actor observation restrictions are checked by a real GPU test. The critic
  retains privileged training observations.
- Timestamped experiment storage is implemented; legacy runs were preserved
  during migration. Full resolved settings were not available for every legacy
  run, and their manifests state that limitation.
- Direct-command evaluation records trajectories, speed/yaw errors and falls.
- A procedural desert course contains three obstacles, rough terrain and a
  ramp to a raised finish.
- Browser controls load policies and drive the simulator. A short local
  browser test passed. Reset/timer/finish contracts have unit tests.
- The notebook screens hopping and repeated flat/on-step handovers before
  loading a pair automatically. Failed evaluations stay in the export.
- The same passing walking checkpoint is used for handover tests and gameplay.
  Standalone evaluations record checkpoint hashes and preserve previous results
  and videos when rerun.

The selected fresh walking policy and the retained matched style comparison
passed the full command suite. The lesson includes a complete recorded course
traversal using walking and hopping policies. Course checkpoints reject
shortcuts around the obstacles.

Fine-tuning seed handling and replay continuation have regression tests.
The selected hopping checkpoint passed repeated terrain screens and flat/on-step
handovers. Current notebook integration passed using completed real training,
with fresh hopping checks, live browser interaction and resumable export.
Buttons, keyboard controls, key release and focus-loss stopping were exercised
against the simulator. A further fresh training run is underway.

Some earlier hopping repeats failed; those results remain in the experiment
history. The recordings do not guarantee every new training run will work.
Instructor recordings and student training results are labeled separately.

Actual Google Colab testing is separate from local validation and remains
unperformed. Detailed logs and development records stay in ignored experiment
or artifact folders.
