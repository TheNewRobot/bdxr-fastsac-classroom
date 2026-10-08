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

The selected walking baseline and both retained variations passed the full
command suite. Their unchanged comparison continuation failed fast-walking
drift and is labeled accordingly. The lesson includes a complete recorded course
traversal using walking and hopping policies. Course checkpoints reject
shortcuts around the obstacles.

Fine-tuning seed handling and replay continuation have regression tests.
The full notebook completed fresh local training and evaluation with no cell
errors. Its walking baseline passed after unchanged continuation; its hopping
policy passed repeated terrain screens and flat/on-step handovers. That pair
completed the course under visual operator commands. Pauses between commands
mean this is a simulated traversal time, not a real-time human play test.

The current notebook's executable content matches the fully executed version.
Its game and export cells were also exercised directly with the fresh policies.
Live Chromium tests covered checkpoint loading, buttons, keyboard controls,
key release, focus-loss stopping and walk/hop changes. The postplay backup
passed ZIP integrity and file-by-file checksum verification, including all
training checkpoints and game recordings.

The fresh style continuations failed drift checks; the notebook reports that
and keeps the passing baseline. Earlier retained instructor variations passed
the full suite and remain a separate matched comparison. Failed training and
evaluation attempts are preserved. Actor inputs and acceptance limits were
not relaxed to select a policy.

Some earlier hopping repeats failed; those results remain in the experiment
history. The recordings do not guarantee every new training run will work.
Instructor recordings and student training results are labeled separately.

Actual Google Colab testing is separate from local validation and remains
unperformed. Detailed logs and development records stay in ignored experiment
or artifact folders.
