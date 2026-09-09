# Godot skill research — 2026-09-10

User-authorized objective: approach the original MuJoCo skill quality in real
Godot/Jolt, continuously iterate for at most eight hours, and extend to all nine
ONNX policies when the evidence supports it.

Start: 2026-09-09 17:39:04 UTC. Hard deadline: 2026-09-10 01:39:04 UTC.
This wall-clock cap includes implementation, diagnostics, training and evaluation.

## Fixed rules

- Preserve the baseline files and their hashes in `results/research_20260910`.
- Compare factory/MuJoCo, factory/Godot, previous finetune/Godot, and candidates.
- Complete task behavior is the objective; training reward is diagnostic only.
- Fix task semantics and physical telemetry before optimizing their rewards.
- Actor interface remains 61 observations / 14 joint-position action offsets.
- No supporting forces, playback poses, or hidden controller assists.
- Ground pick means the official mouth-down reach and return, not object grasping.
- Roller crouch is the phase-controlled crouch/glide/return maneuver.
- Freeze each evaluation protocol version before candidate comparisons. A protocol
  correction requires re-evaluating baselines, never silently moving the target.
- Separate development and final test seeds. Record failures and uncertainty.
- Candidate exports do not become default play policies without promotion.
- Do not overwrite user changes in README.md, SIM2SIM.md or HANDOFF.md.

## First comparison

Standing, walking, left kick: corrected PPO; teacher-regularized PPO; frozen
factory plus bounded residual PPO. Extend to the other tasks as time permits.
Reference anchoring uses deterministic means (factory ONNX has no exploration
distribution). Resume must preserve optimizer, normalizer, LR and RNG state.

The session JSON stores the deadline. Every experiment checks it before sampling
and updating. Results, traces, checkpoints and ONNX artifacts stay under the
session directory. Research source and tests are committed in stages.
