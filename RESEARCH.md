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

## Confirmed reset/limit defect (18:19 UTC)

Native end-stop sweeps exposed two interacting errors in `_rebake_joints`:

1. `_joint_q` read the cached body bases from the previous episode, before the
   teleported pose refreshed them. Consequently the physical limits changed
   after every training reset, depending on the preceding failed trajectory.
2. Godot hinge angles are clockwise, opposite the MuJoCo coordinate convention.
   With the hinge rebaked at `q_reset`, its coordinate is `-(q-q_reset)`.
   XML bounds `[lo, hi]` must therefore become `[q_reset-hi, q_reset-lo]`.

The engine convention is documented in the [Godot Jolt hinge implementation](https://github.com/godotengine/godot/blob/master/modules/jolt_physics/joints/jolt_hinge_joint_3d.cpp).
The correction refreshes the body cache before rebaking and maps these bounds.
It changes neither the XML joint ranges nor the controller/torque parameters.
The converter now applies the same mapping at its generated reference pose.

Evidence: `results/research_20260910/joint_limit_probe.json` records the failed
alternatives; `joint_limit_fixed.json` records 36 successful native end-stop
checks across six joints, three initial poses and both bounds. Largest absolute
error: 0.000014025 radians. `tests/test_joint_limits.py` guards repeated resets
in a reused worker, rather than only checking serialized range properties.

Without any weight changes, the original ground-pick policy then achieved 3/3
complete Godot reaches/returns, with mouth-tip minimum near 0.0194 m versus the
MuJoCo reference near 0.0205 m. Native rendered playback reproduced the headless
physical measurements. Crouch tilt also dropped substantially. These are small
development samples, not final reliability claims.

The first residual runs `w01_residual` and `k01_residual` are invalidated for
promotion because their training workers changed effective bounds on reset.
They remain archived as negative evidence. Protocol v3 retains v2 task outcome
definitions, but re-evaluates all baselines under the corrected physical model.
Training resumes from source weights with immutable corrected limits.

Full-network increments use float64 internally for the subtraction and cast the
increment back to float32 before adding the unmodified native ONNX output.
Nonzero trained exports now pass the unchanged 1e-5 deployment tolerance; initial
exports match exactly. This is distinct from re-running the original ONNX in
float64, which would alter its finite-precision behavior.
