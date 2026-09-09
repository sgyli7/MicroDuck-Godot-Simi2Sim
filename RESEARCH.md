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

Protocol v4 adds two adversarial outcome guards: a kick that falls and later
stands up is not a clean kick, and several revolutions are not a single roulade.
All v3 baseline traces were re-scored under v4, preserving their original files
and simulator fingerprints. Ongoing v3 pilot exports will also be re-scored
before selection. Candidate ranking prioritizes complete success rate, then
the continuous diagnostic score; partial progress never constitutes promotion.

## Entry-distribution correction (18:45 UTC)

Upstream `infer_policy.trigger_behavior` preserves `last_action` when switching
from standing to a kick/roulade. A cold reset with a zero action history is not
equivalent. Native tests verified the difference: source MuJoCo left-kick
completion rose from 1/3 cold starts to 3/3 actual standing-policy handoffs;
the first adapted kick finished 3/3 cold starts but 0/3 handoffs. It is therefore
not deployable despite its cold-start result.

The evaluator now explicitly labels `entry=reset` and `entry=standing`, and
supports a combined `both` suite. The latter performs one second of real source
standing control, preserves the action history, places the ball at the skill
trigger, then evaluates the candidate alone for the full task window. The
standing controller never assists within that evaluated window. All original
cold-start results remain archived as a separate stress test.

New training trials use a 50/50 mixture of those entry distributions. Independent
standing warmups are stepped together for efficiency. Finite one-shot maneuvers
end naturally at their task deadline; only continuous-control tasks bootstrap
artificial time-limit truncations. These changes and source checkpoint ancestry
are recorded per run; earlier and later trial scores must not be conflated.

The exact upstream bilateral transform is also being evaluated as a right-kick
initialization. It preserves the 61/14 interface and has a zero-error 10,000-input
ONNX check. It is a candidate policy transform, not evidence of physical symmetry
or a substitute for separate right-foot contact/direction tests.

## Runtime contracts and recovery curriculum (19:05 UTC)

The play state machine now gives phase zero on the first skill action, runs
ground pick for four seconds and kicks/roulade/crouch for five, and preserves
the preceding action history. Roller idle uses the roller policy with a zero
twist; the crouch policy is a separate five-second phase action on key 2/Y.
Normal MicroDuck play loads the free-ball scene and places the ball at each
kick trigger in the current heading frame, matching source play semantics.

Native capture exposed two rendering omissions in the new ball scene: OBJ
assets needed Godot import, and a colliding primitive sphere had no visual
mesh. The converter now emits a SphereMesh and play imports missing assets.
The current generated ball scene received only the visual nodes/resources;
its existing physical nodes/resources and physical spec are preserved. The
same kick trace before and after this display fix has identical measurements.
Generated JSON mapping descriptions change when regenerated by the corrected
converter, but their body/joint/geom/actuator data are unchanged.

The mixed-entry left-kick checkpoint `k07_mixed_resume/iteration_00163.pt`
passed 6/6 initial and 16/16 additional development episodes (both entries).
This establishes useful progress, not final held-out reliability. Subsequent
checkpoints reduce action variation but still turn more than source MuJoCo;
the categorical score saturates, so raw smoothness, tilt, direction and yaw
metrics remain necessary for final selection. An optional recorded heading
penalty and stronger existing smoothness/direction weights support polishing
trials without changing the default objective or evaluator.

`curriculum.py` constructs 184 physical starts from eight successful source
MuJoCo rolls using training seeds 40000–40007. Each retains full generalized
position/velocity and the actual last action. The source progress and pivot
history initialize the training critic/reward accumulator so replaying earlier
rotation does not earn new progress. No source actions run after that reset.
The first controlled comparison uses 50% such training starts and 50% ordinary
mixed entries. Evaluation always remains a full roll from normal cold and
standing entries. Tests verify pose/history transfer and clean accumulator
reset; a short end-to-end smoke run also completed with exact initial export.

## Reference clarification and walking hypotheses (19:18 UTC)

Upstream roller play sets passive-wheel frictionloss to 0.003 at runtime,
whereas its training XML leaves it zero. Both source MuJoCo references are now
retained explicitly: `source_play_reference/` applies the script override and
records it in the fingerprint; the original XML reference is unchanged.
Godot physics is unchanged. The override improves source glide/idle and still
leaves large source turn-rate errors. It also reduces crouch travel, so the
phase/crouch quality and actual glide distance must be reported separately.

The low-parameter command-conditioning search changes only an internal affine
layer on the three velocity-command features. Public observations and requested
velocities used for evaluation do not change. The exported ONNX includes the
adapter, source hash and coefficients, and its 10,000-input parity is exact.
The first probe's decimal filenames collided; it is explicitly invalidated and
re-run with unique filenames under `conditioning_probe_v2`. These scans are
development diagnostics, not evidence of general task completion.

An additional architecture hypothesis addresses the residual bound: on the old
walking policy's idle trajectories, 57.7% of per-joint standing-teacher action
differences exceed the 0.2-radian bound. `distill_walk.py` therefore teaches a
full-network increment from real standing/old-walking demonstrations generated
on separate training seeds 60000–60002. Expert switching is confined to data
collection. Physical evaluation executes the single exported student for the
whole trajectory, including idle and transitions. A held demonstration seed
checks supervised error; the existing physical development suite checks actual
behavior. Final evaluation seeds 1000+ remain unused.

Protocol v5 closes an idle loophole in locomotion evaluation: a small nonzero
velocity/yaw error can accumulate large drift over ten seconds. After allowing
0.5 seconds of braking in each idle interval, completion now also requires less
than 5 cm travel and 5 degrees yaw drift. Original v4 results remain intact;
`rescore.py` writes adjacent `summary.v5.json` files from their saved traces.
Kicks, roulade, standing and posture outcomes are unchanged. In-progress v4
experiments retain their recorded version and are rescored before comparison.

The 2,000-step distilled walking student achieves six clean idle episodes under
that stricter gate, with negligible travel and mean yaw drift about 0.3 degrees.
Its source gait still under-tracks translational commands. Unbounded 2x command
conditioning improved slow walking but caused falls at the fastest test; those
candidates are not suitable for promotion. Bounded conditioning on the distilled
student is evaluated separately across the complete development schedule.

New PPO logs separate physical reward from time-limit value bootstrapping and
include critic error, explained variance and increment size. Earlier continuous
task logs reported the bootstrapped training target as `reward`; physical
outcomes, rather than those reward curves, have always determined selection.

## Native roll review and alignment guard (19:39 UTC)

The reverse curriculum produced a cold-start roll that recovered to a stable
stance, but native video exposed a 155-degree change in final heading. The
corresponding source MuJoCo roll ends about 14 degrees from its initial heading.
It is recovery progress, not a satisfactory straight forward roll. Also, raw
unwrapped Euler yaw during inversion is misleading (the source trace reports
374 degrees while its actual final heading error is only 14 degrees).

Protocol v6 therefore adds an explicit wrapped final heading error and requires
it below 30 degrees for roulade completion. All six source development rolls
remain within 3–15 degrees. Other task criteria are unchanged from v5. Archived
traces are rescored into `summary.v6.json`; earlier outcome versions are retained.
The next roll trial adds a recorded aligned-landing reward, excess-rotation
penalty, leg-only final-pose term and the upstream bilateral consistency loss.
The final tucked head remains available as a physical cue distinguishing the
end of the maneuver from its initial standing pose.

A direct bilateral-mean ONNX ensemble of the factory and first recovery actor
was also tested; neither completed the roll. These are negative architectural
probes, not replacements for training. A three-iteration smoke run verified
the new regularized PPO path and unchanged export tolerance.

The distilled walker with bounded internal conditioning completes 42/72 strict
development episodes, has no falls, and retains six clean idle cases. It still
under-tracks fast travel and lateral motion. The first PPO continuation failed
before collecting data because nested ONNX metadata keys were duplicated;
the exporter now replaces current-stage metadata uniquely, with a nonzero
second-adaptation parity test. The continuation is retried as `wd02`.

The first alignment trial (`r04`) exposed another scale issue: an unbounded
squared excess-rotation penalty dominated the return after repeated rolls
(mean term around -157 versus ordinary positive terms around 1–3). It was
stopped after four minutes. `r05` restarts from the same `r03` parent using a
penalty normalized by pi and capped at one before its recorded multiplier.
Physical rewards and critic fitting returned to a useful range. A regression
check prevents this penalty from growing without bound.

## Interactive walking and bounded-roll result (20:04 UTC)

A small recorded command coupling (internal yaw command += 0.9 * forward
command) on `wd02` iteration 80 improves strict development completion to
45/72, score 0.5931, with zero falls. It corrects backwards heading and preserves
clean idle/turns. Fast travel, lateral travel and stop transitions remain weak.
`wd02` continued PPO eventually regressed to 17 falls in its iteration-154
development evaluation despite growing training reward. Later weights are not
promoted. This illustrates why checkpoints are selected by physical outcomes.

`wd03` starts from the verified coupled candidate. Training-only interactive
command tapes replace 25% of constant/standard-sequence episodes: seeded
0.5–2.5 s segments, bounded continuous twist commands, 0.1 s ramps and idle
segments. Both engines receive identical tapes; standing entry and action
history remain real. The existing fixed development/final tests are unchanged.
An explicitly recorded yaw reward variance is tightened from 0.18 to 0.04.
Contract checks cover reproducibility, bounds, ramps, both backends and reset
back to ordinary commands. A short PPO/export/evaluation smoke run validates
the complete path before the longer trial.

`r05` fixes numerical reward domination but still completes no normal-entry
rolls in three checks: trajectories repeat several revolutions. It is stopped
for the next architectural trial. `r06` uses full-network adaptation from
`r03`'s actual recovery parent, rather than inheriting `r05`'s degraded policy.
The 30-degree final heading guard is retained.

## COM velocity correction (20:07 UTC)

A fresh independent kinematic check found that `mj_objectVelocity(mjOBJ_BODY)`
already reports the body inertial COM velocity. The legacy state transfer
incorrectly treated it as the regular body-origin velocity and added omega ×
(xipos - xpos) a second time. At a 4 rad/s forward angular velocity this created
0.091 m/s trunk and 0.131 m/s jaw transfer errors. The MuJoCo COM Jacobian times
qvel matches the unmodified API velocity to numerical precision on every body.
The engine source explicitly distinguishes BODY (xipos) and XBODY (xpos):
https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_core_util.c

The backend transfer and reference COM metric now use the unmodified BODY
velocity; the ordinary backend state also stops using subtree-COM `cvel` as
trunk velocity. A random articulated-velocity regression checks every body's
linear/angular transfer against its COM Jacobian. Normal cold/handoff Godot
rollouts start from zero angular/joint velocities, so their physics and
outcomes are unchanged. Source mid-roll starts are materially affected.
`r06` is stopped and repeated as `r07_com_transfer`; previous curriculum runs
remain documented negatives under their original reset convention.

Protocol v7 keeps v6 outcome thresholds and records the corrected COM velocity
contract. Godot archives can be rescored unchanged; old MuJoCo traces require
fresh reference rollouts rather than silently relabeling the wrong velocities.
Both the source XML and source-play wheel-friction references are re-run.

## Explicit time input for a finite roll (20:15 UTC)

A separate architecture experiment (`r08_time_input`) gives the learned actor
a monotonic elapsed-time input in the otherwise zero first command slot:
cmd[0] = clip(seconds_since_trigger / 5, 0, 1). Input/output sizes stay 61/14.
This lets the policy distinguish standing before its roll from standing after
it. The ONNX declares `sim2sim_time_input_s=5`; training, independent evaluation,
native capture and interactive play read the same model contract. Factory and
previous policies without this metadata continue receiving their original
commands. The five-second physical task and completion thresholds do not change.

The frozen parent sees zero command slots inside the ONNX; only its trainable
increment receives elapsed time, with a unit normalization denominator for that
new feature. The critic uses the same scaling. Initial output equals the
zero-command parent, and a nonzero full-network update passes the unchanged
export tolerance. Training-only mid-roll states retain source elapsed time;
normal resets and real standing handoffs restart it at zero. Regression tests
verify that play and training count the first action at exactly time zero.
No pose sequence is played back: all 14 action offsets come from the exported
network under native Jolt dynamics. `r07` remains the no-time-input comparison.

Walking's coupled candidate also completed 119/192 additional development
episodes (seeds 110–117, both entries), with zero falls. Idle completed 15/16;
transition stopping still fails. These additional development samples do not
replace the final unseen-seed evaluation.

## State aggregation and motion feedback (20:23 UTC)

`distill_transitions.py` collects walking states under the student itself,
then labels zero-command states with factory standing and moving states with
the verified parent walker. It aggregates two successive student distributions
from training seeds 80000+, including random command tapes and real handoffs.
The exported single student is evaluated for every full trajectory. Balanced
idle/moving supervision is a training objective, never a runtime policy switch.
The first 1,000-step student completes 48/72 development cases with no falls.

A bounded roller yaw-gain search on separate development seed 300 tested
1, 1.5, 2, 2.5, 3, 4 and 5 times the old policy's command. Larger gains caused
falls; gain one remains best, so no conditioned roller is promoted. The full
old-roller comparison is retained under `roller_conditioning/best_full`.

For the next timed-roll trial, `roll_motion.py` records eight successful original
MuJoCo training-seed rolls, builds a median joint-pose / gravity-direction /
height / net-rotation reference, and supplies only reward errors to PPO.
The reference never sets native states or outputs actions. Its hash and source
physics are recorded. A regression and short PPO/export run check retained
curriculum time and nonzero updates. `r09_timed_motion` will compare this dense
feedback with the time-input-only `r08`. Evaluation still imports neither
training rewards nor motion-reference code.

The machine has 20 CPU threads. A third bounded lane runs `r08` with eight
worlds and 1,024 steps per update, preserving the same 8,192-sample update size
while the two existing lanes continue. CPU throughput is monitored; this is
independent simulator processing, not additional decision-making agents.

## Native landing demonstrations and explicit neural experts (20:43 UTC)

The on-policy walking distillation finishes at 54/72 basic development cases
and 142/192 additional cases, without falls. It solves all 16 extra idle cases
(max travel about 1 mm, max yaw 0.5 degrees), and 14/16 command-sequence cases.
Fast forward and lateral tracking remain weak; fast-forward yaw also needs
polish. This is a substantial improvement over the previous walker, not a
claim that every requested twist is achieved.

Native roll diagnostics show a first revolution followed by repeated launches.
For training demonstrations only, the partially adapted roll expert handed
control to the original balance expert after actual inversion and a supported
upright return. Nine of 24 training-seed demonstrations complete the strict
physical task. A time-aware monolithic student trained on these demonstrations
does not yet reproduce reliable recovery. A gated-increment ablation preserves
the parent's launch exactly for the first 1.8 seconds and also remains
unqualified. These teacher demonstrations are not counted as candidate results.

A further architecture probe (`roll_experts.py`) explicitly packages both
frozen neural experts and their time-input blend into one ONNX. It is a
**two-expert actor**, not a claim of monolithic distillation. The simulator
executes that single graph throughout every five-second evaluation, with no
external policy handoff or physical assistance. The ONNX records both expert
hashes and its blend interval; parity over 10,000 inputs is exact.

A 2.30–2.45 s blend achieves six stable single-revolution landings on the basic
development set, but only one also finishes within 30 degrees of the initial
heading (the others are about 31–78 degrees away). Thus it improves recovery
but is not reliable enough to promote. An alternative 2.10–2.25 s blend fails
the normal development set. `r10_expert_residual` trains a recorded, time-gated
residual on the stronger two-expert actor. `r08` and `r07` are stopped after
several consecutive unqualified checks; `r09` retains the separate source-motion
feedback hypothesis. All architectures retain the same physical outcome gates.

Time gating is part of the exported neural graph. Resume restores its recorded
configuration and rejects an incompatible change, while old checkpoints with
no gate remain compatible. Tests verify exact unchanged launch actions and
nonzero late adaptation in the actual exported model.

## Heading error is introduced before landing (20:47 UTC)

A diagnostic based on the trunk lateral axis, which remains meaningful through
a sagittal inversion, locates most yaw drift around 1.0–1.5 s. For cold seed
100 the error grows from -9 to -41 degrees before the late correction gate can
act. This motivates `r11_early_yaw_control`: a small 0.1-radian residual is
enabled from 0.2–0.5 s and a bounded world-vertical angular-velocity cost is
added. A regression confirms that this cost leaves pure forward rotation
unpenalized. The late-only `r10` remains a separate comparison.

`r09` is stopped after three normal-entry zero-completion checks; its dense
reference experiment is an unsuccessful bounded-budget trial, not a claim
that motion-reference learning cannot work. The current strongest roll
architecture has stable single-revolution recovery in all six basic cases;
only initial-heading recovery prevents most from qualifying.

## Direct heading observability and portability (21:06 UTC)

The heading objective previously depended on a trigger-relative angle that
neither actor nor critic received directly. Gyro/action history can help
implicitly, but identical balanced poses can still have different heading
returns. A separately declared variant supplies sin/cos of the trunk lateral
axis's angle relative to the trigger frame in cmd[1:3]. This axis stays
meaningful through a sagittal inversion; tests cover an entire forward turn
and reflected headings. The actor still has 61 inputs and 14 outputs, but
this variant explicitly changes two previously unused command slots.

`sim2sim_heading_input=lateral_axis_sin_cos` records that contract. A wrapper
masks the new slots for its frozen parent, and the trainable actor/critic use
unit scaling. Bilateral consistency negates heading sine and preserves cosine.
Play, generic rollout, native capture and independent evaluation derive the
same input from actual simulator orientation and the trigger frame. Consumer
code must honor the model metadata; the models do not infer elapsed time or
initial heading from a bare zero-filled observation. No physical control is
performed by the input adapter.

The original time-mixture parent repeats identically in two fresh full
development suites. Its heading-ready wrapper also gives identical trajectories
and outcomes (1/6): enabling the input itself does not improve the model.
A tiny trained heading-aware increment changes the contact trajectory enough
to achieve 4/6 basic and 8/16 extra cases. This sensitivity is why broader
unseen tests remain essential. `r12` is queued from that verified short-run
candidate. Meanwhile, early yaw-spin adaptation without direct heading input
(`r11` iteration 50) reaches 5/6 basic and 10/16 extra cases. Neither is yet
qualified at original reliability.

The walking candidate's fast-forward yaw bias is corrected by a recorded
input hinge: internal yaw += -3 * max(public_vx - 0.3, 0). Slower commands are
bitwise unchanged; 10,000-input parity is exact. The full suite remains 54/72
with improved continuous score 0.6146 and no falls. Absolute fast/lateral
velocity tracking remains below the requested target.

Measured CPU inference is comfortably within the 20 ms control interval:
the distilled walking graph is about 7.25 MB and 0.17 ms p99, and the two-expert
roll graph about 4.87 MB and 0.106 ms p99 in a 2,000-call one-thread sample.
These are inference-only timings under concurrent training, not full simulator
round-trip latency.

## Checkpoint continuity and worker recovery (21:22 UTC)

The actor/critic optimizers and global Python/NumPy/Torch RNGs were already
restored, but the vector environment's independent generator and episode
counter restarted. Checkpoints now also preserve that generator and counter.
Resumes intentionally begin fresh physical episodes; they do not pretend to
restore Godot's hidden solver/contact state. Legacy checkpoints use a disjoint
episode-seed range above their maximum possible prior episode count. Configs
record this distinction and hashes of observation and backend transfer code.

A native reset test verifies that a restored generator produces the same next
condition, RNG state and fresh observation. End-to-end save/resume smoke runs
advance iteration 2 to 5 and episode counter 2 to 4. A legacy timed-model resume
retains its 0.2–0.5 s gate and advances into a new seed range (counter 1030).

At 21:21, lanes A/B and their coordinators were absent without Python errors
or completion records. The original logs and checkpoints remain unchanged;
wh01 resumes at iteration 30 for 15 minutes, and r11 at iteration 70 for
25 minutes in new run directories. Lane C continued normally. The precise
external process-exit cause is unknown. All runs retain the original 8-hour
wall-clock cap; recovery does not extend it.

For r12, the stronger r11 iteration 50 parent replaces the earlier 4-step
heading-aware candidate: 5/6 basic and 10/16 extra cases versus 4/6 and 8/16.
Its wrapper masks the new heading slots for the parent while preserving time.
The new heading-aware residual uses the same small bound 0.1 and std 0.02 as
r11, making the heading-observability comparison more focused.

## Bilateral roll ensemble and real skill exits (21:31 UTC)

A renewed symmetry probe on the much stronger r11 iteration 50 actor succeeds
where the early factory/recovery probes failed. The single ONNX averages the
actor with its exact reflected counterpart; its architecture is explicitly
a bilateral ensemble containing four frozen neural expert branches plus the
learned corrections. Ten thousand input checks show zero export discrepancy
and zero bilateral discrepancy. Time remains an explicitly required input.

After 6/6 exploratory training-seed rolls, it passes 20/22 development rolls
(100–102 and 110–117, both entries), continuous score 0.9776. All 22 complete
one revolution and stand; two miss the 30-degree heading gate at 31.65 and
33.05 degrees. It is a stronger candidate, not yet proof of source reliability.
All 22 additional three-second handoffs to the factory standing actor remain
stable without resetting physics or action history. Maximum post-exit motion
is 3.8 mm and 2.08 degrees. Primary metrics match the earlier standalone
evaluations exactly; post-exit assistance cannot turn a primary failure into
a success. This separate check is implemented in research/handoff_check.py.

Native frame inspection and trace analysis reveal remaining quality gaps:
the ensemble settles around 2.83 s versus original MuJoCo 1.64 s, and its
rotation frontier is 7.18–7.90 rad versus 6.31–6.46 rad. It briefly leans
forward again before recovering. These continuous gaps remain visible even
when the binary task gate passes. Earlier internal balance blending was
therefore tested at 1.75, 1.9, 2.0, 2.1, 2.2 and 2.3 s on separate training
seeds. The first four degrade recovery; 2.2 passes 5/6 and 2.3 passes 6/6.
No timing change is selected. A separate factory-roll-plus-balance bilateral
ensemble at six earlier timings also passes 0/6 throughout; standing after
an incomplete maneuver does not count.

The mirror exporter now preserves phase sine for phase-conditioned maneuvers
and heading cosine for heading-aware rolls, matching the actor-training
reflection. Unit checks cover those input semantics and nested blend retiming
against independently rebuilt graphs. Existing right-kick transforms are
unchanged. The fast-walk yaw correction passes an additional 16 native tests
without falls (yaw RMSE 0.059 rad/s); speed remains 0.172 m/s and still fails
the strict 0.4 m/s target.
