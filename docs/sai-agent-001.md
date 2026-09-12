# Sai_Agent_001 profile

```sh
uv sync --extra sai
uv run --extra sai sim2sim-play --robot Sai_Agent_001
```

The optional profile pins the Sai source/model/policies to commit
`6eb18b2abf97ce6de11a18daba93729848736adf`. It supplies its own 26-body,
23-hinge/two-slider Jolt scene, 82D observation and 16D mixed position/velocity
controller. MicroDuck remains the default and retains its existing scene/policy
contracts. The launcher starts the policy service automatically; no separate
training server is needed.

W/S forward/reverse, A/D yaw, hold Shift to crouch and release to stand, R restart, Esc quit.
Godot 4.7.2 must be installed and on PATH (`--godot-bin` selects another location).

```sh
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --stairs .02
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --headless --case W --output results/sai-W.json
```

The pinned package passed eight flat keyboard cases and four-riser 20/40 mm
ascent/descent in Godot/Jolt. It uses simulated ground raycasts; camera/VLA terrain
reconstruction and 60 mm stairs are not validated. A separate full-MuJoCo tread/yaw holdout passed 14/16 cases; this is not arbitrary-terrain acceptance. Other MicroDuck
`--scene` presets do not automatically convert into the Sai profile; this first
adapter launches the package's own scene. The SO101 and cargo joints remain real
articulations, not fixed display parts.

## Physical pickup and loaded transport

```sh
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --task cargo
```

The new pinned alpha includes the separate 100 g box pickup, opposing-pad clamp
and loaded crawl task. The public package passed 18/25 mm obstacles in MuJoCo
and Godot with no lost/unclamped physics steps. `--no-clamp` intentionally stops
before transport and exits 3; no object is welded or moved by pose assignment.
The pickup uses scripted model-state IK, not VLA; loaded crawl preserves its
frozen 57D actor and does not replace the 82D keyboard controller.

The launcher recreates both physics and controller state on R. Visible playback
also disables VSync and caps rendering to 60 FPS to avoid the observed 1 FPS
throttling on the development Linux display. Physics rates and contacts remain
unchanged. This is not a universal hardware-performance guarantee.
