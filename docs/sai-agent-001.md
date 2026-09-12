# Sai_Agent_001 profile

```sh
uv sync --extra sai
uv run --extra sai sim2sim-play --robot Sai_Agent_001
```

The optional profile pins the Sai source/model/policies to commit
`8ea8348d95193c2bad91874816a76e5527238e30`. It supplies its own 26-body,
23-hinge/two-slider Jolt scene, 82D observation and 16D mixed position/velocity
controller. MicroDuck remains the default and retains its existing scene/policy
contracts. The launcher starts the policy service automatically; no separate
training server is needed.

W/S forward/reverse, A/D yaw, hold Shift to crouch and release to stand, Esc quit.
Godot 4.7.2 must be installed and on PATH (`--godot-bin` selects another location).

```sh
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --stairs .02
uv run --extra sai sim2sim-play --robot Sai_Agent_001 --headless --case W --output results/sai-W.json
```

The pinned package passed eight flat keyboard cases and four-riser 20/40 mm
ascent/descent in Godot/Jolt. It uses simulated ground raycasts; camera/VLA terrain
reconstruction and 60 mm/held-out stairs are not validated. Other MicroDuck
`--scene` presets do not automatically convert into the Sai profile; this first
adapter launches the package's own scene. The SO101 and cargo joints remain real
articulations, not fixed display parts.
