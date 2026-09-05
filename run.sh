#!/usr/bin/env bash
# One-shot Sim2Sim gate: convert → import → spikes → calib → dual rollout → compare.
# Kick is intentionally NOT in this HARD FAIL path (Godot kick is known-fail today).
# Independent kick gate: ./run_kick_gate.sh   or   uv run sim2sim-kick-gate
# Soft mode exits 0 with KNOWN_FAIL; does not poison SIM2SIM_RUN / LOCAL_GATE walk green.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export GODOT="${GODOT:-$HOME/.local/bin/godot}"
export SIM2SIM_ROOT="$ROOT"
export MICRODUCK_RL="${MICRODUCK_RL:-$HOME/Projects/microduck_rl}"
export MICRODUCK_POLICIES="${MICRODUCK_POLICIES:-$HOME/Projects/MicroDuck/policies}"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "== godot =="
"$GODOT" --headless --version

echo "== uv sync =="
uv sync

echo "== convert =="
uv run mjcf2godot \
  --mjcf "$MICRODUCK_RL/src/mjlab_microduck/robot/microduck/scene.xml" \
  --out "$ROOT/godot/generated/microduck"

echo "== godot import =="
rm -rf "$ROOT/godot/.godot"
"$GODOT" --headless --path "$ROOT/godot" --import --quit-after 1

if [[ -n "${DISPLAY:-}" ]]; then
  echo "== windowed smoke (quit-after) =="
  timeout 12 "$GODOT" --path "$ROOT/godot" --quit-after 60 || echo "windowed smoke: timeout/non-zero (recorded, continuing)"
fi

echo "== spikes =="
uv run sim2sim-spikes | tee "$RESULTS/spikes.log"

echo "== obs parity =="
uv run python -m sim2sim.obs_parity | tee "$RESULTS/obs_parity.log"

echo "== calib mujoco =="
uv run sim2sim-calib --backend mujoco --out "$RESULTS/calib_mujoco.json"

echo "== calib godot =="
uv run sim2sim-calib --backend godot --out "$RESULTS/calib_godot.json"

ALPHA="$MICRODUCK_POLICIES/alpha_walking.onnx"
LOCAL="$MICRODUCK_POLICIES/local-ppo/local_velocity_walk_run_idle.onnx"

echo "== rollout alpha_walking mujoco =="
uv run sim2sim-runner --backend mujoco --onnx "$ALPHA" --out "$RESULTS/alpha_mujoco.npz"
echo "== rollout alpha_walking godot =="
uv run sim2sim-runner --backend godot --onnx "$ALPHA" --out "$RESULTS/alpha_godot.npz"
echo "== compare alpha_walking =="
ALPHA_RC=0
uv run sim2sim-compare --mujoco "$RESULTS/alpha_mujoco.npz" --godot "$RESULTS/alpha_godot.npz" \
  --out "$RESULTS/alpha_compare" --title "alpha_walking Sim2Sim" || ALPHA_RC=$?

echo "== rollout local_ppo mujoco =="
uv run sim2sim-runner --backend mujoco --onnx "$LOCAL" --out "$RESULTS/ppo_mujoco.npz"
echo "== rollout local_ppo godot =="
uv run sim2sim-runner --backend godot --onnx "$LOCAL" --out "$RESULTS/ppo_godot.npz"
echo "== compare local_ppo =="
PPO_RC=0
uv run sim2sim-compare --mujoco "$RESULTS/ppo_mujoco.npz" --godot "$RESULTS/ppo_godot.npz" \
  --out "$RESULTS/ppo_compare" --title "local_ppo Sim2Sim" || PPO_RC=$?

echo "== gate summary =="
python3 - <<'PY'
from pathlib import Path
root = Path("results")
print((root/"alpha_compare/report.md").read_text() if (root/"alpha_compare/report.md").exists() else "no alpha report")
print((root/"ppo_compare/report.md").read_text() if (root/"ppo_compare/report.md").exists() else "no ppo report")
PY

echo "COMPARE_ALPHA_RC=$ALPHA_RC COMPARE_PPO_RC=$PPO_RC"
if [[ "$ALPHA_RC" != "0" || "$PPO_RC" != "0" ]]; then
  echo "SIM2SIM_RUN: GATE HARD FAIL  results=$RESULTS"
  exit 1
fi
echo "SIM2SIM_RUN: done  results=$RESULTS"
