#!/usr/bin/env bash
set -euo pipefail
WORLD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/robot-godot-worlds"
mkdir -p -- "$WORLD_STATE"
exec 9>"$WORLD_STATE/desktop.lock"
if ! flock -n 9; then
    if command -v xdotool >/dev/null; then
        xdotool search --onlyvisible --class 'Robot Godot Worlds' windowactivate >/dev/null 2>&1 || true
    fi
    exit 0
fi
if [ "$#" -eq 0 ]; then set -- --choose-scene; fi
export PYTHONUNBUFFERED=1
if "$WORLD_ROOT/run-workshop.sh" "$@" >"$WORLD_STATE/desktop.log" 2>&1; then
    exit 0
else
    WORLD_STATUS=$?
    if command -v notify-send >/dev/null; then
        notify-send --app-name='风口科学站' '场景启动失败' "运行记录：$WORLD_STATE/desktop.log" || true
    fi
    exit "$WORLD_STATUS"
fi
