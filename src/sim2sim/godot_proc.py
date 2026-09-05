"""Launch a Godot scene as a lockstep physics server."""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from pathlib import Path

from sim2sim.paths import sim2sim_root
from sim2sim.protocol import JsonLineClient, wait_connect


GODOT_PROJECT = sim2sim_root() / "godot"


def godot_bin() -> str:
    return os.environ.get("GODOT") or os.path.expanduser("~/.local/bin/godot")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def spawn_godot(
    scene: str,
    *,
    port: int | None = None,
    headless: bool = True,
    extra_args: list[str] | None = None,
    cwd: Path | None = None,
) -> tuple[subprocess.Popen, int, JsonLineClient]:
    port = port or free_port()
    bin_ = godot_bin()
    cmd = [bin_]
    # Optional fallback when Vulkan device creation fails (e.g. flaky GPU).
    driver = os.environ.get("GODOT_RENDERING_DRIVER", "").strip()
    if driver:
        cmd += ["--rendering-driver", driver]
    elif not headless and os.environ.get("SIM2SIM_FORCE_GL"):
        # GB10/Spark: Vulkan device creation fails (-3); OpenGL works.
        cmd += ["--rendering-driver", "opengl3"]
    if headless:
        cmd.append("--headless")
        cmd += ["--fixed-fps", "200"]
    else:
        # 200 Hz main loop so 4 lockstep ticks are not bound to 60 Hz vsync.
        cmd += ["--disable-vsync", "--fixed-fps", "200"]
        # On a Wayland-capable desktop, Godot prefers Wayland even when an
        # X11 DISPLAY is set; the screenshot/window tooling here is X11.
        if os.environ.get("SIM2SIM_DISPLAY_DRIVER"):
            cmd += ["--display-driver", os.environ["SIM2SIM_DISPLAY_DRIVER"]]
    cmd += [
        "--path",
        str(cwd or GODOT_PROJECT),
        scene,
        "--",
        f"--port={port}",
    ]
    if extra_args:
        cmd += extra_args
    env = os.environ.copy()
    if env.get("DISPLAY") and not env.get("XAUTHORITY"):
        for cand in (
            Path.home() / ".Xauthority",
            Path(f"/run/user/{os.getuid()}/gdm/Xauthority"),
        ):
            if cand.is_file():
                env["XAUTHORITY"] = str(cand)
                break
    log_path = Path(tempfile.gettempdir()) / f"godot-sim2sim-{port}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    proc._sim2sim_log_path = log_path  # type: ignore[attr-defined]
    proc._sim2sim_log_file = log_file  # type: ignore[attr-defined]
    try:
        client = wait_connect("127.0.0.1", port, timeout=25.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass
        out = ""
        try:
            out = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Godot failed to accept TCP on {port}.\ncmd={cmd}\nlog:\n{out}") from None
    return proc, port, client


def stop_godot(proc: subprocess.Popen, client: JsonLineClient | None) -> str:
    if client is not None:
        try:
            client.call({"cmd": "close"})
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    log_file = getattr(proc, "_sim2sim_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
    log_path = getattr(proc, "_sim2sim_log_path", None)
    if log_path is not None:
        try:
            return Path(log_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    return ""
