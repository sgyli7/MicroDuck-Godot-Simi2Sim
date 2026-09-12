"""One persistent workshop window: native MicroDuck and release Sai controllers."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from sim2sim.paths import sim2sim_root

ROOT = sim2sim_root()


def prepare(runtime: Path, godot: str) -> Path:
    from sai_agent.paths import resource_root
    from sai_agent.cli import prepare_godot
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "godot", runtime, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".godot", "vendor", "*.import", "scenery", "textures", "materials",
                                                   "grass", "levels", "objects", "forest_generated", "addons"))
    # Bound engine worker allocation independently of host logical CPU count.
    # Scheduling only: no time scale, solver, policy or motor parameter changes.
    project = runtime / "project.godot"
    text = project.read_text().replace('config/name="Microduck Sim2Sim"',
        'config/name="Robot Godot Workshop"\nconfig/icon="res://atelier/icon.svg"', 1)
    project.write_text(text + "\n[threading]\nworker_pool/max_threads=2\n")
    bundle = resource_root()
    core = runtime.parent / "sai-release"
    prepare_godot(bundle, core)
    (runtime / "sai_release").mkdir(exist_ok=True)
    # Release code assumes an origin-centred scene in this diagnostic only.
    source = (core / "main.gd").read_text()
    old = "base.global_transform.affine_inverse()*robot.item.position"
    assert source.count(old) == 1
    (runtime / "sai_release/main.gd").write_text(source.replace(old,
        "base.global_transform.affine_inverse()*robot.item.global_position"))
    for name in ("robot.gd", "item_observation.gd"):
        shutil.copy2(core / name, runtime / name)
    shutil.copytree(core / "sai_agent", runtime / "sai_agent", dirs_exist_ok=True)
    # Capture engine-resolved defaults, not guessed Jolt equivalents.
    profiles = {}
    for name, project in (("microduck", runtime), ("sai", core)):
        shutil.copy2(ROOT / "godot/hub/dump_physics.gd", project / "dump_physics.gd")
        output = runtime.parent / f"{name}-physics.json"
        subprocess.run([godot, "--headless", "--path", str(project), "--script", "res://dump_physics.gd"],
                       env=dict(os.environ, HUB_PHYSICS_DUMP=str(output)), check=True)
        profiles[name] = json.loads(output.read_text())
    (runtime / "hub/physics_profiles.json").write_text(json.dumps(profiles, indent=2))
    if not (runtime / "runtime_assets/deployment.json").is_file():
        raise SystemExit("Prepare MicroDuck's native model bundle first; see docs/workshop-hub.md.")
    subprocess.run([godot, "--headless", "--editor", "--path", str(runtime), "--import", "--quit"], check=True)
    return bundle


def serve(listener, stop, bundle, trace):
    from sai_agent.godot_controller import GodotController
    from sai_agent.cargo_godot import CargoGodotController
    while not stop.is_set():
        try:
            client, _ = listener.accept()
        except socket.timeout:
            continue
        with client:
            client.settimeout(.25)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            buffer = b""
            controller = None
            config = None
            while not stop.is_set():
                try:
                    packet = client.recv(65536)
                except socket.timeout:
                    continue
                if not packet:
                    break
                buffer += packet
                if len(buffer) > 1048576:
                    raise ValueError("Oversized Sai state")
                finished = False
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    state = json.loads(line)
                    if state.get("finish"):
                        finished = True
                        break
                    if controller is None:
                        config = state["hub_config"]
                        skill = config.get("skill", "")
                        profile = bundle / "policies/experimental" / f"{skill}.json" if skill else None
                        controller = CargoGodotController() if config["task"] == "cargo" else GodotController(bundle, stair_profile=profile)
                    response = controller.command(state)
                    client.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
                    if trace:
                        trace.write(json.dumps(dict(config=config, state=state, command=response), separators=(",", ":")) + "\n")
                if finished:
                    break


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("microduck", "roller", "sai"), default="microduck")
    parser.add_argument("--task", default="drive", choices=("drive", "cargo18", "cargo25", "up20", "down20", "up40", "down40", "up60", "down60"))
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT") or shutil.which("godot"))
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "results/workshop-hub/runtime")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--plan", type=Path, help="Timed integration test / capture plan")
    parser.add_argument("--output", type=Path, default=ROOT / "results/workshop-hub/play")
    parser.add_argument("--record", action="store_true", help="Timestamped native game frames and policy trace")
    args = parser.parse_args(argv)
    if not args.godot_bin:
        parser.error("Godot 4.7.2 is required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(SIM2SIM_VISUAL_STYLE="legacy", MD_WORKSHOP_COLLISIONS="1", MD_MODE="hub")
    bundle = prepare(args.runtime_dir.resolve(), args.godot_bin)
    options = dict(robot=args.robot, task=args.task, output=str(args.output), record=args.record,
                   plan=json.loads(args.plan.read_text()) if args.plan else {})
    (args.runtime_dir / "hub/options.json").write_text(json.dumps(options))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(.25)
        stop = threading.Event()
        trace = (args.output / "sai-trace.jsonl").open("w") if args.record or args.plan else None
        command = [args.godot_bin, "--path", str(args.runtime_dir.resolve()), "res://hub/main.tscn", "--disable-vsync", "--max-fps", "30"]
        if args.headless:
            command += ["--headless"]
        command += ["--", f"--port={listener.getsockname()[1]}"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            service = pool.submit(serve, listener, stop, bundle, trace)
            child = subprocess.Popen(command)
            started = time.monotonic()
            try:
                while child.poll() is None:
                    stop.wait(.1)
                    if service.done():
                        service.result()
                    if args.plan and time.monotonic() - started > max(180, float(options["plan"].get("seconds",72))*6+60):
                        raise TimeoutError("Workshop plan exceeded its wall-time budget")
                return child.returncode
            finally:
                stop.set()
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                service.result(timeout=5)
                if trace:
                    trace.close()


if __name__ == "__main__":
    raise SystemExit(main())
