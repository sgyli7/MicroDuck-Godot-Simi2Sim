"""rsl_rl 5.0.1 VecEnv: one Godot/Jolt worker per robot, lockstep TCP."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from sim2sim.backends.godot_backend import GodotBackend
from sim2sim.coords import quat_rotate_inverse_wxyz
from sim2sim.fall import fallen
from sim2sim.godot_proc import stop_godot
from sim2sim.obs import DEFAULT_HOME, build_obs
from sim2sim.paths import apply_path_defaults, expand_cfg, load_robot_json, sim2sim_root
from sim2sim.train.commands import CommandConfig, CommandSampler
from sim2sim.train.reset_poses import HomePoseSampler
from sim2sim.train.rewards import RewardComputer, RewardConfig, RewardInputs, world_to_yaw_frame

FOOT_NAMES = ("ankle_left", "ankle_right")
ACTOR_DIM = 61
CRITIC_DIM = 70
NUM_ACTIONS = 14


def load_walk_cfg(path_or_dict: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    root = sim2sim_root()
    os.environ["SIM2SIM_ROOT"] = str(root)
    for key, ok in (
        ("MICRODUCK_POLICIES", lambda p: Path(p).is_dir()),
        ("MICRODUCK_RL", lambda p: (Path(p) / "src/mjlab_microduck").is_dir()),
    ):
        val = os.environ.get(key)
        if val and not ok(val):
            os.environ.pop(key, None)
    apply_path_defaults(root)
    if isinstance(path_or_dict, Mapping):
        cfg = expand_cfg(dict(path_or_dict))
    else:
        raw = yaml.safe_load(Path(path_or_dict).read_text())
        if not isinstance(raw, dict):
            raise TypeError(f"walk cfg is not a mapping: {path_or_dict}")
        cfg = expand_cfg(raw)
    robot_ref = cfg.get("robot")
    if isinstance(robot_ref, dict):
        cfg["robot_cfg"] = expand_cfg(robot_ref)
        return cfg
    robot_path = Path(robot_ref) if isinstance(robot_ref, str) else (root / "robots/microduck.json")
    if not robot_path.is_file():
        robot_path = root / "robots/microduck.json"
    cfg["robot"] = str(robot_path)
    cfg["robot_cfg"] = load_robot_json(robot_path)
    return cfg


def _as_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    d = str(device)
    if d.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def _foot_pack(state, ankle_z_nominal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feet = (state.extra or {}).get("feet") or []
    by = {str(f.get("name")): f for f in feet if isinstance(f, dict)}
    contact = np.zeros(2, dtype=np.float32)
    height = np.zeros(2, dtype=np.float32)
    xy_speed = np.zeros(2, dtype=np.float32)
    for i, name in enumerate(FOOT_NAMES):
        f = by.get(name)
        if f is None:
            continue
        contact[i] = 1.0 if f.get("contact") else 0.0
        pos = f.get("pos") or [0.0, 0.0, 0.0]
        height[i] = float(pos[2]) - float(ankle_z_nominal[i])
        lv = f.get("linvel") or [0.0, 0.0, 0.0]
        xy_speed[i] = float(np.hypot(float(lv[0]), float(lv[1])))
    return contact, height, xy_speed


def _state_finite(state) -> bool:
    chunks = (state.q, state.qd, state.base_pos, state.base_quat_wxyz, state.base_linvel, state.base_angvel_local)
    return all(np.isfinite(np.asarray(x)).all() for x in chunks)


class GodotVecEnv(VecEnv):
    """One-robot-per-process Godot workers matching rsl_rl ``VecEnv``."""

    num_actions: int = NUM_ACTIONS

    def __init__(
        self,
        cfg_path_or_dict: str | Path | Mapping[str, Any],
        num_envs: int | None = None,
        device: str = "cpu",
        seed: int | None = None,
        headless: bool | None = None,
    ) -> None:
        cfg = load_walk_cfg(cfg_path_or_dict)
        robot = cfg["robot_cfg"]
        self.cfg = cfg
        self.num_envs = int(num_envs if num_envs is not None else cfg.get("num_envs", 8))
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.device = _as_device(device)
        seed = int(cfg.get("seed", 0) if seed is None else seed)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.headless = cfg.get("headless", True) if headless is None else bool(headless)
        self.home = np.asarray(robot.get("home", DEFAULT_HOME), dtype=np.float32).reshape(-1)
        if self.home.size != NUM_ACTIONS:
            raise ValueError(f"home len {self.home.size} != {NUM_ACTIONS}")
        self.scale = float(robot.get("action_scale", 1.0))
        self.dt_phys = float(robot.get("timestep", 0.005))
        self.decimation = int(robot.get("decimation", 4))
        self.dt = self.dt_phys * self.decimation
        self.episode_s = float(cfg.get("episode_s", 20.0))
        self.max_episode_length = int(round(self.episode_s / self.dt))
        self.recv_timeout = float(cfg.get("recv_timeout_s", 10.0))
        self.stagger_s = float(cfg.get("spawn_stagger_s", 0.0))
        self.max_faults_per_step = int(cfg.get("max_faults_per_step", 8))
        self.faults_jsonl = Path(cfg.get("faults_jsonl") or (sim2sim_root() / "results/faults.jsonl"))
        self.faults_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.spec_path = Path(robot["godot_spec"])
        self.current_limit_a = float(robot.get("current_limit_a", 1.75))
        self.base_body = str(robot.get("base_body", "trunk_base"))
        self.tilt_deg = float((cfg.get("termination") or {}).get("tilt_deg", 70.0))
        self.min_z = float((cfg.get("termination") or {}).get("min_z", 0.055))
        reset_cfg = cfg.get("reset") or {}
        self.yaw_range = tuple(float(x) for x in reset_cfg.get("yaw_range", (-np.pi, np.pi)))
        self.joint_noise = float(reset_cfg.get("joint_noise_rad", 0.05))
        noise_cfg = cfg.get("obs_noise") or {}
        self.noise_enabled = bool(noise_cfg.get("enabled", True))
        self.noise_kind = str(noise_cfg.get("kind", "uniform"))
        self.noise_amp = {
            "gyro": float(noise_cfg.get("gyro", 0.03)),
            "grav": float(noise_cfg.get("grav", 0.01)),
            "q": float(noise_cfg.get("q", 0.001)),
            "qd": float(noise_cfg.get("qd", 0.25)),
        }
        push_cfg = cfg.get("pushes") or {}
        self.push_enabled = bool(push_cfg.get("enabled", True))
        self.push_interval = tuple(float(x) for x in push_cfg.get("interval_s", (3.0, 6.0)))
        self.push_xy_speed = float(push_cfg.get("xy_speed", 0.3))

        self.sampler = HomePoseSampler(robot)
        self.commands = CommandSampler(CommandConfig.from_dict(cfg.get("commands")), self.num_envs, self._rng)
        self.rew = RewardComputer(
            RewardConfig.from_dict(cfg.get("rewards")),
            self.num_envs,
            self.dt,
            self.home,
            self.sampler.joint_lo.astype(np.float32),
            self.sampler.joint_hi.astype(np.float32),
        )

        self._workers: list[GodotBackend | None] = [None] * self.num_envs
        self._states: list[Any] = [None] * self.num_envs
        self._last_action = np.zeros((self.num_envs, NUM_ACTIONS), dtype=np.float32)
        self._push_ttl = np.zeros(self.num_envs, dtype=np.float64)
        self.faults = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._obs = TensorDict(
            {
                "actor": torch.zeros(self.num_envs, ACTOR_DIM, device=self.device),
                "critic": torch.zeros(self.num_envs, CRITIC_DIM, device=self.device),
            },
            batch_size=[self.num_envs],
        )
        self.ankle_z_nominal = np.array(self.sampler.ankle_z_nominal, dtype=np.float32, copy=True)
        self._closed = False
        try:
            self._spawn_all()
            self._calibrate_ankle_z()
            self.reset_idx(list(range(self.num_envs)))
        except Exception:
            self.close()
            raise

    def _spawn_all(self) -> None:
        for i in range(self.num_envs):
            self._workers[i] = self._spawn_one()
            if self.stagger_s > 0 and i + 1 < self.num_envs:
                time.sleep(self.stagger_s)

    def _spawn_one(self) -> GodotBackend:
        return GodotBackend(
            self.spec_path,
            timestep=self.dt_phys,
            headless=self.headless,
            base_body=self.base_body,
            current_limit_a=self.current_limit_a,
            recv_timeout=self.recv_timeout,
        )

    def _calibrate_ankle_z(self) -> None:
        w0 = self._workers[0]
        assert w0 is not None
        st = self._reset_backend(w0, self.sampler.nominal())
        feet = (st.extra or {}).get("feet") or []
        by = {str(f.get("name")): f for f in feet if isinstance(f, dict)}
        zs = []
        for name in FOOT_NAMES:
            f = by.get(name)
            if f is None:
                zs.append(float(self.sampler.ankle_z_nominal[len(zs)]))
            else:
                zs.append(float((f.get("pos") or [0, 0, 0])[2]))
        self.ankle_z_nominal = np.asarray(zs, dtype=np.float32)

    def _reset_backend(self, worker: GodotBackend, poses: list[dict]):
        st = worker.reset(ctrl=self.home, bodies=poses)
        raw = (st.extra or {}).get("raw") or {}
        missing = raw.get("missing") or []
        if missing:
            raise RuntimeError(f"reset missing bodies: {missing}")
        applied = raw.get("applied") or []
        names = {p["name"] for p in poses}
        if names and applied and len(applied) < len(names):
            raise RuntimeError(f"reset applied {len(applied)}/{len(names)} bodies")
        return st

    def _reset_one(self, i: int) -> None:
        w = self._workers[i]
        assert w is not None
        poses, _q0, _ctrl = self.sampler.sample(
            self._rng, yaw_range=self.yaw_range, joint_noise_rad=self.joint_noise
        )
        self._states[i] = self._reset_backend(w, poses)
        self.commands.reset([i])
        self._last_action[i] = 0.0
        self.rew.zero_sums([i])
        self._push_ttl[i] = float(self._rng.uniform(*self.push_interval)) if self.push_enabled else 1e9
        self.episode_length_buf[i] = 0

    def reset_idx(self, env_ids: list[int] | np.ndarray | torch.Tensor) -> None:
        ids = [int(i) for i in np.asarray(env_ids, dtype=np.int64).reshape(-1)]
        for i in ids:
            self._reset_one(i)
        self._pack_obs(ids)

    def get_observations(self) -> TensorDict:
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self._closed:
            raise RuntimeError("GodotVecEnv is closed")
        n = self.num_envs
        act = np.asarray(actions.detach().to("cpu").numpy(), dtype=np.float32).reshape(n, NUM_ACTIONS)
        nan_act = ~np.isfinite(act).all(axis=1)
        act = np.where(nan_act[:, None], 0.0, act).astype(np.float32)
        ctrl = self.home[None, :] + act * self.scale

        self.episode_length_buf += 1
        self._maybe_push()

        send_faults: list[tuple[int, str]] = []
        for i, w in enumerate(self._workers):
            assert w is not None
            try:
                w.send_step(ctrl[i], n_substeps=self.decimation, report="lite")
            except Exception as e:
                send_faults.append((i, f"send:{type(e).__name__}:{e}"))

        recv_faults: list[tuple[int, str]] = []
        send_bad = {i for i, _ in send_faults}
        for i, w in enumerate(self._workers):
            assert w is not None
            if i in send_bad:
                continue
            try:
                self._states[i] = w.recv_step()
            except Exception as e:
                recv_faults.append((i, f"recv:{type(e).__name__}:{e}"))

        faults = send_faults + recv_faults
        if len(faults) > self.max_faults_per_step:
            raise RuntimeError(f"faults this step {len(faults)} > max_faults_per_step={self.max_faults_per_step}")

        fault_ids = {i for i, _ in faults}
        for i, reason in faults:
            self._handle_fault(i, reason)

        contact = np.zeros((n, 2), dtype=np.float32)
        height = np.zeros((n, 2), dtype=np.float32)
        xy_speed = np.zeros((n, 2), dtype=np.float32)
        q = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        gyro = np.zeros((n, 3), dtype=np.float32)
        grav = np.zeros((n, 3), dtype=np.float32)
        linvel = np.zeros((n, 3), dtype=np.float32)
        quat = np.zeros((n, 4), dtype=np.float64)
        pos = np.zeros((n, 3), dtype=np.float64)
        finite = np.ones(n, dtype=bool)
        for i in range(n):
            st = self._states[i]
            q[i] = np.asarray(st.q, dtype=np.float32).reshape(-1)[:NUM_ACTIONS]
            gyro[i] = np.asarray(st.base_angvel_local, dtype=np.float32).reshape(3)
            grav[i] = quat_rotate_inverse_wxyz(st.base_quat_wxyz, np.array([0.0, 0.0, -1.0])).astype(np.float32)
            linvel[i] = np.asarray(st.base_linvel, dtype=np.float32).reshape(3)
            quat[i] = np.asarray(st.base_quat_wxyz, dtype=np.float64).reshape(4)
            pos[i] = np.asarray(st.base_pos, dtype=np.float64).reshape(3)
            contact[i], height[i], xy_speed[i] = _foot_pack(st, self.ankle_z_nominal)
            finite[i] = _state_finite(st) and np.isfinite(contact[i]).all() and np.isfinite(height[i]).all()

        lin_yaw = world_to_yaw_frame(quat, linvel)
        cmd = self.commands.step(self.dt)
        total, terms = self.rew.compute(
            RewardInputs(
                q=q,
                gyro=gyro,
                grav=grav,
                base_linvel_yaw=lin_yaw,
                cmd13=cmd,
                action=act,
                last_action=self._last_action,
                contact=contact,
                foot_height=height,
                foot_xy_speed=xy_speed,
                joint_lo=self.sampler.joint_lo,
                joint_hi=self.sampler.joint_hi,
                home=self.home,
            )
        )

        fell = np.array(
            [fallen(quat[i], pos[i], tilt_deg=self.tilt_deg, min_z=self.min_z) for i in range(n)],
            dtype=bool,
        )
        nan_state = nan_act | ~finite | ~np.isfinite(total)
        time_out = (self.episode_length_buf.detach().cpu().numpy() >= self.max_episode_length)
        is_fault = np.array([i in fault_ids for i in range(n)], dtype=bool)

        # Faults already reset: zero their physics reward (no bogus terminal).
        total = np.where(is_fault | nan_state, 0.0, total).astype(np.float32)

        term = fell | nan_state
        done = term | time_out | is_fault
        to = (time_out | is_fault) & ~term

        reset_ids = [i for i in range(n) if done[i] and not is_fault[i]]
        for i in reset_ids:
            try:
                self._reset_one(i)
            except Exception as e:
                self._handle_fault(i, f"reset:{type(e).__name__}:{e}")
                is_fault[i] = True
                done[i] = True
                to[i] = True

        self._last_action = act.copy()
        self._last_action[done] = 0.0

        actor, critic = self._build_obs_arrays()
        actor = self._noise_actor(actor)
        self._obs = self._td(actor, critic)

        log: dict[str, float] = {f"Episode_Reward/{k}": float(np.mean(v)) for k, v in terms.items()}
        log["Episode_Termination/fell"] = float(np.mean(fell.astype(np.float32)))
        log["Episode_Termination/nan_state"] = float(np.mean(nan_state.astype(np.float32)))
        log["Episode_Termination/time_out"] = float(np.mean(time_out.astype(np.float32)))
        log["faults"] = float(self.faults)

        extras = {
            "time_outs": torch.as_tensor(to, dtype=torch.bool, device=self.device),
            "log": log,
        }
        rewards = torch.as_tensor(total, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        return self._obs, rewards, dones, extras

    def _build_obs_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.num_envs
        actor = np.zeros((n, ACTOR_DIM), dtype=np.float32)
        extra = np.zeros((n, 9), dtype=np.float32)
        for i in range(n):
            st = self._states[i]
            actor[i] = build_obs(st, self._last_action[i], self.commands.cmd[i], home=self.home)
            lin = world_to_yaw_frame(st.base_quat_wxyz, st.base_linvel).reshape(3)
            c, h, _ = _foot_pack(st, self.ankle_z_nominal)
            extra[i, 0:3] = lin
            extra[i, 3:5] = c
            extra[i, 5:7] = h
            extra[i, 7:9] = self.rew.air.air_time[i]
        critic = np.concatenate([actor, extra], axis=1).astype(np.float32)
        return actor, critic

    def _noise_actor(self, obs: np.ndarray) -> np.ndarray:
        if not self.noise_enabled:
            return obs
        o = np.array(obs, dtype=np.float32, copy=True)
        n = o.shape[0]

        def draw(amp: float, dim: int) -> np.ndarray:
            if amp <= 0:
                return np.zeros((n, dim), dtype=np.float32)
            if self.noise_kind == "gaussian":
                return self._rng.normal(0.0, amp, size=(n, dim)).astype(np.float32)
            return self._rng.uniform(-amp, amp, size=(n, dim)).astype(np.float32)

        o[:, 0:3] += draw(self.noise_amp["gyro"], 3)
        o[:, 3:6] += draw(self.noise_amp["grav"], 3)
        o[:, 6:20] += draw(self.noise_amp["q"], 14)
        o[:, 20:34] += draw(self.noise_amp["qd"], 14)
        return o

    def _pack_obs(self, env_ids: list[int] | None = None) -> None:
        actor, critic = self._build_obs_arrays()
        actor = self._noise_actor(actor)
        if env_ids is None:
            self._obs = self._td(actor, critic)
            return
        a = self._obs["actor"].detach().cpu().numpy()
        c = self._obs["critic"].detach().cpu().numpy()
        for i in env_ids:
            a[i] = actor[i]
            c[i] = critic[i]
        self._obs = self._td(a, c)

    def _td(self, actor: np.ndarray, critic: np.ndarray) -> TensorDict:
        return TensorDict(
            {
                "actor": torch.as_tensor(actor, dtype=torch.float32, device=self.device),
                "critic": torch.as_tensor(critic, dtype=torch.float32, device=self.device),
            },
            batch_size=[self.num_envs],
        )

    def _maybe_push(self) -> None:
        if not self.push_enabled:
            return
        self._push_ttl -= self.dt
        for i in range(self.num_envs):
            if self._push_ttl[i] > 0:
                continue
            w = self._workers[i]
            assert w is not None
            ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
            spd = float(self._rng.uniform(0.0, self.push_xy_speed))
            lin = np.array([spd * np.cos(ang), spd * np.sin(ang), 0.0], dtype=np.float64)
            try:
                w.nudge(lin)
            except Exception as e:
                self._handle_fault(i, f"nudge:{type(e).__name__}:{e}")
            self._push_ttl[i] = float(self._rng.uniform(*self.push_interval))

    def _handle_fault(self, i: int, reason: str) -> None:
        self.faults += 1
        rec = {
            "t": time.time(),
            "worker": int(i),
            "reason": str(reason)[:500],
            "faults": int(self.faults),
        }
        try:
            with self.faults_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        self._respawn(i)
        try:
            self._reset_one(i)
        except Exception as e:
            # Last-ditch respawn.
            self._respawn(i)
            self._reset_one(i)
            rec2 = dict(rec)
            rec2["reason"] = f"reset-after-respawn:{type(e).__name__}:{e}"
            try:
                with self.faults_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec2) + "\n")
            except OSError:
                pass

    def _respawn(self, i: int) -> None:
        w = self._workers[i]
        if w is not None:
            self._kill_worker(w)
        self._workers[i] = self._spawn_one()

    def _kill_worker(self, worker: GodotBackend) -> None:
        proc = getattr(worker, "_proc", None)
        client = getattr(worker, "_client", None)
        if client is not None:
            try:
                client.sock.close()
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        try:
            stop_godot(proc, None)
        except Exception:
            pass

    def bench_step_rate(self, n_steps: int) -> dict[str, float]:
        n_steps = int(n_steps)
        zeros = torch.zeros(self.num_envs, NUM_ACTIONS, dtype=torch.float32, device=self.device)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self.step(zeros)
        elapsed = max(time.perf_counter() - t0, 1e-9)
        total = n_steps * self.num_envs
        return {
            "num_envs": float(self.num_envs),
            "n_steps": float(n_steps),
            "elapsed_s": float(elapsed),
            "steps_per_s": float(total / elapsed),
            "latency_s": float(elapsed / n_steps),
            "faults": float(self.faults),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for i, w in enumerate(self._workers):
            if w is None:
                continue
            try:
                w.close()
            except Exception:
                self._kill_worker(w)
            self._workers[i] = None
        try:
            self.sampler.close()
        except Exception:
            pass

    def __enter__(self) -> GodotVecEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
