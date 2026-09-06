"""Vectorized walk rewards (numpy). Port of mjlab/microduck_rl velocity terms.

Whole-body ``angular_momentum`` is skipped: Godot lite step has no angmom sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np

# Actuator order (robot_walk.xml): left 0-4, neck/head 5-8, right 9-13.
LEG_JOINT_IDX = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13], dtype=np.int64)
HEAD_JOINT_IDX = np.array([5, 6, 7, 8], dtype=np.int64)

# variable_posture stds from microduck_velocity_env_cfg (legs only).
_STD_STANDING_LEGS = np.array(
    [0.1, 0.05, 0.15, 0.15, 0.1, 0.1, 0.05, 0.15, 0.15, 0.1], dtype=np.float32
)
_STD_WALKING_LEGS = np.array(
    [0.3, 0.05, 0.4, 0.4, 0.25, 0.3, 0.05, 0.4, 0.4, 0.25], dtype=np.float32
)

TERM_NAMES: tuple[str, ...] = (
    "track_lin_vel",
    "track_ang_vel",
    "upright",
    "air_time",
    "pose_legs",
    "foot_clearance",
    "foot_swing_height",
    "action_rate_l2",
    "foot_slip",
    "body_ang_vel",
    "head_pose_tracking",
    "head_pose_bias",
    "dof_pos_limits",
)


@dataclass
class RewardConfig:
    """Weights and kernel params. Weight 0 disables a term."""

    track_lin_vel: float = 2.0
    track_lin_vel_std2: float = 0.1
    track_ang_vel: float = 2.0
    track_ang_vel_std2: float = 0.5
    upright: float = 2.0
    upright_std2: float = 0.05
    air_time: float = 3.0
    air_time_min: float = 0.125
    air_time_max: float = 0.3
    air_time_cmd_threshold: float = 0.01
    pose_legs: float = 1.0
    pose_walking_threshold: float = 0.01
    foot_clearance: float = -2.0
    foot_clearance_target: float = 0.02
    foot_swing_height: float = -0.25
    foot_swing_target: float = 0.02
    action_rate_l2: float = -1.0
    foot_slip: float = -0.1
    body_ang_vel: float = -0.05
    head_pose_tracking: float = 2.0
    head_pose_tracking_std: float = 0.5
    head_pose_bias: float = -3.0
    head_pose_bias_tau_s: float = 1.0
    dof_pos_limits: float = -1.0
    scale_by_dt: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> RewardConfig:
        if not raw:
            return cls()
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: raw[k] for k in raw if k in allowed}
        return cls(**kwargs)


def track_lin_vel(
    v_cmd_xy: np.ndarray, v_body_xy: np.ndarray, std2: float = 0.1
) -> np.ndarray:
    """exp(-|v_cmd_xy - v_body_xy|^2 / std2)."""
    err = np.asarray(v_cmd_xy, dtype=np.float32) - np.asarray(v_body_xy, dtype=np.float32)
    sq = np.sum(err * err, axis=-1)
    return np.exp(-sq / float(std2)).astype(np.float32)


def track_ang_vel(wz_cmd: np.ndarray, gyro_z: np.ndarray, std2: float = 0.5) -> np.ndarray:
    """exp(-(wz_cmd - gyro_z)^2 / std2)."""
    d = np.asarray(wz_cmd, dtype=np.float32).reshape(-1) - np.asarray(gyro_z, dtype=np.float32).reshape(
        -1
    )
    return np.exp(-(d * d) / float(std2)).astype(np.float32)


def upright(grav: np.ndarray, std2: float = 0.05) -> np.ndarray:
    """mjlab upright: exp(-|grav_xy|^2 / std2). Upright projected gravity is (0,0,-1)."""
    g = np.asarray(grav, dtype=np.float32)
    xy = g[..., 0] * g[..., 0] + g[..., 1] * g[..., 1]
    return np.exp(-xy / float(std2)).astype(np.float32)


def yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def world_to_yaw_frame(quat: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Rotate world linvel into the yaw (heading) frame. vz is unchanged."""
    yaw = yaw_from_quat_wxyz(quat)
    c = np.cos(yaw)
    s = np.sin(yaw)
    v = np.asarray(v_world, dtype=np.float64).reshape(-1, 3)
    bx = c * v[:, 0] + s * v[:, 1]
    by = -s * v[:, 0] + c * v[:, 1]
    return np.stack([bx, by, v[:, 2]], axis=-1).astype(np.float32)


def cmd_speed(cmd13: np.ndarray) -> np.ndarray:
    c = np.asarray(cmd13, dtype=np.float32).reshape(-1, 13)
    lin = np.sqrt(c[:, 0] * c[:, 0] + c[:, 1] * c[:, 1])
    return (lin + np.abs(c[:, 2])).astype(np.float32)


def pose_legs(
    q: np.ndarray,
    home: np.ndarray,
    speed: np.ndarray,
    walking_threshold: float = 0.01,
    std_standing: np.ndarray | None = None,
    std_walking: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian on leg joints vs HOME; tighter std when |cmd| < walking_threshold."""
    q = np.asarray(q, dtype=np.float32).reshape(-1, 14)
    home = np.asarray(home, dtype=np.float32).reshape(14)
    speed = np.asarray(speed, dtype=np.float32).reshape(-1)
    std_s = _STD_STANDING_LEGS if std_standing is None else np.asarray(std_standing, dtype=np.float32)
    std_w = _STD_WALKING_LEGS if std_walking is None else np.asarray(std_walking, dtype=np.float32)
    stand = (speed < float(walking_threshold)).astype(np.float32)[:, None]
    std = stand * std_s[None, :] + (1.0 - stand) * std_w[None, :]
    err = q[:, LEG_JOINT_IDX] - home[LEG_JOINT_IDX]
    z = np.mean((err * err) / (std * std), axis=1)
    return np.exp(-z).astype(np.float32)


def head_pose_tracking(
    q: np.ndarray,
    home: np.ndarray,
    cmd13: np.ndarray,
    std: float = 0.5,
) -> np.ndarray:
    """Mean over 4 head joints of exp(-(err/std)^2). cmd head slots are deltas from HOME."""
    q = np.asarray(q, dtype=np.float32).reshape(-1, 14)
    home = np.asarray(home, dtype=np.float32).reshape(14)
    cmd = np.asarray(cmd13, dtype=np.float32).reshape(-1, 13)
    actual = q[:, HEAD_JOINT_IDX] - home[HEAD_JOINT_IDX]
    err = actual - cmd[:, 3:7]
    per = np.exp(-((err / float(std)) ** 2))
    return np.mean(per, axis=1).astype(np.float32)


def action_rate_l2(action: np.ndarray, last_action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32)
    p = np.asarray(last_action, dtype=np.float32)
    d = a - p
    return np.sum(d * d, axis=-1).astype(np.float32)


def dof_pos_limits(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    below = np.maximum(lo - q, 0.0)
    above = np.maximum(q - hi, 0.0)
    return np.sum(below + above, axis=-1).astype(np.float32)


def body_ang_vel(gyro_xy: np.ndarray) -> np.ndarray:
    g = np.asarray(gyro_xy, dtype=np.float32).reshape(-1, 2)
    return np.sum(g * g, axis=-1).astype(np.float32)


def foot_slip(
    contact: np.ndarray,
    foot_xy_speed: np.ndarray,
    speed: np.ndarray,
    cmd_threshold: float = 0.01,
) -> np.ndarray:
    c = np.asarray(contact, dtype=np.float32).reshape(-1, 2)
    v = np.asarray(foot_xy_speed, dtype=np.float32).reshape(-1, 2)
    active = (np.asarray(speed, dtype=np.float32).reshape(-1) > float(cmd_threshold)).astype(np.float32)
    return (np.sum((v * v) * c, axis=1) * active).astype(np.float32)


def foot_clearance(
    height: np.ndarray,
    foot_xy_speed: np.ndarray,
    speed: np.ndarray,
    target: float = 0.02,
    cmd_threshold: float = 0.01,
) -> np.ndarray:
    h = np.asarray(height, dtype=np.float32).reshape(-1, 2)
    v = np.asarray(foot_xy_speed, dtype=np.float32).reshape(-1, 2)
    active = (np.asarray(speed, dtype=np.float32).reshape(-1) > float(cmd_threshold)).astype(np.float32)
    cost = np.sum(np.abs(h - float(target)) * v, axis=1)
    return (cost * active).astype(np.float32)


def air_time_reward(
    first_contact: np.ndarray,
    last_air_time: np.ndarray,
    cmd_speed_n: np.ndarray,
    tmin: float = 0.125,
    tmax: float = 0.3,
    cmd_threshold: float = 0.01,
) -> np.ndarray:
    """On touchdown, +1 per foot whose completed air time is in (tmin, tmax), if |cmd|>thr."""
    fc = np.asarray(first_contact, dtype=bool)
    air = np.asarray(last_air_time, dtype=np.float32)
    in_win = (air > float(tmin)) & (air < float(tmax))
    r = np.sum(fc & in_win, axis=-1).astype(np.float32)
    spd = np.asarray(cmd_speed_n, dtype=np.float32).reshape(-1)
    r *= (spd > float(cmd_threshold)).astype(np.float32)
    return r


class AirTimeTracker:
    """Per-foot air/contact timers at control rate. Starts in contact (home stand)."""

    def __init__(self, num_envs: int, n_feet: int = 2) -> None:
        self.num_envs = int(num_envs)
        self.n_feet = int(n_feet)
        self.air_time = np.zeros((self.num_envs, self.n_feet), dtype=np.float32)
        self.contact_time = np.zeros((self.num_envs, self.n_feet), dtype=np.float32)
        self.in_contact = np.ones((self.num_envs, self.n_feet), dtype=bool)

    def reset(self, env_ids: np.ndarray | list[int] | slice) -> None:
        self.air_time[env_ids] = 0.0
        self.contact_time[env_ids] = 0.0
        self.in_contact[env_ids] = True

    def step(self, contact: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance timers. Returns (first_contact, last_air_time) for this step."""
        now = np.asarray(contact, dtype=bool).reshape(self.num_envs, self.n_feet)
        first = now & ~self.in_contact
        last_air = np.where(first, self.air_time, 0.0).astype(np.float32)
        dt = float(dt)
        air = self.air_time + dt
        ct = self.contact_time + dt
        self.air_time = np.where(now, 0.0, air).astype(np.float32)
        self.contact_time = np.where(now, ct, 0.0).astype(np.float32)
        self.in_contact = now
        return first, last_air


class HeadPoseBias:
    """1 s EMA of head-joint error; raw term is mean |EMA| (penalty, weight negative)."""

    def __init__(self, num_envs: int, n_joints: int = 4, tau_s: float = 1.0, dt: float = 0.02) -> None:
        self.ema = np.zeros((int(num_envs), int(n_joints)), dtype=np.float32)
        self.alpha = min(1.0, float(dt) / max(float(tau_s), 1e-6))

    def reset(self, env_ids: np.ndarray | list[int] | slice) -> None:
        self.ema[env_ids] = 0.0

    def step(self, err: np.ndarray) -> np.ndarray:
        e = np.asarray(err, dtype=np.float32)
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * e
        return np.mean(np.abs(self.ema), axis=-1).astype(np.float32)


class SwingPeakTracker:
    def __init__(self, num_envs: int, n_feet: int = 2) -> None:
        self.peak = np.zeros((int(num_envs), int(n_feet)), dtype=np.float32)

    def reset(self, env_ids: np.ndarray | list[int] | slice) -> None:
        self.peak[env_ids] = 0.0

    def step(
        self,
        in_air: np.ndarray,
        height: np.ndarray,
        first_contact: np.ndarray,
        target: float,
        speed: np.ndarray,
        cmd_threshold: float,
    ) -> np.ndarray:
        air = np.asarray(in_air, dtype=bool)
        h = np.asarray(height, dtype=np.float32)
        self.peak = np.where(air, np.maximum(self.peak, h), self.peak).astype(np.float32)
        fc = np.asarray(first_contact, dtype=bool)
        tgt = max(float(target), 1e-8)
        err = self.peak / tgt - 1.0
        cost = np.sum((err * err) * fc.astype(np.float32), axis=1)
        active = (np.asarray(speed, dtype=np.float32).reshape(-1) > float(cmd_threshold)).astype(
            np.float32
        )
        self.peak = np.where(fc, 0.0, self.peak).astype(np.float32)
        return (cost * active).astype(np.float32)


@dataclass
class RewardInputs:
    q: np.ndarray
    gyro: np.ndarray
    grav: np.ndarray
    base_linvel_yaw: np.ndarray
    cmd13: np.ndarray
    action: np.ndarray
    last_action: np.ndarray
    contact: np.ndarray
    foot_height: np.ndarray
    foot_xy_speed: np.ndarray
    joint_lo: np.ndarray
    joint_hi: np.ndarray
    home: np.ndarray


class RewardComputer:
    """Stateful batch rewards + episode sums for rsl_rl extras['log']."""

    def __init__(
        self,
        cfg: RewardConfig,
        num_envs: int,
        dt: float,
        home: np.ndarray,
        joint_lo: np.ndarray,
        joint_hi: np.ndarray,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.home = np.asarray(home, dtype=np.float32).reshape(-1)
        self.joint_lo = np.asarray(joint_lo, dtype=np.float32).reshape(-1)
        self.joint_hi = np.asarray(joint_hi, dtype=np.float32).reshape(-1)
        self.air = AirTimeTracker(self.num_envs)
        self.bias = HeadPoseBias(
            self.num_envs, tau_s=cfg.head_pose_bias_tau_s, dt=self.dt
        )
        self.swing = SwingPeakTracker(self.num_envs)
        self.episode_sums = {n: np.zeros(self.num_envs, dtype=np.float32) for n in TERM_NAMES}

    def reset(self, env_ids: np.ndarray | list[int]) -> dict[str, float]:
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        extras: dict[str, float] = {}
        if ids.size:
            for n in TERM_NAMES:
                extras[n] = float(np.mean(self.episode_sums[n][ids]))
                self.episode_sums[n][ids] = 0.0
            self.air.reset(ids)
            self.bias.reset(ids)
            self.swing.reset(ids)
        return extras

    def zero_sums(self, env_ids: np.ndarray | list[int]) -> None:
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if ids.size:
            for n in TERM_NAMES:
                self.episode_sums[n][ids] = 0.0
            self.air.reset(ids)
            self.bias.reset(ids)
            self.swing.reset(ids)

    def compute(self, inp: RewardInputs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg = self.cfg
        n = self.num_envs
        speed = cmd_speed(inp.cmd13)
        first, last_air = self.air.step(inp.contact, self.dt)
        in_air = ~np.asarray(inp.contact, dtype=bool)
        q = np.asarray(inp.q, dtype=np.float32).reshape(n, -1)
        home = self.home
        cmd = np.asarray(inp.cmd13, dtype=np.float32).reshape(n, 13)
        head_err = (q[:, HEAD_JOINT_IDX] - home[HEAD_JOINT_IDX]) - cmd[:, 3:7]

        raw: dict[str, np.ndarray] = {
            "track_lin_vel": track_lin_vel(
                cmd[:, :2], inp.base_linvel_yaw[:, :2], cfg.track_lin_vel_std2
            ),
            "track_ang_vel": track_ang_vel(cmd[:, 2], inp.gyro[:, 2], cfg.track_ang_vel_std2),
            "upright": upright(inp.grav, cfg.upright_std2),
            "air_time": air_time_reward(
                first,
                last_air,
                speed,
                tmin=cfg.air_time_min,
                tmax=cfg.air_time_max,
                cmd_threshold=cfg.air_time_cmd_threshold,
            ),
            "pose_legs": pose_legs(q, home, speed, cfg.pose_walking_threshold),
            "foot_clearance": foot_clearance(
                inp.foot_height,
                inp.foot_xy_speed,
                speed,
                target=cfg.foot_clearance_target,
                cmd_threshold=cfg.air_time_cmd_threshold,
            ),
            "foot_swing_height": self.swing.step(
                in_air,
                inp.foot_height,
                first,
                cfg.foot_swing_target,
                speed,
                cfg.air_time_cmd_threshold,
            ),
            "action_rate_l2": action_rate_l2(inp.action, inp.last_action),
            "foot_slip": foot_slip(
                inp.contact, inp.foot_xy_speed, speed, cfg.air_time_cmd_threshold
            ),
            "body_ang_vel": body_ang_vel(inp.gyro[:, :2]),
            "head_pose_tracking": head_pose_tracking(
                q, home, cmd, std=cfg.head_pose_tracking_std
            ),
            "head_pose_bias": self.bias.step(head_err),
            "dof_pos_limits": dof_pos_limits(q, self.joint_lo, self.joint_hi),
        }

        scale = self.dt if cfg.scale_by_dt else 1.0
        total = np.zeros(n, dtype=np.float32)
        weighted: dict[str, np.ndarray] = {}
        for name in TERM_NAMES:
            w = float(getattr(cfg, name))
            if w == 0.0:
                term = np.zeros(n, dtype=np.float32)
            else:
                term = (raw[name].astype(np.float32) * w * scale).astype(np.float32)
                term = np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
            weighted[name] = term
            total += term
            self.episode_sums[name] += term
        return total.astype(np.float32), weighted
