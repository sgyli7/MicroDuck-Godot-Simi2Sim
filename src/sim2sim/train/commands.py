"""Piecewise-constant 13-D velocity commands (twist + zero head/body)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from sim2sim.obs import command_13


@dataclass
class CommandConfig:
    resample_s: tuple[float, float] = (3.0, 8.0)
    vx: tuple[float, float] = (-0.4, 0.4)
    vy: tuple[float, float] = (-0.3, 0.3)
    wz: tuple[float, float] = (-1.0, 1.0)
    standing_frac: float = 0.25
    turn_in_place_frac: float = 0.15
    turn_wz: tuple[float, float] = (0.4, 1.0)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CommandConfig:
        if not raw:
            return cls()
        kwargs: dict[str, Any] = {}
        if "resample_s" in raw:
            lo, hi = raw["resample_s"]
            kwargs["resample_s"] = (float(lo), float(hi))
        for key in ("vx", "vy", "wz", "turn_wz"):
            if key in raw:
                lo, hi = raw[key]
                kwargs[key] = (float(lo), float(hi))
        if "standing_frac" in raw:
            kwargs["standing_frac"] = float(raw["standing_frac"])
        if "turn_in_place_frac" in raw:
            kwargs["turn_in_place_frac"] = float(raw["turn_in_place_frac"])
        return cls(**kwargs)


class CommandSampler:
    """Per-env piecewise-constant twist. Deterministic given ``rng``."""

    def __init__(self, cfg: CommandConfig | Mapping[str, Any], num_envs: int, rng: np.random.Generator) -> None:
        self.cfg = cfg if isinstance(cfg, CommandConfig) else CommandConfig.from_dict(cfg)
        self.num_envs = int(num_envs)
        self.rng = rng
        self.cmd = np.zeros((self.num_envs, 13), dtype=np.float32)
        self.ttl = np.zeros(self.num_envs, dtype=np.float64)
        self.reset(np.arange(self.num_envs))

    def reset(self, env_ids: np.ndarray | list[int]) -> None:
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        for i in ids:
            self.cmd[i] = self._sample_one()
            self.ttl[i] = self._sample_ttl()

    def step(self, dt: float) -> np.ndarray:
        self.ttl -= float(dt)
        due = np.nonzero(self.ttl <= 0.0)[0]
        if due.size:
            self.reset(due)
        return self.cmd

    def _sample_ttl(self) -> float:
        lo, hi = self.cfg.resample_s
        return float(self.rng.uniform(lo, hi))

    def _sample_one(self) -> np.ndarray:
        cfg = self.cfg
        vx = float(self.rng.uniform(*cfg.vx))
        vy = float(self.rng.uniform(*cfg.vy))
        wz = float(self.rng.uniform(*cfg.wz))
        standing = float(self.rng.random()) < cfg.standing_frac
        turn = float(self.rng.random()) < cfg.turn_in_place_frac
        if standing:
            vx = vy = wz = 0.0
        if turn:
            vx = 0.0
            vy = 0.0
            mag = float(self.rng.uniform(*cfg.turn_wz))
            sign = -1.0 if float(self.rng.random()) < 0.5 else 1.0
            wz = sign * mag
        return command_13(np.array([vx, vy, wz], dtype=np.float32))
