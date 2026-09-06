"""Shared fallen / tilt predicate for play, compare, and training termination."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim2sim.coords import quat_rotate_inverse_wxyz

_DOWN = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass(frozen=True)
class FallCriteria:
    tilt_deg: float = 70.0
    min_z: float = 0.055


def _projected_gravity(base_quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_wxyz(base_quat_wxyz, _DOWN)


def tilt_deg(base_quat_wxyz: np.ndarray) -> float:
    """Tilt from vertical in degrees (0 = upright)."""
    grav = _projected_gravity(base_quat_wxyz)
    c = float(np.clip(-float(grav[2]), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def fallen(
    base_quat_wxyz: np.ndarray,
    base_pos: np.ndarray,
    *,
    tilt_deg: float = 70.0,
    min_z: float = 0.055,
) -> bool:
    """True if projected-gravity z > -cos(tilt) (lean past tilt_deg) or trunk z < min_z."""
    grav = _projected_gravity(base_quat_wxyz)
    z = float(np.asarray(base_pos, dtype=np.float64).reshape(-1)[2])
    return float(grav[2]) > -math.cos(math.radians(float(tilt_deg))) or z < float(min_z)
