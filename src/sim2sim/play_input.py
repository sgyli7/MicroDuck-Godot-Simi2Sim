"""Keyboard / HUD mapping matching microduck_rl infer_policy hold-to-move.

Hold W/↑ forward, S/↓ back, A/← yaw left, D/→ yaw right, Q/E strafe.
Release = idle. One-shot taps: pick / sit / kick / roll / reset / quit / push.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Godot keycode names (Input.is_physical_key_pressed) → hold bits
KEY_TO_HOLD: dict[str, str] = {
    "W": "fwd",
    "UP": "fwd",
    "S": "back",
    "DOWN": "back",
    "A": "left",
    "LEFT": "left",
    "D": "right",
    "RIGHT": "right",
    "Q": "strafe_l",
    "E": "strafe_r",
    "SPACE": "idle",
}

# One-shot taps (keyboard or HUD). Multiple aliases collapse to one action.
KEY_TO_TAP: dict[str, str] = {
    "G": "pick",
    "KEY_1": "pick",
    "Y": "sit",
    "KEY_2": "sit",
    "K": "kick_left",
    "KEY_3": "kick_left",
    "L": "kick_right",
    "KEY_4": "kick_right",
    "R": "roulade",
    "KEY_5": "roulade",
    "KEY_6": "switch_robot",
    "KEY_0": "reset",
    "BACKSPACE": "reset",
    "P": "push",
    "ESCAPE": "quit",
}

SKILL_TAPS = ("pick", "sit", "kick_left", "kick_right", "roulade")
LOCO_HOLDS = ("fwd", "back", "left", "right", "strafe_l", "strafe_r", "idle")

TIME_SCALE_MIN = 0.25
TIME_SCALE_MAX = 3.0
TIME_SCALE_DEFAULT = 1.0


def clamp_time_scale(value: float) -> float:
    return min(TIME_SCALE_MAX, max(TIME_SCALE_MIN, float(value)))


def wall_dt(sim_dt: float, time_scale: float) -> float:
    """Wall-clock seconds to wait for this control tick. Physics dt is unchanged."""
    return float(sim_dt) / clamp_time_scale(time_scale)


@dataclass(frozen=True)
class TwistLimits:
    vmax_x: float = 0.3
    vmin_x: float = -0.3
    vmax_y: float = 0.2
    vmin_y: float = -0.2
    vmax_ang: float = 1.5
    switch_threshold: float = 0.05


def keys_to_held(keys: set[str]) -> set[str]:
    held: set[str] = set()
    for k in keys:
        bit = KEY_TO_HOLD.get(k.upper() if len(k) == 1 else k)
        if bit:
            held.add(bit)
    return held


def relaunch_argv(argv: list[str], *, want_roller: bool, executable: str) -> list[str]:
    """Rebuild process argv to toggle walk ⇄ roller. Same trick as infer_policy os.execve."""
    rest = [a for a in argv[1:] if a != "--roller"]
    out = [executable, "-u", argv[0], *rest]
    if want_roller:
        out.append("--roller")
    return out


def keys_to_taps(keys: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        ku = k.upper() if len(k) == 1 else k
        action = KEY_TO_TAP.get(ku)
        if action and action not in seen:
            seen.add(action)
            out.append(action)
    return out


def held_twist(held: set[str], lim: TwistLimits | None = None) -> tuple[float, float, float]:
    """Map a set of held directions to a twist command. Empty set = idle."""
    lim = lim or TwistLimits()
    if "idle" in held:
        return 0.0, 0.0, 0.0
    vx = (lim.vmax_x if "fwd" in held else 0.0) + (lim.vmin_x if "back" in held else 0.0)
    vy = (lim.vmax_y if "strafe_l" in held else 0.0) + (lim.vmin_y if "strafe_r" in held else 0.0)
    yaw = (lim.vmax_ang if "left" in held else 0.0) + (-lim.vmax_ang if "right" in held else 0.0)
    return float(vx), float(vy), float(yaw)


@dataclass
class BrainOut:
    policy: str
    command: np.ndarray
    reset: bool = False
    quit: bool = False
    push: bool = False
    switch_robot: bool = False
    status: str = ""


@dataclass
class PlayBrain:
    """Policy / command state machine. No ONNX, no Godot — unit-testable."""

    has_walking: bool = True
    has_standing: bool = True
    has_sitstand: bool = True
    has_pick: bool = True
    has_kick_left: bool = True
    has_kick_right: bool = True
    has_roulade: bool = True
    lim: TwistLimits = field(default_factory=TwistLimits)
    pick_period: float = 4.0
    kick_duration: float = 3.0
    roulade_duration: float = 2.0

    policy: str = "standing"
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    sit: bool = False
    pick_phase: float = 0.0
    behavior_t: float = 0.0

    def __post_init__(self) -> None:
        if self.has_standing:
            self.policy = "standing"
        elif self.has_walking:
            self.policy = "walking"
        elif self.has_sitstand:
            self.policy = "sitstand"

    def _busy(self) -> bool:
        return self.policy in ("ground_pick", "kick_left", "kick_right", "roulade")

    def reset_motion(self) -> None:
        self.vel[:] = 0.0
        self.sit = False
        self.pick_phase = 0.0
        self.behavior_t = 0.0
        if self.has_standing:
            self.policy = "standing"
        elif self.has_walking:
            self.policy = "walking"
        elif self.has_sitstand:
            self.policy = "sitstand"

    def _set_loco(self, held: set[str]) -> None:
        if self._busy() or self.sit:
            return
        vx, vy, yaw = held_twist(held, self.lim)
        self.vel[:] = (vx, vy, yaw)
        mag = float(np.linalg.norm(self.vel))
        if self.has_walking and self.has_standing:
            self.policy = "standing" if mag <= self.lim.switch_threshold else "walking"
        elif self.has_walking:
            self.policy = "walking"
        elif self.has_standing:
            self.policy = "standing"

    def _tap(self, action: str) -> None:
        if action == "sit":
            if not self.has_sitstand:
                return
            if self._busy():
                return
            self.sit = not self.sit
            self.vel[:] = 0.0
            self.policy = "sitstand"
            return
        if action == "pick":
            if not self.has_pick or self._busy() or self.sit:
                return
            self.policy = "ground_pick"
            self.pick_phase = 0.0
            self.vel[:] = 0.0
            return
        if action in ("kick_left", "kick_right", "roulade"):
            has = {
                "kick_left": self.has_kick_left,
                "kick_right": self.has_kick_right,
                "roulade": self.has_roulade,
            }[action]
            if not has or self._busy() or self.sit:
                return
            self.policy = action
            self.behavior_t = self.roulade_duration if action == "roulade" else self.kick_duration
            self.vel[:] = 0.0

    def _advance(self, dt: float) -> None:
        if self.policy == "ground_pick":
            self.pick_phase += dt / self.pick_period
            if self.pick_phase >= 1.0:
                self.pick_phase = 0.0
                self.reset_motion()
            return
        if self.policy in ("kick_left", "kick_right", "roulade"):
            self.behavior_t -= dt
            if self.behavior_t <= 0.0:
                self.reset_motion()

    def command_13(self) -> np.ndarray:
        cmd = np.zeros(13, dtype=np.float32)
        if self.policy in ("kick_left", "kick_right", "roulade"):
            return cmd
        if self.policy == "ground_pick":
            cmd[0] = np.cos(2 * np.pi * self.pick_phase)
            cmd[1] = np.sin(2 * np.pi * self.pick_phase)
            return cmd
        if self.policy == "sitstand":
            cmd[0] = 1.0 if self.sit else 0.0
            return cmd
        if self.policy == "walking":
            cmd[0:3] = self.vel
        return cmd

    def tick(self, held: set[str], taps: list[str], dt: float) -> BrainOut:
        reset = "reset" in taps
        quit_ = "quit" in taps
        push = "push" in taps
        switch_robot = "switch_robot" in taps
        held_now = set(held)
        if "idle" in taps:
            held_now.add("idle")
        if reset:
            self.reset_motion()
        else:
            for action in taps:
                if action in SKILL_TAPS:
                    self._tap(action)
            self._set_loco(held_now)
            self._advance(dt)
        status = self.policy
        if self.policy == "sitstand":
            status = "sit" if self.sit else "sitstand-stand"
        elif self.policy == "walking":
            status = f"walk vx={self.vel[0]:+.2f} vy={self.vel[1]:+.2f} w={self.vel[2]:+.2f}"
        return BrainOut(
            policy=self.policy,
            command=self.command_13(),
            reset=reset,
            quit=quit_,
            push=push,
            switch_robot=switch_robot,
            status=status,
        )
