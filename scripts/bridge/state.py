"""Thread-safe command state for the LLM bridge.

The HTTP thread submits commands; the sim thread drains and applies them.
Caps match the trained command ranges in microduck_velocity_env_cfg.py.
"""

import math
import threading
from collections import deque
from dataclasses import dataclass

VX_MAX = 0.4           # m/s
VY_MAX = 0.3           # m/s
WZ_MAX = 1.0           # rad/s
HEAD_PITCH_MAX = 1.1   # rad, trained neck/head_pitch cap
HEAD_YAW_MAX = 1.4     # rad, trained head_yaw cap

WALK_DEFAULT_S = 3.0
WALK_MAX_S = 10.0

GESTURES = ("nod", "shake")


@dataclass(frozen=True)
class WalkCmd:
    vx: float
    vy: float
    wz: float
    seconds: float


@dataclass(frozen=True)
class LookCmd:
    pitch: float
    yaw: float


@dataclass(frozen=True)
class GestureCmd:
    name: str


@dataclass(frozen=True)
class StopCmd:
    pass


def _clamp(value: float, limit: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be a finite number")
    return max(-limit, min(limit, value))


class BridgeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending: deque = deque()
        self._status: dict = {"ready": False}

    def submit_walk(self, vx, vy, wz, seconds) -> dict:
        cvx, cvy, cwz = _clamp(vx, VX_MAX), _clamp(vy, VY_MAX), _clamp(wz, WZ_MAX)
        orig_seconds = seconds
        if seconds is None:
            seconds = WALK_DEFAULT_S
        seconds = max(0.0, min(float(seconds), WALK_MAX_S))
        speeds_clamped = (cvx, cvy, cwz) != (float(vx), float(vy), float(wz))
        seconds_clamped = (orig_seconds is not None) and (seconds != float(orig_seconds))
        clamped = speeds_clamped or seconds_clamped
        with self._lock:
            self._pending.append(WalkCmd(cvx, cvy, cwz, seconds))
        return {"vx": cvx, "vy": cvy, "wz": cwz, "seconds": seconds, "clamped": clamped}

    def submit_look(self, pitch, yaw) -> dict:
        cp, cy = _clamp(pitch, HEAD_PITCH_MAX), _clamp(yaw, HEAD_YAW_MAX)
        clamped = (cp, cy) != (float(pitch), float(yaw))
        with self._lock:
            self._pending.append(LookCmd(cp, cy))
        return {"pitch": cp, "yaw": cy, "clamped": clamped}

    def submit_gesture(self, name) -> dict:
        if name not in GESTURES:
            raise ValueError(f"unknown gesture {name!r}, expected one of {GESTURES}")
        with self._lock:
            self._pending.append(GestureCmd(name))
        return {"gesture": name}

    def submit_stop(self) -> dict:
        with self._lock:
            self._pending.append(StopCmd())
        return {"stopped": True}

    def drain(self) -> list:
        with self._lock:
            drained = list(self._pending)
            self._pending.clear()
        return drained

    def set_status(self, status: dict) -> None:
        with self._lock:
            self._status = dict(status)

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)
