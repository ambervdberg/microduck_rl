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


def _clamp_speed(value: float, limit: float) -> float:
    """Clamp a speed to +/- limit; reject non-finite input."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError("value must be a finite number")

    return max(-limit, min(limit, value))


def _clamp_seconds(seconds) -> float:
    """Fill in the default walk duration and clamp it to the trained max."""
    if seconds is None:
        seconds = WALK_DEFAULT_S

    return max(0.0, min(float(seconds), WALK_MAX_S))


def _walk_clamped(vx, vy, wz, seconds, cvx, cvy, cwz, cseconds) -> bool:
    """True if clamping changed any submitted walk value."""
    speeds_changed = (cvx, cvy, cwz) != (float(vx), float(vy), float(wz))
    seconds_changed = (seconds is not None) and (cseconds != float(seconds))

    return speeds_changed or seconds_changed


class BridgeState:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending: deque = deque()
        self._status: dict = {"ready": False}

    def submit_walk(self, vx, vy, wz, seconds) -> dict:
        """Clamp a walk command, queue it, and echo back what will run."""
        cvx, cvy, cwz = _clamp_speed(vx, VX_MAX), _clamp_speed(vy, VY_MAX), _clamp_speed(wz, WZ_MAX)
        cseconds = _clamp_seconds(seconds)
        clamped = _walk_clamped(vx, vy, wz, seconds, cvx, cvy, cwz, cseconds)

        self._enqueue(WalkCmd(cvx, cvy, cwz, cseconds))

        return {"vx": cvx, "vy": cvy, "wz": cwz, "seconds": cseconds, "clamped": clamped}

    def submit_look(self, pitch, yaw) -> dict:
        """Clamp a head pose, queue it, and echo back what will run."""
        cp, cy = _clamp_speed(pitch, HEAD_PITCH_MAX), _clamp_speed(yaw, HEAD_YAW_MAX)
        clamped = (cp, cy) != (float(pitch), float(yaw))

        self._enqueue(LookCmd(cp, cy))

        return {"pitch": cp, "yaw": cy, "clamped": clamped}

    def submit_gesture(self, name) -> dict:
        """Queue a named gesture; reject anything outside the known set."""
        if name not in GESTURES:
            raise ValueError(f"unknown gesture {name!r}, expected one of {GESTURES}")

        self._enqueue(GestureCmd(name))

        return {"gesture": name}

    def submit_stop(self) -> dict:
        """Queue an immediate stop."""
        self._enqueue(StopCmd())

        return {"stopped": True}

    def drain(self) -> list:
        """Return and clear all pending commands, oldest first."""
        with self._lock:
            drained = list(self._pending)
            self._pending.clear()

        return drained

    def set_status(self, status: dict) -> None:
        """Replace the published status snapshot."""
        with self._lock:
            self._status = dict(status)

    def get_status(self) -> dict:
        """Return a copy of the latest published status."""
        with self._lock:
            return dict(self._status)

    def _enqueue(self, cmd) -> None:
        """Append one command to the pending queue."""
        with self._lock:
            self._pending.append(cmd)
