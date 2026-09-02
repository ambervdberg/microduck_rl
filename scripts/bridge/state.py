"""Thread-safe command state for the LLM bridge.

The HTTP thread submits commands. The sim thread drains and applies them.
Every value is clamped to the envelope the policy itself was given, so the
bridge can never ask for more than the keyboard can.
"""

import math
import threading
from collections import deque
from dataclasses import dataclass

# scripts/ is sys.path[0] when infer_policy.py runs.
from gestures import default_gestures

HEAD_PITCH_MAX = 1.1   # rad, trained neck/head_pitch cap
HEAD_YAW_MAX = 1.4     # rad, trained head_yaw cap

WALK_DEFAULT_S = 3.0
WALK_MAX_S = 10.0

# Seconds of silence from the brain before the bridge releases twist and head.
BRAIN_TIMEOUT_S = 10.0

_GESTURE_TABLE = default_gestures()

# Short API name -> keyboard key that starts the gesture.
GESTURE_KEYS = {cfg.name.split()[0]: key for key, cfg in _GESTURE_TABLE.items()}

# Player gesture name -> the short name /gesture and /status both speak.
GESTURE_SHORT_NAMES = {cfg.name: cfg.name.split()[0] for cfg in _GESTURE_TABLE.values()}

GESTURES = tuple(GESTURE_KEYS)


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


@dataclass(frozen=True)
class Envelope:
    """Speed and head limits the bridge clamps against."""

    vx_min: float
    vx_max: float
    vy_min: float
    vy_max: float
    wz_max: float
    head_pitch_max: float
    head_yaw_max: float


def policy_envelope(policy) -> Envelope:
    """Read the envelope main() installed on the policy, tightened per head axis."""
    head_max = float(policy.head_max)

    return Envelope(
        vx_min=float(policy.vel_min_x),
        vx_max=float(policy.vel_max_x),
        vy_min=float(policy.vel_min_y),
        vy_max=float(policy.vel_max_y),
        wz_max=float(policy.vel_max_ang),
        head_pitch_max=min(head_max, HEAD_PITCH_MAX),
        head_yaw_max=min(head_max, HEAD_YAW_MAX),
    )


def _clamp(value, low: float, high: float) -> float:
    """Clamp a value into [low, high]. Non-finite input is rejected."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError("value must be a finite number")

    return max(low, min(high, value))


def _clamp_seconds(seconds) -> float:
    """Fill in the default walk duration and clamp it to the bridge max."""
    if seconds is None:
        seconds = WALK_DEFAULT_S

    return max(0.0, min(float(seconds), WALK_MAX_S))


def _walk_clamped(vx, vy, wz, seconds, cvx, cvy, cwz, cseconds) -> bool:
    """True if clamping changed any submitted walk value."""
    speeds_changed = (cvx, cvy, cwz) != (float(vx), float(vy), float(wz))
    seconds_changed = (seconds is not None) and (cseconds != float(seconds))

    return speeds_changed or seconds_changed


class BridgeState:
    """Command queue plus the status snapshot and the brain liveness counter."""

    def __init__(self, policy):
        self._policy = policy
        self._lock = threading.Lock()
        self._pending: deque = deque()
        self._status: dict = {"ready": False}
        self._request_count = 0

    def submit_walk(self, vx, vy, wz, seconds) -> dict:
        """Clamp a walk to the policy envelope, then queue it and echo what will run."""
        self._note_request()
        self._require_walking_policy()

        envelope = policy_envelope(self._policy)
        cvx = _clamp(vx, envelope.vx_min, envelope.vx_max)
        cvy = _clamp(vy, envelope.vy_min, envelope.vy_max)
        cwz = _clamp(wz, -envelope.wz_max, envelope.wz_max)
        cseconds = _clamp_seconds(seconds)
        clamped = _walk_clamped(vx, vy, wz, seconds, cvx, cvy, cwz, cseconds)

        self._enqueue(WalkCmd(cvx, cvy, cwz, cseconds))

        return {"vx": cvx, "vy": cvy, "wz": cwz, "seconds": cseconds, "clamped": clamped}

    def submit_look(self, pitch, yaw) -> dict:
        """Clamp a head pose, then queue it and echo back what will run."""
        self._note_request()

        envelope = policy_envelope(self._policy)
        cp = _clamp(pitch, -envelope.head_pitch_max, envelope.head_pitch_max)
        cy = _clamp(yaw, -envelope.head_yaw_max, envelope.head_yaw_max)
        clamped = (cp, cy) != (float(pitch), float(yaw))

        self._enqueue(LookCmd(cp, cy))

        return {"pitch": cp, "yaw": cy, "clamped": clamped}

    def submit_gesture(self, name) -> dict:
        """Queue a named gesture. An unknown name or an unbound key is rejected."""
        self._note_request()

        key = GESTURE_KEYS.get(name)

        if key is None:
            raise ValueError(f"unknown gesture {name!r}, expected one of {GESTURES}")

        if key not in self._policy.gesture_player.keys():
            raise ValueError(f"gesture {name!r} is not bound on this policy")

        self._enqueue(GestureCmd(name))

        return {"gesture": name}

    def submit_stop(self) -> dict:
        """Queue an immediate stop."""
        self._note_request()
        self._enqueue(StopCmd())

        return {"stopped": True}

    def drain(self) -> list:
        """Return and clear all pending commands, oldest first."""
        with self._lock:
            drained = list(self._pending)
            self._pending.clear()

        return drained

    def set_status(self, status: dict) -> None:
        """Replace the published status snapshot. Not a brain request."""
        with self._lock:
            self._status = dict(status)

    def get_status(self) -> dict:
        """Return a copy of the latest published status. Counts as a brain request."""
        with self._lock:
            self._request_count += 1

            return dict(self._status)

    def request_count(self) -> int:
        """Number of brain requests served so far, status polls included."""
        with self._lock:
            return self._request_count

    def _note_request(self) -> None:
        """Mark the brain as alive. The watchdog reads this count."""
        with self._lock:
            self._request_count += 1

    def _require_walking_policy(self) -> None:
        """Reject walk commands the command block would silently drop."""
        if not self._policy.walking_session:
            raise ValueError("no walking policy loaded, start infer_policy.py with --walking")

    def _enqueue(self, cmd) -> None:
        """Append one command to the pending queue."""
        with self._lock:
            self._pending.append(cmd)
