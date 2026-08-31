"""Scripted head gestures (nod yes / shake no) for Microduck.

A gesture needs no policy of its own. The velocity policy already takes a
four-dimensional head pose command -- [neck_pitch, head_pitch, head_yaw,
head_roll] -- and was trained to track it (`head_pose_tracking` is a primary
reward in the velocity env). So "nod yes" is head_pitch swinging over time and
"shake no" is head_yaw swinging over time: a command trajectory, not a network.

This module produces that trajectory and nothing else. It knows nothing about
MuJoCo, ONNX or the keyboard. `infer_policy.py` adds the offset returned here
on top of whatever head pose the user has dialled in, so a gesture composes
with manual head control instead of fighting it.

Shape of the motion: a sine on one axis, multiplied by a Hann window spanning
the whole gesture. The window is what keeps the first and last servo command
from stepping. A bare sine begins and ends at zero but at maximum velocity,
which on real servos reads as a twitch at both ends.

Gestures only track properly under `--new-cmd-obs`, where the head pose is a
command the policy sees. In legacy mode the same offset is added on top of the
policy's output instead, which still moves the head but is not tracked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Head pose command layout, matching the 13D command vector
# [twist(3), head_pose(4), body_pose(6)] used by the runtime.
HEAD_POSE_DIM = 4

NECK_PITCH = 0
HEAD_PITCH = 1
HEAD_YAW = 2
HEAD_ROLL = 3

AXIS_NAMES = {
    NECK_PITCH: "neck_pitch",
    HEAD_PITCH: "head_pitch",
    HEAD_YAW: "head_yaw",
    HEAD_ROLL: "head_roll",
}


@dataclass(frozen=True)
class GestureCfg:
    """One gesture: a windowed sine on a single head axis.

    Amplitudes stay well inside the ±1.4 rad head command range the velocity
    env trains on, so the policy is never asked to track a pose it has not
    seen during training.
    """

    name: str
    axis: int
    amplitude_rad: float
    period_s: float
    cycles: float

    @property
    def duration_s(self) -> float:
        return self.period_s * self.cycles

    def offset_at(self, elapsed_s: float) -> np.ndarray:
        """Head pose offset `elapsed_s` seconds into the gesture."""
        offset = np.zeros(HEAD_POSE_DIM, dtype=np.float32)
        swing = math.sin(2.0 * math.pi * elapsed_s / self.period_s)
        window = 0.5 * (1.0 - math.cos(2.0 * math.pi * elapsed_s / self.duration_s))
        offset[self.axis] = self.amplitude_rad * swing * window
        return offset


NOD_YES = GestureCfg(
    name="nod yes",
    axis=HEAD_PITCH,
    amplitude_rad=0.35,   # ~20 deg
    period_s=0.6,
    cycles=3,
)

SHAKE_NO = GestureCfg(
    name="shake no",
    axis=HEAD_YAW,
    amplitude_rad=0.45,   # ~26 deg
    period_s=0.55,
    cycles=3,
)


def default_gestures() -> dict[str, GestureCfg]:
    """Keyboard key -> gesture, for `infer_policy.py`."""
    return {"n": NOD_YES, "m": SHAKE_NO}


class GesturePlayer:
    """Plays at most one gesture at a time and reports its head pose offset.

    The player owns only the gesture's own motion. Where the head sits when a
    gesture starts is the caller's business, which is what lets a nod happen
    while the head is already turned.
    """

    def __init__(self, gestures: dict[str, GestureCfg]):
        self._gestures = dict(gestures)
        self._active: GestureCfg | None = None
        self._elapsed_s = 0.0

    @property
    def is_playing(self) -> bool:
        return self._active is not None

    @property
    def active_name(self) -> str | None:
        return self._active.name if self._active else None

    def keys(self) -> tuple[str, ...]:
        return tuple(self._gestures)

    def trigger(self, key: str) -> GestureCfg | None:
        """Start the gesture bound to `key`, restarting it if already playing.

        Returns None for an unbound key, so the caller can fall through to its
        other key handling.
        """
        cfg = self._gestures.get(key)
        if cfg is None:
            return None
        self._active = cfg
        self._elapsed_s = 0.0
        return cfg

    def cancel(self) -> None:
        self._active = None
        self._elapsed_s = 0.0

    def advance(self, dt_s: float) -> np.ndarray | None:
        """Advance the clock by `dt_s` and return the current head pose offset.

        Returns None while idle, so the caller leaves the head pose alone
        rather than writing a zero over it every step. Returns a zero offset on
        the step that ends a gesture, so the head settles back exactly onto the
        pose it started from.
        """
        if self._active is None:
            return None
        self._elapsed_s += dt_s
        if self._elapsed_s >= self._active.duration_s:
            self.cancel()
            return np.zeros(HEAD_POSE_DIM, dtype=np.float32)
        return self._active.offset_at(self._elapsed_s)
