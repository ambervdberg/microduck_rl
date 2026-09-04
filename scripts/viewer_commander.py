"""Applies bridge commands to a live mjlab env, so the brain can drive the viser viewer.

The bridge (scripts/bridge) queues WalkCmd / LookCmd / GestureCmd / PostureCmd /
StopCmd / ResetCmd on a BridgeState. In infer_policy.py a SkillRunner applies them
to PolicyInference. Here ViewerCommander applies them to the env's command terms
instead, once per policy step, and publishes the status the bridge serves on /status.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from bridge.state import (
    GESTURE_KEYS,
    BridgeState,
    GestureCmd,
    LookCmd,
    PostureCmd,
    ResetCmd,
    StopCmd,
    WalkCmd,
    available_actions,
)
from bridge.watchdog import BrainWatchdog

# Head command layout: [neck_pitch, head_pitch, head_yaw, head_roll].
HEAD_PITCH = 1
HEAD_YAW = 2

GESTURE_SECONDS = 1.6
GESTURE_AMPLITUDE_RAD = 0.35

# How long the stand up takes: the trained flag flip is about a 2 s glide.
RISE_SECONDS = 2.5
FALLEN_GRAVITY_Z = -0.5  # projected gravity z above this means the trunk is over


class _GestureKeys:
    """Stands in for PolicyInference.gesture_player where the bridge only asks which keys exist."""

    def keys(self):
        return tuple(GESTURE_KEYS.values())


@dataclass
class ViewerLimits:
    """The envelope fields bridge.state.policy_envelope reads off a policy."""

    vel_min_x: float = -0.4
    vel_max_x: float = 0.4
    vel_min_y: float = -0.3
    vel_max_y: float = 0.3
    vel_max_ang: float = 1.0
    head_max: float = 1.4
    walking_session: bool = True  # truthy: a walking policy is loaded
    sit_session: bool = False  # truthy: a sitstand policy is loaded
    gesture_player: _GestureKeys = field(default_factory=_GestureKeys)


@dataclass
class _Gesture:
    name: str
    started_at: float


class ViewerCommander:
    """Drains the bridge queue each step and writes the env command tensors."""

    def __init__(self, env, state: BridgeState, control_dt: float):
        self._env = env
        self._state = state
        self._dt = control_dt
        self._time = 0.0
        self._twist = [0.0, 0.0, 0.0]
        self._head = [0.0, 0.0, 0.0, 0.0]
        self._walk_until = 0.0
        self._gesture: _Gesture | None = None
        self._posture = "standing"
        self._rise_until = 0.0
        self._watchdog = BrainWatchdog(state, control_dt)
        self._actions = available_actions(state.policy())
        self._pin_command_terms()

    def tick(self) -> None:
        """One policy step: apply new commands, expire walks, animate gestures, publish."""
        self._time += self._dt

        for cmd in self._state.drain():
            self._apply(cmd)

        if self._time >= self._walk_until:
            self._twist = [0.0, 0.0, 0.0]

        self._expire_rise()

        if self._watchdog.tick():
            self._release()

        self._write_tensors()
        self._publish_status()

    def active_policy(self) -> str:
        """Which loaded ONNX drives the robot right now."""
        return "sit" if self._posture in ("sitting", "rising") else "walking"

    # Command handlers.

    def _apply(self, cmd) -> None:
        self._release_joystick()
        if isinstance(cmd, WalkCmd):
            self._twist = [cmd.vx, cmd.vy, cmd.wz]
            self._walk_until = self._time + cmd.seconds
        elif isinstance(cmd, LookCmd):
            self._gesture = None
            self._head = [0.0, cmd.pitch, cmd.yaw, 0.0]
        elif isinstance(cmd, GestureCmd):
            self._gesture = _Gesture(cmd.name, self._time)
        elif isinstance(cmd, PostureCmd):
            self._set_posture(cmd.sit)
        elif isinstance(cmd, StopCmd):
            self._clear_commands()
        elif isinstance(cmd, ResetCmd):
            self._clear_commands()
            self._posture = "standing"
            self._env.reset()

    def _set_posture(self, sit: bool) -> None:
        """Sit down at once, or start the rise glide back to standing."""
        if sit:
            self._posture = "sitting"
            self._twist = [0.0, 0.0, 0.0]
            self._walk_until = 0.0
            return

        if self._posture == "sitting":
            self._posture = "rising"
            self._rise_until = self._time + RISE_SECONDS

    def _expire_rise(self) -> None:
        """The rise glide is over: the robot is standing again."""
        if self._posture == "rising" and self._time >= self._rise_until:
            self._posture = "standing"

    def _release(self) -> None:
        """The brain went quiet: zero the twist and, unless a gesture is playing, the head."""
        self._twist = [0.0, 0.0, 0.0]
        self._walk_until = 0.0

        if self._gesture is None:
            self._head = [0.0, 0.0, 0.0, 0.0]

    def _clear_commands(self) -> None:
        """Zero the twist, the head, the gesture and the walk countdown."""
        self._twist = [0.0, 0.0, 0.0]
        self._head = [0.0, 0.0, 0.0, 0.0]
        self._gesture = None
        self._walk_until = 0.0

    # Tensors.

    def _pin_command_terms(self) -> None:
        """Stop the env from resampling commands. The brain owns them now."""
        for name in ("twist", "head_pose", "body_pose"):
            term = self._env.command_manager.get_term(name)
            term._resample_command = lambda env_ids: None
        twist = self._env.command_manager.get_term("twist")
        if hasattr(twist, "is_standing_env"):
            twist.is_standing_env[:] = False

    def _release_joystick(self) -> None:
        """A chat command takes control back from the viser sliders (their Enable box)."""
        joystick = getattr(self._env.command_manager.get_term("twist"), "_joystick_enabled", None)
        if joystick is not None and joystick.value:
            joystick.value = False

    def _write_tensors(self) -> None:
        manager = self._env.command_manager
        twist = manager.get_term("twist")
        for i, value in enumerate(self._twist_command()):
            twist.vel_command_b[:, i] = value

        head = list(self._head)
        offset = self._gesture_offset()
        if offset is not None:
            axis, value = offset
            head[axis] += value
        head_term = manager.get_term("head_pose")
        for i, value in enumerate(head):
            head_term._command[:, i] = value

    def _twist_command(self) -> list[float]:
        """Sitting writes the posture flag in the vx slot, rising writes the stand flag, zero."""
        if self._posture == "sitting":
            return [1.0, 0.0, 0.0]

        if self._posture == "rising":
            return [0.0, 0.0, 0.0]

        return self._twist

    def _gesture_offset(self) -> tuple[int, float] | None:
        """Sine wave on head pitch (nod) or yaw (shake) for GESTURE_SECONDS."""
        if self._gesture is None:
            return None
        elapsed = self._time - self._gesture.started_at
        if elapsed >= GESTURE_SECONDS:
            self._gesture = None
            return None
        axis = HEAD_PITCH if self._gesture.name == "nod" else HEAD_YAW
        cycles = 2.0
        value = GESTURE_AMPLITUDE_RAD * math.sin(2.0 * math.pi * cycles * elapsed / GESTURE_SECONDS)
        return axis, value

    # Status.

    def _is_fallen(self) -> bool:
        robot = self._env.scene["robot"]
        return float(robot.data.projected_gravity_b[0, 2]) > FALLEN_GRAVITY_Z

    def _measured_twist(self) -> list[float]:
        data = self._env.scene["robot"].data
        lin = data.root_link_lin_vel_b[0]
        ang = data.root_link_ang_vel_b[0]
        return [round(float(lin[0]), 3), round(float(lin[1]), 3), round(float(ang[2]), 3)]

    def _publish_status(self) -> None:
        self._state.set_status({
            "ready": True,
            "policy": self.active_policy(),
            "twist": list(self._twist),
            "measured_twist": self._measured_twist(),
            "head": list(self._head),
            "walk_seconds_left": max(0.0, self._walk_until - self._time),
            "gesture": self._gesture.name if self._gesture else None,
            "sitting": self._posture == "sitting",
            "posture": self._posture,
            "fallen": self._is_fallen(),
            "actions": dict(self._actions),
        })
