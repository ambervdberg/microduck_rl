"""Applies queued bridge commands to one PolicyInference on the sim thread.

Timers run in sim time: tick() is called once per control step and counts
down by control_dt, so a slow viewer never shortens a walk. Head pose
indices: head_offset[1] is head_pitch, head_offset[2] is head_yaw.
"""

from bridge.state import (
    BRAIN_TIMEOUT_S,
    GESTURE_KEYS,
    GESTURE_SHORT_NAMES,
    BridgeState,
    GestureCmd,
    LookCmd,
    ResetCmd,
    StopCmd,
    WalkCmd,
    available_actions,
)

FALLEN_GRAVITY_Z = -0.5  # projected gravity z is near -1 upright, near 0 on the ground


class SkillRunner:
    """Runs one control step: drain the queue, apply commands, publish status."""

    def __init__(self, policy, state: BridgeState, control_dt: float):
        self._policy = policy
        self._state = state
        self._control_dt = float(control_dt)
        self._walk_seconds_left = 0.0
        self._quiet_seconds = 0.0
        self._last_request_count = state.request_count()
        self._released = False
        self._actions = available_actions(policy)

        # Command class to the bound handler that applies it.
        # Reset lands on _stop: PolicyInference has no sim to respawn.
        self._handlers = {
            WalkCmd: self._walk,
            StopCmd: self._stop,
            ResetCmd: self._stop,
            LookCmd: self._look,
            GestureCmd: self._gesture,
        }

    def tick(self, policy_enabled: bool = True) -> None:
        """Apply queued commands, run the sim-time timers, refresh status."""
        for cmd in self._state.drain():
            self._apply(cmd)

        fallen = self._is_fallen()

        if fallen:
            self._stop_walking()

        if policy_enabled:
            self._expire_walk()
            self._watch_brain()

        self._publish_status(policy_enabled, fallen)

    def _apply(self, cmd) -> None:
        """Dispatch one queued command to its handler."""
        handler = self._handlers[type(cmd)]
        handler(cmd)

    def _walk(self, cmd: WalkCmd) -> None:
        """Start walking and arm the sim-time countdown that stops it."""
        self._policy.set_vel_cmd(cmd.vx, cmd.vy, cmd.wz)
        self._walk_seconds_left = cmd.seconds

    def _stop(self, _cmd) -> None:
        """Cancel any walk and gesture, and zero the head and velocity."""
        self._policy.gesture_player.cancel()
        self._policy.head_offset[:] = 0.0
        self._stop_walking()
        self._policy._update_command()

    def _look(self, cmd: LookCmd) -> None:
        """Cancel any gesture and hold the head at the given pose."""
        self._policy.gesture_player.cancel()
        self._policy.head_offset[1] = cmd.pitch
        self._policy.head_offset[2] = cmd.yaw
        self._policy._update_command()

    def _gesture(self, cmd: GestureCmd) -> None:
        """Play the named gesture."""
        self._policy.start_gesture(GESTURE_KEYS[cmd.name])

    def _is_fallen(self) -> bool:
        """True when the trunk is no longer upright, NaN gravity included."""
        gravity_z = float(self._policy.get_projected_gravity()[2])

        # not (<=) reads NaN as fallen, unlike a direct > comparison.
        return not (gravity_z <= FALLEN_GRAVITY_Z)

    def _expire_walk(self) -> None:
        """Count the walk down by one control step and stop it at zero."""
        if self._walk_seconds_left <= 0.0:
            return

        self._walk_seconds_left = max(0.0, self._walk_seconds_left - self._control_dt)

        if self._walk_seconds_left <= 0.0:
            self._stop_walking()

    def _watch_brain(self) -> None:
        """Release twist and head once when no brain request has arrived in a while."""
        count = self._state.request_count()

        if count != self._last_request_count:
            self._last_request_count = count
            self._quiet_seconds = 0.0
            self._released = False
            return

        self._quiet_seconds += self._control_dt

        if self._released or self._quiet_seconds < BRAIN_TIMEOUT_S:
            return

        self._released = True
        self._release()

    def _release(self) -> None:
        """Zero the twist and, unless a gesture is playing, the head pose."""
        self._stop_walking()

        if not self._policy.gesture_player.is_playing:
            self._policy.head_offset[:] = 0.0

        self._policy._update_command()

    def _stop_walking(self) -> None:
        """Cancel the walk countdown and zero the twist if it is not already zero."""
        self._walk_seconds_left = 0.0

        if any(float(v) != 0.0 for v in self._policy.vel_cmd):
            self._policy.set_vel_cmd(0.0, 0.0, 0.0)

    def _publish_status(self, policy_enabled: bool, fallen: bool) -> None:
        """Publish the twist the policy actually sees, plus the live timers."""
        self._state.set_status({
            "ready": policy_enabled,
            "policy": self._policy.current_policy,
            "twist": [float(v) for v in self._policy.command[0:3]],
            "head": [float(v) for v in self._policy.head_offset],
            "walk_seconds_left": self._walk_seconds_left,
            "gesture": self._active_gesture(),
            "fallen": fallen,
            "actions": dict(self._actions),
        })

    def _active_gesture(self) -> str | None:
        """Active gesture under the short name /gesture accepts."""
        name = self._policy.gesture_player.active_name

        return GESTURE_SHORT_NAMES.get(name, name)
