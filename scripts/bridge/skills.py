"""Applies queued bridge commands to one PolicyInference on the sim thread.

Every walk has a deadline; expiry returns to the zero command, which is
the trained stand-still state. Head pose indices: head_offset[1] is
head_pitch, head_offset[2] is head_yaw.
"""

from bridge.state import BridgeState, GestureCmd, LookCmd, StopCmd, WalkCmd

GESTURE_KEYS = {"nod": "n", "shake": "m"}

FALLEN_GRAVITY_Z = -0.5  # projected gravity z is near -1 upright, near 0 on the ground


class SkillRunner:
    """Runs one tick: drain the queue, apply commands, publish status."""

    def __init__(self, policy, state: BridgeState):
        self._policy = policy
        self._state = state
        self._walk_deadline: float | None = None

        # Command class to the bound handler that applies it.
        self._handlers = {
            WalkCmd: self._walk,
            StopCmd: self._stop,
            LookCmd: self._look,
            GestureCmd: self._gesture,
        }

    def tick(self, now: float) -> None:
        """Drain pending commands, apply them, expire walks, refresh status."""
        for cmd in self._state.drain():
            self._apply(cmd, now)

        self._expire_walk(now)
        self._publish_status(now)

    def _apply(self, cmd, now: float) -> None:
        """Dispatch one queued command to its handler."""
        handler = self._handlers[type(cmd)]
        handler(cmd, now)

    def _walk(self, cmd: WalkCmd, now: float) -> None:
        """Start walking and arm the deadline that stops it."""
        self._policy.set_vel_cmd(cmd.vx, cmd.vy, cmd.wz)
        self._walk_deadline = now + cmd.seconds

    def _stop(self, _cmd: StopCmd, _now: float) -> None:
        """Cancel any walk and gesture, and zero the head and velocity."""
        self._walk_deadline = None
        self._policy.gesture_player.cancel()
        self._policy.head_offset[:] = 0.0
        self._policy.set_vel_cmd(0.0, 0.0, 0.0)
        self._policy._update_command()

    def _look(self, cmd: LookCmd, _now: float) -> None:
        """Cancel any gesture and hold the head at the given pose."""
        self._policy.gesture_player.cancel()
        self._policy.head_offset[1] = cmd.pitch
        self._policy.head_offset[2] = cmd.yaw
        self._policy._update_command()

    def _gesture(self, cmd: GestureCmd, _now: float) -> None:
        """Play the named gesture."""
        self._policy.start_gesture(GESTURE_KEYS[cmd.name])

    def _expire_walk(self, now: float) -> None:
        """Zero velocity if walk deadline has passed."""
        if self._walk_deadline is not None and now >= self._walk_deadline:
            self._walk_deadline = None
            self._policy.set_vel_cmd(0.0, 0.0, 0.0)

    def _publish_status(self, now: float) -> None:
        """Update state with current policy snapshot."""
        gravity_z = float(self._policy.get_projected_gravity()[2])
        # not (<=) reads NaN as fallen, unlike a direct > comparison.
        fallen = not (gravity_z <= FALLEN_GRAVITY_Z)
        self._state.set_status({
            "ready": True,
            "policy": self._policy.current_policy,
            "twist": [float(v) for v in self._policy.vel_cmd],
            "head": [float(v) for v in self._policy.head_offset],
            "walk_seconds_left": max(0.0, self._walk_deadline - now) if self._walk_deadline else 0.0,
            "gesture": self._policy.gesture_player.active_name,
            "fallen": fallen,
        })
