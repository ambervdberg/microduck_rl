"""Applies queued bridge commands to one PolicyInference on the sim thread.

Timers run in sim time: tick() is called once per control step and counts
down by control_dt, so a slow viewer never shortens a walk. Head pose
indices: head_offset[1] is head_pitch, head_offset[2] is head_yaw.
"""

from bridge.state import (
    GESTURE_KEYS,
    GESTURE_SHORT_NAMES,
    NO_TRICK,
    RISE_SECONDS,
    TRICKS,
    BallCmd,
    BridgeState,
    GestureCmd,
    LookCmd,
    PostureCmd,
    ResetCmd,
    StopCmd,
    Trick,
    TrickCmd,
    WalkCmd,
    available_actions,
)
from bridge.watchdog import BrainWatchdog

FALLEN_GRAVITY_Z = -0.5  # projected gravity z is near -1 upright, near 0 on the ground


def _count_down(seconds_left: float, control_dt: float) -> float:
    """Take one control step off a sim-time timer, floored at zero."""
    return max(0.0, seconds_left - control_dt)


class SkillRunner:
    """Runs one control step: drain the queue, apply commands, publish status."""

    def __init__(self, policy, state: BridgeState, control_dt: float):
        self._policy = policy
        self._state = state
        self._control_dt = float(control_dt)
        self._walk_seconds_left = 0.0
        self._rise_seconds_left = 0.0
        self._trick_seconds_left = 0.0
        self._trick_name = NO_TRICK
        self._watchdog = BrainWatchdog(state, control_dt)
        self._actions = available_actions(policy)

        # Command class to the bound handler that applies it.
        self._handlers = {
            WalkCmd: self._walk,
            StopCmd: self._stop,
            ResetCmd: self._reset,
            LookCmd: self._look,
            GestureCmd: self._gesture,
            PostureCmd: self._posture,
            TrickCmd: self._trick,
            BallCmd: self._ball,
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
            self._expire_rise()
            self._expire_trick()

            if self._watchdog.tick():
                self._release()

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

    def _reset(self, cmd) -> None:
        """Stop everything and stand back up. PolicyInference has no sim to respawn."""
        self._stop(cmd)
        self._rise_seconds_left = 0.0
        self._clear_trick()

        if self._policy.sit_mode:
            self._policy.toggle_sit()

    def _look(self, cmd: LookCmd) -> None:
        """Cancel any gesture and hold the head at the given pose."""
        self._policy.gesture_player.cancel()
        self._policy.head_offset[1] = cmd.pitch
        self._policy.head_offset[2] = cmd.yaw
        self._policy._update_command()

    def _gesture(self, cmd: GestureCmd) -> None:
        """Play the named gesture."""
        self._policy.start_gesture(GESTURE_KEYS[cmd.name])

    def _posture(self, cmd: PostureCmd) -> None:
        """Flip the sitstand posture flag, and only when it is not already there."""
        if cmd.sit:
            self._stop_walking()

        if bool(self._policy.sit_mode) == cmd.sit:
            return

        self._policy.toggle_sit()
        self._rise_seconds_left = 0.0 if cmd.sit else RISE_SECONDS

    def _trick(self, cmd: TrickCmd) -> None:
        """Hand the robot to a trick policy and arm the countdown that ends it."""
        trick = TRICKS[cmd.name]

        self._stop_walking()
        self._start_behavior(cmd.name, trick)
        self._trick_name = trick.status
        self._trick_seconds_left = trick.seconds

    def _start_behavior(self, name: str, trick: Trick) -> None:
        """The ground pick runs its own phase clock. Every other trick is a plain session swap."""
        if name == "ground_pick":
            self._policy.trigger_ground_pick()
            return

        self._policy.trigger_behavior(trick.behavior)

    def _ball(self, cmd: BallCmd) -> None:
        """Put the ball in front of one foot. The runner does the teleport, or says there is no ball."""
        self._policy._place_ball(f"kick_{cmd.foot}")

    def trick_name(self) -> str:
        """The running trick under the name /status speaks, or "none"."""
        return self._trick_name

    def _is_fallen(self) -> bool:
        """True when the trunk is no longer upright, NaN gravity included."""
        gravity_z = float(self._policy.get_projected_gravity()[2])

        # not (<=) reads NaN as fallen, unlike a direct > comparison.
        return not (gravity_z <= FALLEN_GRAVITY_Z)

    def _expire_walk(self) -> None:
        """Count the walk down by one control step and stop it at zero."""
        if self._walk_seconds_left <= 0.0:
            return

        self._walk_seconds_left = _count_down(self._walk_seconds_left, self._control_dt)

        if self._walk_seconds_left <= 0.0:
            self._stop_walking()

    def _expire_rise(self) -> None:
        """Count the stand up down by one control step. At zero the robot is standing."""
        if self._rise_seconds_left > 0.0:
            self._rise_seconds_left = _count_down(self._rise_seconds_left, self._control_dt)

    def _expire_trick(self) -> None:
        """Count the trick down by one control step and clear it at zero."""
        if self._trick_seconds_left <= 0.0:
            return

        self._trick_seconds_left = _count_down(self._trick_seconds_left, self._control_dt)

        if self._trick_seconds_left <= 0.0:
            self._clear_trick()

    def _clear_trick(self) -> None:
        """Drop the trick timer and report no trick again."""
        self._trick_seconds_left = 0.0
        self._trick_name = NO_TRICK

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
            "twist": self._status_twist(),
            "head": [float(v) for v in self._policy.head_offset],
            "walk_seconds_left": self._walk_seconds_left,
            "gesture": self._active_gesture(),
            "sitting": bool(self._policy.sit_mode),
            "posture": self._posture_name(),
            "trick": self._trick_name,
            "fallen": fallen,
            "actions": dict(self._actions),
        })

    def _status_twist(self) -> list[float]:
        """The twist the policy sees. On the sit policy slot 0 is the posture flag, not vx."""
        if self._policy.current_policy == "sit":
            return [0.0, 0.0, 0.0]

        return [float(v) for v in self._policy.command[0:3]]

    def _posture_name(self) -> str:
        """Sitting, still getting up, or standing."""
        if self._policy.sit_mode:
            return "sitting"

        return "rising" if self._rise_seconds_left > 0.0 else "standing"

    def _active_gesture(self) -> str | None:
        """Active gesture under the short name /gesture accepts."""
        name = self._policy.gesture_player.active_name

        return GESTURE_SHORT_NAMES.get(name, name)
