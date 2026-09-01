"""Applies queued bridge commands to a PolicyInference on the sim thread.

Every walk has a deadline; expiry returns to the zero command, which is
the trained stand-still state. Head pose indices: head_offset[1] is
head_pitch, head_offset[2] is head_yaw.
"""

from bridge.state import BridgeState, GestureCmd, LookCmd, StopCmd, WalkCmd

GESTURE_KEYS = {"nod": "n", "shake": "m"}

FALLEN_GRAVITY_Z = -0.5  # projected gravity z is near -1 upright, near 0 on the ground

_walk_deadline: float | None = None


def tick(policy, state: BridgeState, now: float) -> None:
    """Drain pending commands, apply them, expire walks, refresh status."""
    global _walk_deadline

    for cmd in state.drain():
        if isinstance(cmd, WalkCmd):
            policy.set_vel_cmd(cmd.vx, cmd.vy, cmd.wz)
            _walk_deadline = now + cmd.seconds
        elif isinstance(cmd, StopCmd):
            _walk_deadline = None
            policy.gesture_player.cancel()
            policy.head_offset[:] = 0.0
            policy.set_vel_cmd(0.0, 0.0, 0.0)
        elif isinstance(cmd, LookCmd):
            policy.gesture_player.cancel()
            policy.head_offset[1] = cmd.pitch
            policy.head_offset[2] = cmd.yaw
            policy._update_command()
        elif isinstance(cmd, GestureCmd):
            policy.start_gesture(GESTURE_KEYS[cmd.name])

    if _walk_deadline is not None and now >= _walk_deadline:
        _walk_deadline = None
        policy.set_vel_cmd(0.0, 0.0, 0.0)

    gravity_z = float(policy.get_projected_gravity()[2])
    state.set_status({
        "policy": policy.current_policy,
        "twist": [float(v) for v in policy.vel_cmd],
        "head": [float(v) for v in policy.head_offset],
        "walk_seconds_left": max(0.0, _walk_deadline - now) if _walk_deadline else 0.0,
        "gesture": policy.gesture_player.active_name,
        "fallen": gravity_z > FALLEN_GRAVITY_Z,
    })


def reset() -> None:
    """Clear module state (tests only)."""
    global _walk_deadline
    _walk_deadline = None
