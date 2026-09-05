"""Applies bridge commands to a live mjlab env, so the brain can drive the viser viewer.

The bridge (scripts/bridge) queues WalkCmd / LookCmd / GestureCmd / PostureCmd /
TrickCmd / BallCmd / FollowBallCmd / FaceBallCmd / StopCmd / ResetCmd on a BridgeState. In infer_policy.py
a SkillRunner applies them to PolicyInference. Here ViewerCommander applies them to the env's
command terms instead, once per policy step, and publishes the status the bridge serves on
/status. During a ground pick the twist slots carry the phase encoding the policy was trained
on. A follow reads the head camera every PICTURE_EVERY ticks and steers the head pose slots.
A face writes the turn rate into the twist yaw slot with zero forward speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from bridge.ball_finder import BallSighting, find_ball
from bridge.follower import BallFollower, BodyTurner
from bridge.state import (
    GESTURE_KEYS,
    NO_TRICK,
    RISE_SECONDS,
    TRICKS,
    BallCmd,
    BridgeState,
    FaceBallCmd,
    FollowBallCmd,
    GestureCmd,
    LookCmd,
    PostureCmd,
    ResetCmd,
    StopCmd,
    TrickCmd,
    WalkCmd,
    available_actions,
)
from bridge.watchdog import BrainWatchdog
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import BALL_OFFSET_ABS_Y, BALL_OFFSET_X, BALL_RADIUS

# Head command layout: [neck_pitch, head_pitch, head_yaw, head_roll].
HEAD_PITCH = 1
HEAD_YAW = 2

BODY_POSE_SLOTS = 6

# Name of the head camera sensor, also its key in env.scene.
HEAD_CAMERA = "head_camera"

# Ticks between pictures: 10 pictures per second at the 50 Hz policy rate.
PICTURE_EVERY = 5

GESTURE_SECONDS = 1.6
GESTURE_AMPLITUDE_RAD = 0.35

FALLEN_GRAVITY_Z = -0.5  # projected gravity z above this means the trunk is over


class _GestureKeys:
    """Stands in for PolicyInference.gesture_player where the bridge only asks which keys exist."""

    def keys(self):
        return tuple(GESTURE_KEYS.values())


def _yaw(quat) -> float:
    """Heading of a (w, x, y, z) quaternion about the world z axis."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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
    roulade_session: bool = False  # truthy: a roulade policy is loaded
    standup_session: bool = False  # truthy: a get up policy is loaded
    kick_right_session: bool = False  # truthy: a right foot kick policy is loaded
    kick_left_session: bool = False  # truthy: a left foot kick policy is loaded
    ground_pick_session: bool = False  # truthy: a ground pick policy is loaded
    camera: bool = False  # truthy: a head camera renders in the viewer scene
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
        self._trick: str | None = None
        self._trick_started = 0.0
        self._trick_until = 0.0
        self._following = False
        self._follower: BallFollower | None = None
        self._ticks_since_picture = 0
        self._facing = False
        self._turner: BodyTurner | None = None
        self._turn = 0.0
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
        self._expire_trick()

        if self._watchdog.tick():
            self._release()

        self._follow_tick()

        self._write_tensors()
        self._publish_status()

    def active_policy(self) -> str:
        """Which loaded ONNX drives the robot right now."""
        if self._trick is not None:
            return self._trick

        return "sit" if self._posture in ("sitting", "rising") else "walking"

    # Command handlers.

    def _apply(self, cmd) -> None:
        if self._trick_runs() and not isinstance(cmd, (StopCmd, ResetCmd)):
            return

        self._release_joystick()
        if isinstance(cmd, WalkCmd):
            self._walk(cmd)
        elif isinstance(cmd, LookCmd):
            self._gesture = None
            self._stop_follow()
            self._head = [0.0, cmd.pitch, cmd.yaw, 0.0]
        elif isinstance(cmd, GestureCmd):
            if self._following:
                self._head = [0.0, 0.0, 0.0, 0.0]
            self._stop_follow()
            self._gesture = _Gesture(cmd.name, self._time)
        elif isinstance(cmd, PostureCmd):
            self._set_posture(cmd.sit)
        elif isinstance(cmd, TrickCmd):
            self._start_trick(cmd.name)
        elif isinstance(cmd, BallCmd):
            self._place_ball(cmd.foot)
        elif isinstance(cmd, FollowBallCmd):
            self._start_follow()
        elif isinstance(cmd, FaceBallCmd):
            self._start_face()
        elif isinstance(cmd, StopCmd):
            self._clear_commands()
        elif isinstance(cmd, ResetCmd):
            self._clear_commands()
            self._clear_trick()
            self._posture = "standing"
            self._env.reset()

    def _walk(self, cmd: WalkCmd) -> None:
        """Start walking. A walk drained after a sit in the same tick is dropped."""
        if self._posture != "standing":
            return

        self._stop_face()
        self._twist = [cmd.vx, cmd.vy, cmd.wz]
        self._walk_until = self._time + cmd.seconds

    def _set_posture(self, sit: bool) -> None:
        """Sit down at once, or start the rise glide back to standing."""
        if sit:
            self._posture = "sitting"
            self._twist = [0.0, 0.0, 0.0]
            self._walk_until = 0.0
            self._stop_follow()
            return

        if self._posture == "sitting":
            self._posture = "rising"
            self._rise_until = self._time + RISE_SECONDS

    def _expire_rise(self) -> None:
        """The rise glide is over: the robot is standing again."""
        if self._posture == "rising" and self._time >= self._rise_until:
            self._posture = "standing"

    def _start_trick(self, name: str) -> None:
        """Hand the robot to a trick policy on an all-zero command block."""
        self._clear_commands()
        self._trick = name
        self._trick_started = self._time
        self._trick_until = self._time + TRICKS[name].seconds

    def _expire_trick(self) -> None:
        """The trick is over: the walking policy takes over on a zero twist."""
        if self._trick_runs() and self._time >= self._trick_until:
            self._clear_trick()

    def _clear_trick(self) -> None:
        """Drop the trick timer and report no trick again."""
        self._trick = None
        self._trick_until = 0.0

    def _trick_runs(self) -> bool:
        """True while a trick policy owns the robot."""
        return self._trick is not None

    def _trick_status(self) -> str:
        """The running trick under the name /status speaks, or "none"."""
        if self._trick is None:
            return NO_TRICK

        return TRICKS[self._trick].status

    def _release(self) -> None:
        """The brain went quiet: zero the twist and, unless a gesture or a follow runs, the head."""
        self._twist = [0.0, 0.0, 0.0]
        self._walk_until = 0.0
        self._stop_face()

        if self._gesture is None and not self._following:
            self._head = [0.0, 0.0, 0.0, 0.0]

    def _clear_commands(self) -> None:
        """Zero the twist, the head, the gesture, the walk countdown and any follow."""
        self._twist = [0.0, 0.0, 0.0]
        self._head = [0.0, 0.0, 0.0, 0.0]
        self._gesture = None
        self._walk_until = 0.0
        self._stop_follow()

    # Ball.

    def _place_ball(self, foot: str) -> None:
        """Put the ball at the training spot in front of one foot, at rest."""
        robot = self._env.scene["robot"].data
        x, y = float(robot.root_link_pos_w[0, 0]), float(robot.root_link_pos_w[0, 1])
        yaw = _yaw(robot.root_link_quat_w[0])
        off_y = -BALL_OFFSET_ABS_Y if foot == "right" else BALL_OFFSET_ABS_Y

        ball_x = x + math.cos(yaw) * BALL_OFFSET_X - math.sin(yaw) * off_y
        ball_y = y + math.sin(yaw) * BALL_OFFSET_X + math.cos(yaw) * off_y
        pose = torch.tensor([[ball_x, ball_y, BALL_RADIUS, 1.0, 0.0, 0.0, 0.0]], device=self._env.device)

        ball = self._env.scene["ball"]
        ball.write_root_link_pose_to_sim(pose)
        ball.write_root_link_velocity_to_sim(torch.zeros(1, 6, device=self._env.device))

    # Follow ball.

    def _start_follow(self) -> None:
        """The head camera takes over the head. The follower runs at picture rate."""
        self._gesture = None
        self._follower = BallFollower(PICTURE_EVERY * self._dt)
        self._following = True
        self._ticks_since_picture = 0

    def _stop_follow(self) -> None:
        """Drop the follower. The head stays where it is until the next command."""
        self._stop_face()
        self._following = False
        self._follower = None

    def _follow_tick(self) -> None:
        """Every PICTURE_EVERY ticks: read the picture, find the ball, move the head toward it."""
        if not self._following or self._is_fallen():
            self._turn = 0.0
            return

        self._ticks_since_picture += 1
        if self._ticks_since_picture < PICTURE_EVERY:
            return

        self._ticks_since_picture = 0
        sighting = find_ball(self._picture())
        pitch, yaw = self._follower.update(sighting, self._head[HEAD_PITCH], self._head[HEAD_YAW])
        self._head = [0.0, pitch, yaw, 0.0]
        self._turn_tick(yaw)

    def _picture(self):
        """The latest head camera picture as height by width by 3, on the CPU."""
        return self._env.scene[HEAD_CAMERA].data.rgb[0].cpu().numpy()

    def _sighting(self) -> BallSighting | None:
        """The ball as of the last picture, or None."""
        if self._follower is None:
            return None

        return self._follower.sighting

    def _ball_status(self) -> dict | None:
        """The latest sighting as x, y and size for /status, or None."""
        sighting = self._sighting()
        if sighting is None:
            return None

        return {"x": round(sighting.x, 3), "y": round(sighting.y, 3), "size": sighting.size}

    def _searching(self) -> bool:
        return self._follower is not None and self._follower.searching

    def _start_face(self) -> None:
        """Follow with the head and let the body catch up with it."""
        self._start_follow()
        self._turner = BodyTurner(float(self._state.policy().vel_max_ang))
        self._facing = True

    def _stop_face(self) -> None:
        """Stop turning. The follow, if any, goes on."""
        self._facing = False
        self._turner = None
        self._turn = 0.0

    def _turn_tick(self, yaw: float) -> None:
        """Turn rate from the head yaw, once per picture."""
        if not self._facing:
            return

        self._turn = self._turner.update(yaw, self._follower.searching)

    def _turning(self) -> bool:
        return self._turner is not None and self._turner.turning

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

        head_term = manager.get_term("head_pose")
        for i, value in enumerate(self._head_command()):
            head_term._command[:, i] = value

        # The play cfg samples a random body pose at reset and the term is pinned.
        # Every policy here was trained on a zero or near zero body pose.
        body_term = manager.get_term("body_pose")
        for i in range(BODY_POSE_SLOTS):
            body_term._command[:, i] = 0.0

    def _twist_command(self) -> list[float]:
        """The twist slots: posture flag while sitting, phase encoding during a ground pick, else the walk."""
        if self._trick == "ground_pick":
            return self._ground_pick_phase_command()

        if self._posture == "sitting":
            return [1.0, 0.0, 0.0]

        if self._posture == "rising":
            return [0.0, 0.0, 0.0]

        if self._facing:
            return [0.0, 0.0, self._turn]

        return self._twist

    def _ground_pick_phase_command(self) -> list[float]:
        """[cos(2πφ), sin(2πφ), 0], φ from 0 at the start to 1 at the end of the cycle."""
        phase = (self._time - self._trick_started) / TRICKS["ground_pick"].seconds
        return [math.cos(2.0 * math.pi * phase), math.sin(2.0 * math.pi * phase), 0.0]

    def _head_command(self) -> list[float]:
        """The head slots: zero during a trick, else the held pose plus the gesture wave."""
        if self._trick_runs():
            return [0.0, 0.0, 0.0, 0.0]

        head = list(self._head)
        offset = self._gesture_offset()
        if offset is not None:
            axis, value = offset
            head[axis] += value
        return head

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
            "trick": self._trick_status(),
            "fallen": self._is_fallen(),
            "following": self._following,
            "ball_seen": self._sighting() is not None,
            "ball": self._ball_status(),
            "searching": self._searching(),
            "facing": self._facing,
            "turning": self._turning(),
            "actions": dict(self._actions),
        })
