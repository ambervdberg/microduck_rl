"""ViewerCommander applies bridge commands to a fake env's command terms."""
import math
import os
import sys
import types

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from bridge.follower import (
    FACE_START,
    FACE_STOP,
    FACE_TURN_MIN,
    HUNT_RATE,
    LOST_AFTER_S,
    SEARCH_PITCH,
    TAN_HALF_WIDTH,
)
from bridge.state import (
    BRAIN_TIMEOUT_S,
    GET_UP_SECONDS,
    GROUND_PICK_SECONDS,
    KICK_SECONDS,
    NO_TRICK,
    ROLL_SECONDS,
    BallCmd,
    BridgeState,
    WalkCmd,
)
from bridge.watchdog import BrainWatchdog
from viewer_commander import (
    GESTURE_SECONDS,
    HEAD_PITCH,
    HEAD_YAW,
    PICTURE_EVERY,
    RISE_SECONDS,
    ViewerCommander,
    ViewerLimits,
)

DT = 0.02


class _Term:
    def __init__(self, dim):
        self.vel_command_b = torch.zeros(1, 3)
        self._command = torch.zeros(1, dim)
        self.is_standing_env = torch.ones(1, dtype=torch.bool)

    def _resample_command(self, env_ids):
        raise AssertionError("resample must be pinned off")


class _Manager:
    def __init__(self):
        self.terms = {"twist": _Term(3), "head_pose": _Term(4), "body_pose": _Term(6)}

    def get_term(self, name):
        return self.terms[name]


class _Robot:
    class data:
        projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]])
        root_link_lin_vel_b = torch.tensor([[0.25, 0.0, 0.0]])
        root_link_ang_vel_b = torch.tensor([[0.0, 0.0, 0.1]])
        root_link_pos_w = torch.tensor([[1.0, 2.0, 0.11]])
        # Yaw 90 deg: the robot faces world +y.
        root_link_quat_w = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])


class _Ball:
    """Records the root writes the commander makes."""

    def __init__(self):
        self.poses = []
        self.velocities = []

    def write_root_link_pose_to_sim(self, pose, env_ids=None):
        self.poses.append(pose[0].tolist())

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        self.velocities.append(velocity[0].tolist())


class _CameraData:
    def __init__(self):
        self.rgb = torch.zeros(1, 120, 160, 3, dtype=torch.uint8)


class _Camera:
    """A picture the tests paint: black, with an orange dot where the test puts it."""

    def __init__(self):
        self._data = _CameraData()
        self.reads = 0

    @property
    def data(self):
        self.reads += 1
        return self._data

    def paint(self, col, row, radius=4):
        self._data.rgb.zero_()
        self._data.rgb[0, row - radius:row + radius, col - radius:col + radius] = torch.tensor(
            [255, 140, 0], dtype=torch.uint8)

    def clear(self):
        self._data.rgb.zero_()


class _FallenRobot:
    class data:
        projected_gravity_b = torch.tensor([[0.0, 0.0, 0.0]])
        root_link_lin_vel_b = torch.tensor([[0.0, 0.0, 0.0]])
        root_link_ang_vel_b = torch.tensor([[0.0, 0.0, 0.0]])
        root_link_pos_w = torch.tensor([[1.0, 2.0, 0.05]])
        root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


class _Env:
    def __init__(self, ball: bool = False, camera: bool = False):
        self.command_manager = _Manager()
        self.scene = {"robot": _Robot()}
        if ball:
            self.scene["ball"] = _Ball()
        if camera:
            self.scene["head_camera"] = _Camera()
        self.resets = 0
        self.device = "cpu"

    def reset(self):
        self.resets += 1


def _setup(
    sit_session=False,
    roll=False,
    get_up=False,
    kick_right=False,
    kick_left=False,
    ground_pick=False,
    camera=False,
):
    env = _Env(ball=kick_right or kick_left, camera=camera)
    limits = ViewerLimits(
        sit_session=sit_session,
        roulade_session=roll,
        standup_session=get_up,
        kick_right_session=kick_right,
        kick_left_session=kick_left,
        ground_pick_session=ground_pick,
        camera=camera,
    )
    state = BridgeState(limits)
    commander = ViewerCommander(env, state, DT)
    return env, state, commander


def _sitting_setup():
    """A commander with a sitstand policy loaded, already seated."""
    env, state, commander = _setup(sit_session=True)
    state.submit_posture(True)
    commander.tick()
    return env, state, commander


def _twist(env):
    return env.command_manager.get_term("twist").vel_command_b[0].tolist()


def _head(env):
    return env.command_manager.get_term("head_pose")._command[0].tolist()


def test_pinning_disables_resampling_and_standing():
    env, _, _ = _setup()
    twist = env.command_manager.get_term("twist")
    twist._resample_command(torch.tensor([0]))  # no longer raises
    assert not bool(twist.is_standing_env[0])


def test_walk_is_written_then_expires():
    env, state, commander = _setup()
    state.submit_walk(0.3, 0.0, 0.5, 1.0)
    commander.tick()
    assert _twist(env) == pytest.approx([0.3, 0.0, 0.5])

    for _ in range(int(1.0 / DT) + 1):
        commander.tick()
    assert _twist(env) == [0.0, 0.0, 0.0]


def test_walk_is_clamped_to_limits():
    env, state, commander = _setup()
    echo = state.submit_walk(2.0, 0.0, -9.0, 2.0)
    commander.tick()
    assert echo["clamped"] is True
    assert _twist(env) == pytest.approx([0.4, 0.0, -1.0])


def test_look_sets_head_pitch_and_yaw():
    env, state, commander = _setup()
    state.submit_look(0.4, -0.6)
    commander.tick()
    assert _head(env) == pytest.approx([0.0, 0.4, -0.6, 0.0])


def test_stop_zeroes_everything():
    env, state, commander = _setup()
    state.submit_walk(0.3, 0.0, 0.0, 5.0)
    state.submit_look(0.4, 0.2)
    commander.tick()
    state.submit_stop()
    commander.tick()
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]


def test_reset_zeroes_everything_and_respawns_the_robot():
    env, state, commander = _setup()
    state.submit_walk(0.3, 0.0, 0.0, 5.0)
    state.submit_look(0.4, 0.2)
    commander.tick()
    state.submit_reset()
    commander.tick()
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]
    assert env.resets == 1
    assert state.get_status()["walk_seconds_left"] == 0.0


def test_status_lists_the_actions_the_viewer_policy_can_run():
    _env, state, commander = _setup()
    commander.tick()
    actions = state.get_status()["actions"]
    assert actions["walk"] is True
    assert actions["nod"] is True
    assert actions["roulade"] is False


def test_nod_moves_pitch_then_returns():
    env, state, commander = _setup()
    state.submit_gesture("nod")
    commander.tick()
    for _ in range(int(0.2 / DT)):
        commander.tick()
    assert _head(env)[1] != 0.0
    assert _head(env)[2] == 0.0

    for _ in range(int(GESTURE_SECONDS / DT) + 2):
        commander.tick()
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]
    assert state.get_status()["gesture"] is None


def test_status_reports_twist_head_and_upright():
    _env, state, commander = _setup()
    state.submit_walk(0.2, 0.0, 0.0, 3.0)
    commander.tick()
    status = state.get_status()
    assert status["ready"] is True
    assert status["twist"] == pytest.approx([0.2, 0.0, 0.0])
    assert status["fallen"] is False
    assert status["walk_seconds_left"] > 2.9
    assert status["measured_twist"] == pytest.approx([0.25, 0.0, 0.1])


class _Checkbox:
    value = True


def test_chat_command_switches_the_viser_sliders_off():
    env, state, commander = _setup()
    env.command_manager.get_term("twist")._joystick_enabled = _Checkbox()
    state.submit_walk(0.2, 0.0, 0.0, 3.0)
    commander.tick()
    assert env.command_manager.get_term("twist")._joystick_enabled.value is False


def test_reset_takes_control_back_from_the_viser_sliders():
    env, state, commander = _setup()
    env.command_manager.get_term("twist")._joystick_enabled = _Checkbox()
    state.submit_reset()
    commander.tick()
    assert env.command_manager.get_term("twist")._joystick_enabled.value is False


# The bridge default is 10 s, as long as the longest walk. A short one keeps the tests quick.
SHORT_TIMEOUT_S = 0.2


def _tick_silently(commander, seconds):
    """Run the commander for a stretch of sim time with no brain request in between."""
    for _ in range(int(round(seconds / DT))):
        commander.tick()


def _impatient(commander, state):
    """Swap in a watchdog that gives up after SHORT_TIMEOUT_S instead of BRAIN_TIMEOUT_S."""
    commander._watchdog = BrainWatchdog(state, DT, timeout_s=SHORT_TIMEOUT_S)


def test_watchdog_zeroes_a_look_after_silence():
    env, state, commander = _setup()
    state.submit_look(0.4, -0.6)
    commander.tick()

    _tick_silently(commander, BRAIN_TIMEOUT_S + 1.0)
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]


def test_watchdog_stops_a_walk_in_progress():
    env, state, commander = _setup()
    _impatient(commander, state)
    state.submit_walk(0.3, 0.0, 0.0, 5.0)
    commander.tick()

    _tick_silently(commander, SHORT_TIMEOUT_S + 0.1)
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert state.get_status()["walk_seconds_left"] == 0.0


def test_watchdog_leaves_a_running_gesture_alone():
    env, state, commander = _setup()
    _impatient(commander, state)
    state.submit_gesture("nod")
    commander.tick()

    _tick_silently(commander, 0.3)
    assert state.get_status()["gesture"] == "nod"
    assert _head(env)[1] != 0.0


def test_sit_and_stand_are_offered_only_with_a_sitstand_policy():
    _env, state, commander = _setup()
    commander.tick()
    assert state.get_status()["actions"]["sit"] is False

    _env, state, commander = _setup(sit_session=True)
    commander.tick()
    actions = state.get_status()["actions"]
    assert actions["sit"] is True
    assert actions["stand up"] is True


def test_sitting_writes_the_posture_flag_in_the_twist_slot():
    env, state, commander = _sitting_setup()
    assert _twist(env) == pytest.approx([1.0, 0.0, 0.0])
    status = state.get_status()
    assert status["sitting"] is True
    assert status["posture"] == "sitting"
    assert status["policy"] == "sit"


def test_sitting_cancels_a_running_walk():
    env, state, commander = _setup(sit_session=True)
    state.submit_walk(0.3, 0.0, 0.0, 5.0)
    commander.tick()
    state.submit_posture(True)
    commander.tick()
    assert _twist(env) == pytest.approx([1.0, 0.0, 0.0])
    assert state.get_status()["walk_seconds_left"] == 0.0


def test_rising_writes_a_zero_flag_then_returns_to_walking():
    env, state, commander = _sitting_setup()
    state.submit_posture(False)
    commander.tick()
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert state.get_status()["posture"] == "rising"
    assert commander.active_policy() == "sit"

    for _ in range(int(RISE_SECONDS / DT) + 2):
        commander.tick()
    assert state.get_status()["posture"] == "standing"
    assert commander.active_policy() == "walking"


def test_walk_is_refused_while_seated_and_allowed_after_the_rise():
    _env, state, commander = _sitting_setup()
    with pytest.raises(ValueError, match="stand up first"):
        state.submit_walk(0.2, 0.0, 0.0, 2.0)

    state.submit_posture(False)
    for _ in range(int(RISE_SECONDS / DT) + 2):
        commander.tick()
    state.submit_walk(0.2, 0.0, 0.0, 2.0)


def test_looking_still_works_while_seated():
    env, state, commander = _sitting_setup()
    state.submit_look(0.4, -0.6)
    commander.tick()
    assert _head(env) == pytest.approx([0.0, 0.4, -0.6, 0.0])
    assert _twist(env) == pytest.approx([1.0, 0.0, 0.0])


def test_reset_stands_the_robot_back_up():
    env, state, commander = _sitting_setup()
    state.submit_reset()
    commander.tick()
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert state.get_status()["posture"] == "standing"
    assert env.resets == 1


def test_watchdog_release_leaves_the_robot_seated():
    env, state, commander = _sitting_setup()
    _impatient(commander, state)

    _tick_silently(commander, SHORT_TIMEOUT_S + 0.1)
    assert _twist(env) == pytest.approx([1.0, 0.0, 0.0])
    assert state.get_status()["posture"] == "sitting"


def test_a_walk_drained_after_a_sit_in_the_same_tick_is_dropped():
    env, state, commander = _setup(sit_session=True)
    state.submit_posture(True)

    # Queued past the bridge guard on purpose: the commander must drop it too.
    state._enqueue(WalkCmd(0.3, 0.0, 0.0, 5.0))
    commander.tick()

    status = state.get_status()
    assert status["twist"] == [0.0, 0.0, 0.0]
    assert status["walk_seconds_left"] == 0.0
    assert _twist(env) == pytest.approx([1.0, 0.0, 0.0])


def _rolling_setup():
    """A commander with a roulade policy loaded, one tick into the roll."""
    env, state, commander = _setup(roll=True)
    state.submit_trick("roll")
    commander.tick()
    return env, state, commander


def _tick_for(commander, seconds):
    """Run the commander over a stretch of sim time, with a margin of two steps."""
    for _ in range(int(seconds / DT) + 2):
        commander.tick()


def test_tricks_are_offered_only_with_their_policy():
    _env, state, commander = _setup()
    commander.tick()
    actions = state.get_status()["actions"]
    assert actions["roulade"] is False
    assert actions["get up off the floor"] is False

    _env, state, commander = _setup(roll=True, get_up=True)
    commander.tick()
    actions = state.get_status()["actions"]
    assert actions["roulade"] is True
    assert actions["get up off the floor"] is True


def test_roll_without_a_roll_policy_is_refused():
    _env, state, _commander = _setup()
    with pytest.raises(ValueError, match="no roll policy loaded"):
        state.submit_trick("roll")


def test_get_up_without_a_get_up_policy_is_refused():
    _env, state, _commander = _setup()
    with pytest.raises(ValueError, match="no get_up policy loaded"):
        state.submit_trick("get_up")


def test_roll_swaps_the_active_policy_and_zeroes_the_command():
    env, state, commander = _rolling_setup()
    assert commander.active_policy() == "roll"
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]
    assert state.get_status()["trick"] == "rolling"


def test_get_up_swaps_to_its_own_policy():
    _env, state, commander = _setup(get_up=True)
    state.submit_trick("get_up")
    commander.tick()
    assert commander.active_policy() == "get_up"
    assert state.get_status()["trick"] == "getting_up"


def test_roll_cancels_a_running_walk():
    env, state, commander = _setup(roll=True)
    state.submit_walk(0.3, 0.0, 0.0, 5.0)
    commander.tick()
    state.submit_trick("roll")
    commander.tick()
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert state.get_status()["walk_seconds_left"] == 0.0


def test_look_is_ignored_while_a_trick_runs():
    env, state, commander = _rolling_setup()
    state.submit_look(0.4, -0.6)
    commander.tick()
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]


def test_gesture_is_ignored_while_a_trick_runs():
    env, state, commander = _rolling_setup()
    state.submit_gesture("nod")
    _tick_for(commander, 0.2)
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]
    assert state.get_status()["gesture"] is None


def test_roll_hands_back_to_walking_at_the_timer():
    env, state, commander = _rolling_setup()
    _tick_for(commander, ROLL_SECONDS)
    assert commander.active_policy() == "walking"
    assert state.get_status()["trick"] == NO_TRICK
    assert _twist(env) == [0.0, 0.0, 0.0]


def test_get_up_runs_for_its_own_duration():
    _env, state, commander = _setup(get_up=True)
    state.submit_trick("get_up")
    commander.tick()

    _tick_for(commander, ROLL_SECONDS)
    assert commander.active_policy() == "get_up"

    _tick_for(commander, GET_UP_SECONDS)
    assert commander.active_policy() == "walking"


def test_walk_is_refused_while_a_trick_runs_and_allowed_after():
    _env, state, commander = _rolling_setup()
    with pytest.raises(ValueError, match="trick is running"):
        state.submit_walk(0.2, 0.0, 0.0, 2.0)

    _tick_for(commander, ROLL_SECONDS)
    state.submit_walk(0.2, 0.0, 0.0, 2.0)


def test_reset_clears_a_running_trick():
    env, state, commander = _rolling_setup()
    state.submit_reset()
    commander.tick()
    assert commander.active_policy() == "walking"
    assert state.get_status()["trick"] == NO_TRICK
    assert env.resets == 1


def test_status_reports_no_trick_while_walking():
    _env, state, commander = _setup(roll=True)
    commander.tick()
    assert state.get_status()["trick"] == NO_TRICK


def test_body_pose_is_zeroed_every_tick():
    env, _, commander = _setup()
    body = env.command_manager.get_term("body_pose")
    body._command[:] = 0.3
    commander.tick()
    assert body._command[0].tolist() == [0.0] * 6


def test_kick_is_offered_only_with_its_own_policy():
    _env, state, commander = _setup(kick_left=True)
    commander.tick()
    actions = state.get_status()["actions"]
    assert actions["kick left"] is True
    assert actions["kick right"] is False


def test_kick_swaps_to_its_own_policy_and_zeroes_the_command():
    env, state, commander = _setup(kick_right=True)
    state.submit_kick("right")
    commander.tick()
    assert commander.active_policy() == "kick_right"
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]
    assert state.get_status()["trick"] == "kicking"


def test_kick_hands_back_to_walking_at_the_timer():
    _env, state, commander = _setup(kick_left=True)
    state.submit_kick("left")
    commander.tick()
    _tick_for(commander, KICK_SECONDS)
    assert commander.active_policy() == "walking"
    assert state.get_status()["trick"] == NO_TRICK


def test_new_ball_lands_in_front_of_the_right_foot_in_the_robot_frame():
    env, state, commander = _setup(kick_right=True)
    state.submit_ball("right")
    commander.tick()
    ball = env.scene["ball"]
    assert ball.poses[-1] == pytest.approx([1.042, 2.09, 0.035, 1.0, 0.0, 0.0, 0.0], abs=1e-4)
    assert ball.velocities[-1] == [0.0] * 6


def test_new_ball_lands_in_front_of_the_left_foot_in_the_robot_frame():
    env, state, commander = _setup(kick_left=True)
    state.submit_ball("left")
    commander.tick()
    assert env.scene["ball"].poses[-1] == pytest.approx([0.958, 2.09, 0.035, 1.0, 0.0, 0.0, 0.0], abs=1e-4)


def test_new_ball_is_ignored_while_a_trick_runs():
    env, state, commander = _setup(roll=True, kick_right=True)
    state.submit_trick("roll")
    commander.tick()
    state._enqueue(BallCmd("right"))
    commander.tick()
    assert env.scene["ball"].poses == []


def test_new_ball_does_not_touch_the_command():
    env, state, commander = _setup(kick_right=True)
    state.submit_look(0.3, 0.0)
    commander.tick()
    state.submit_ball("right")
    commander.tick()
    assert _head(env) == pytest.approx([0.0, 0.3, 0.0, 0.0])


def _ticks(commander, count):
    for _ in range(count):
        commander.tick()


def test_ground_pick_is_offered_only_with_its_policy():
    _env, state, commander = _setup()
    commander.tick()
    assert state.get_status()["actions"]["ground pick"] is False

    _env, state, commander = _setup(ground_pick=True)
    commander.tick()
    assert state.get_status()["actions"]["ground pick"] is True


def test_ground_pick_writes_the_phase_start_and_swaps_policy():
    env, state, commander = _setup(ground_pick=True)
    state.submit_ground_pick()
    commander.tick()
    assert commander.active_policy() == "ground_pick"
    assert state.get_status()["trick"] == "picking"
    assert _twist(env) == pytest.approx([1.0, 0.0, 0.0])


def test_ground_pick_phase_advances_with_time():
    env, state, commander = _setup(ground_pick=True)
    state.submit_ground_pick()
    commander.tick()
    _ticks(commander, int(round(GROUND_PICK_SECONDS / 4 / DT)))
    assert _twist(env) == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    _ticks(commander, int(round(GROUND_PICK_SECONDS / 4 / DT)))
    assert _twist(env) == pytest.approx([-1.0, 0.0, 0.0], abs=1e-6)


def test_ground_pick_zeroes_the_head_even_after_a_look():
    env, state, commander = _setup(ground_pick=True)
    state.submit_look(0.4, 0.2)
    commander.tick()
    state.submit_ground_pick()
    commander.tick()
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]


def test_ground_pick_hands_back_to_walking_at_four_seconds():
    env, state, commander = _setup(ground_pick=True)
    state.submit_ground_pick()
    commander.tick()
    _tick_for(commander, GROUND_PICK_SECONDS)
    assert commander.active_policy() == "walking"
    assert state.get_status()["trick"] == NO_TRICK
    assert _twist(env) == [0.0, 0.0, 0.0]


def _following_setup(col=120, row=60):
    """A commander with a camera, a ball painted in the picture, one tick into the follow."""
    env, state, commander = _setup(kick_right=True, camera=True)
    env.scene["head_camera"].paint(col, row)
    state.submit_follow_ball()
    commander.tick()
    return env, state, commander


def _pictures(commander, count):
    """Enough ticks for this many pictures."""
    _ticks(commander, count * PICTURE_EVERY)


def test_follow_ball_is_offered_only_with_a_camera():
    _env, state, commander = _setup(kick_right=True)
    commander.tick()
    assert state.get_status()["actions"]["follow ball"] is False

    _env, state, commander = _setup(kick_right=True, camera=True)
    commander.tick()
    assert state.get_status()["actions"]["follow ball"] is True


def test_follow_ball_without_a_camera_is_refused():
    _env, state, _commander = _setup(kick_right=True)
    with pytest.raises(ValueError, match="no head camera"):
        state.submit_follow_ball()


def test_the_picture_is_read_every_fifth_tick():
    env, _state, commander = _following_setup()
    camera = env.scene["head_camera"]
    _ticks(commander, PICTURE_EVERY - 2)
    assert camera.reads == 0
    commander.tick()
    assert camera.reads == 1
    _ticks(commander, PICTURE_EVERY)
    assert camera.reads == 2


def test_a_ball_on_the_right_turns_the_head_right():
    env, _state, commander = _following_setup(col=120, row=60)
    _pictures(commander, 1)
    assert _head(env)[HEAD_YAW] < 0.0
    assert _head(env)[HEAD_PITCH] == pytest.approx(0.0, abs=0.01)


def test_a_ball_low_in_the_picture_tilts_the_head_down():
    env, _state, commander = _following_setup(col=80, row=100)
    _pictures(commander, 1)
    assert _head(env)[HEAD_PITCH] > 0.0
    assert _head(env)[HEAD_YAW] == pytest.approx(0.0, abs=0.01)


def test_the_head_keeps_moving_until_the_ball_is_centred():
    env, _state, commander = _following_setup(col=120, row=60)
    _pictures(commander, 1)
    first = _head(env)[HEAD_YAW]
    _pictures(commander, 1)
    assert _head(env)[HEAD_YAW] < first


def test_status_reports_the_follow_and_the_sighting():
    _env, state, commander = _following_setup(col=120, row=60)
    _pictures(commander, 1)
    status = state.get_status()
    assert status["following"] is True
    assert status["ball_seen"] is True
    assert status["searching"] is False
    assert status["ball"]["x"] > 0.4
    assert status["ball"]["size"] == 64


def test_an_empty_picture_reports_no_ball_then_a_search():
    env, state, commander = _following_setup()
    env.scene["head_camera"].clear()
    _pictures(commander, 1)
    status = state.get_status()
    assert status["ball_seen"] is False
    assert status["ball"] is None
    assert status["searching"] is False

    _tick_for(commander, LOST_AFTER_S)
    status = state.get_status()
    assert status["searching"] is True
    assert _head(env)[HEAD_PITCH] == pytest.approx(SEARCH_PITCH)
    assert _head(env)[HEAD_YAW] != 0.0


def test_stop_ends_the_follow_and_zeroes_the_head():
    env, state, commander = _following_setup()
    _pictures(commander, 1)
    state.submit_stop()
    commander.tick()
    assert state.get_status()["following"] is False
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]


def test_a_look_takes_the_head_back():
    env, state, commander = _following_setup()
    _pictures(commander, 1)
    state.submit_look(0.2, 0.3)
    commander.tick()
    _pictures(commander, 2)
    assert state.get_status()["following"] is False
    assert _head(env) == pytest.approx([0.0, 0.2, 0.3, 0.0])


def test_a_gesture_ends_the_follow():
    _env, state, commander = _following_setup()
    state.submit_gesture("nod")
    commander.tick()
    assert state.get_status()["following"] is False


def test_a_gesture_after_a_follow_starts_from_home():
    env, state, commander = _following_setup(col=120, row=60)
    _pictures(commander, 1)
    assert _head(env)[HEAD_YAW] != 0.0

    state.submit_gesture("nod")
    commander.tick()
    assert _head(env)[HEAD_YAW] == 0.0


def test_a_sit_ends_the_follow():
    _env, state, commander = _setup(sit_session=True, kick_right=True, camera=True)
    state.submit_follow_ball()
    commander.tick()
    state.submit_posture(True)
    commander.tick()
    assert state.get_status()["following"] is False


def test_a_trick_ends_the_follow():
    _env, state, commander = _following_setup()
    state.submit_kick("right")
    commander.tick()
    assert state.get_status()["following"] is False
    _tick_for(commander, KICK_SECONDS)
    assert state.get_status()["following"] is False


def test_a_walk_keeps_the_follow():
    env, state, commander = _following_setup(col=120, row=60)
    state.submit_walk(0.2, 0.0, 0.0, 2.0)
    commander.tick()
    _pictures(commander, 1)
    assert state.get_status()["following"] is True
    assert _twist(env) == pytest.approx([0.2, 0.0, 0.0])
    assert _head(env)[HEAD_YAW] < 0.0


def test_a_fall_pauses_the_head_moves_and_keeps_the_follow():
    env, state, commander = _following_setup(col=120, row=60)
    env.scene["robot"] = _FallenRobot()
    _pictures(commander, 2)
    assert env.scene["head_camera"].reads == 0
    assert _head(env)[HEAD_YAW] == 0.0
    status = state.get_status()
    assert status["following"] is True
    assert status["fallen"] is True

    env.scene["robot"] = _Robot()
    _pictures(commander, 1)
    assert _head(env)[HEAD_YAW] < 0.0


def test_the_watchdog_leaves_a_running_follow_alone():
    env, state, commander = _following_setup(col=120, row=60)
    _pictures(commander, 1)
    _tick_silently(commander, BRAIN_TIMEOUT_S + 1.0)
    assert state.peek_status()["following"] is True
    assert _head(env)[HEAD_YAW] < 0.0


def test_reset_ends_the_follow():
    _env, state, commander = _following_setup()
    state.submit_reset()
    commander.tick()
    assert state.get_status()["following"] is False


def _facing_setup(col=120, row=60):
    """A commander with a camera and a walker, a ball painted, one tick into the face."""
    env, state, commander = _setup(kick_right=True, camera=True)
    env.scene["head_camera"].paint(col, row)
    state.submit_face_ball()
    commander.tick()
    return env, state, commander


def _turn_head_past_start(commander, pictures=4):
    """Enough pictures for a ball at the picture edge to push the head yaw past FACE_START."""
    _pictures(commander, pictures)


def test_face_ball_is_offered_only_with_a_camera():
    _env, state, commander = _setup(kick_right=True, camera=True)
    commander.tick()
    assert state.get_status()["actions"]["face ball"] is True

    _env, state, commander = _setup(kick_right=True)
    commander.tick()
    assert state.get_status()["actions"]["face ball"] is False


def test_facing_starts_the_follow_too():
    _env, state, _commander = _facing_setup()
    status = state.get_status()
    assert status["facing"] is True
    assert status["following"] is True


def test_the_body_turns_right_once_the_head_looks_far_right():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    assert _head(env)[HEAD_YAW] < -FACE_START
    assert _twist(env)[2] < 0.0
    assert _twist(env)[0] == 0.0
    assert state.get_status()["turning"] is True


def test_the_body_holds_while_the_ball_is_straight_ahead():
    # Col 80 is the picture's true middle, so the bearing here is 0 rad.
    env, state, commander = _facing_setup(col=80, row=60)
    _pictures(commander, 2)
    assert abs(_head(env)[HEAD_YAW]) < FACE_STOP
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert state.get_status()["turning"] is False


def test_the_turn_stays_under_the_envelope():
    env, state, commander = _facing_setup(col=156, row=60)
    _pictures(commander, 20)
    assert _twist(env)[2] == pytest.approx(-state.policy().vel_max_ang)


def test_a_chat_walk_ends_the_face_and_keeps_the_follow():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    state.submit_walk(0.2, 0.0, 0.0, 2.0)
    commander.tick()
    status = state.get_status()
    assert status["facing"] is False
    assert status["following"] is True
    assert _twist(env) == pytest.approx([0.2, 0.0, 0.0])


def test_a_follow_during_a_face_ends_the_turn_and_keeps_following():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    state.submit_follow_ball()
    commander.tick()
    status = state.get_status()
    assert status["facing"] is False
    assert status["following"] is True
    assert status["turning"] is False
    assert _twist(env) == [0.0, 0.0, 0.0]


def test_stop_ends_the_face():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    state.submit_stop()
    commander.tick()
    assert state.get_status()["facing"] is False
    assert _twist(env) == [0.0, 0.0, 0.0]


def test_a_trick_ends_the_face():
    _env, state, commander = _facing_setup()
    state.submit_kick("right")
    commander.tick()
    assert state.get_status()["facing"] is False


def test_a_fall_ends_the_face():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    env.scene["robot"] = _FallenRobot()
    commander.tick()
    assert _twist(env) == [0.0, 0.0, 0.0]
    status = state.get_status()
    assert status["facing"] is False
    assert status["turning"] is False
    assert status["following"] is True


def test_the_watchdog_ends_the_face_and_keeps_the_follow():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    _tick_silently(commander, BRAIN_TIMEOUT_S + 1.0)
    status = state.peek_status()
    assert status["facing"] is False
    assert status["following"] is True
    assert _twist(env) == [0.0, 0.0, 0.0]


def test_follow_ball_alone_never_turns_the_body():
    env, _state, commander = _following_setup(col=156, row=60)
    _turn_head_past_start(commander)
    assert _head(env)[HEAD_YAW] < -FACE_START
    assert _twist(env) == [0.0, 0.0, 0.0]


class _TurningRobot:
    """A standing robot whose heading the test sets."""

    def __init__(self):
        self.data = types.SimpleNamespace(
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_link_lin_vel_b=torch.tensor([[0.0, 0.0, 0.0]]),
            root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.0]]),
            root_link_pos_w=torch.tensor([[1.0, 2.0, 0.11]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    def face(self, yaw):
        self.data.root_link_quat_w = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])


def _lose_the_ball(env, commander):
    """Clear the picture and wait until the follower searches."""
    env.scene["head_camera"].clear()
    _tick_for(commander, LOST_AFTER_S)
    _pictures(commander, 1)


def _bearing_at(col, yaw):
    """The bearing for a ball painted at this column, given the head yaw after that picture."""
    x = col / 80 - 1
    return yaw - math.atan(x * TAN_HALF_WIDTH)


def _hunt_until_lost(env, commander):
    """Spin the fake body past a full turn with no ball in view."""
    robot = _TurningRobot()
    env.scene["robot"] = robot
    for step in range(1, 16):
        robot.face(-0.5 * step)
        _pictures(commander, 1)


def test_a_lost_ball_turns_the_body_the_way_the_head_looked():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    _lose_the_ball(env, commander)
    assert _twist(env) == pytest.approx([0.0, 0.0, -HUNT_RATE])
    status = state.get_status()
    assert status["turning"] is True
    assert status["lost"] is False


def test_a_sighting_during_the_hunt_hands_back_to_the_turner():
    # Col 36 puts the bearing inside the FACE_STOP..FACE_START gap: without the handover's
    # engage() call the turner would stay off here, so this proves engage() ran.
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    _lose_the_ball(env, commander)
    env.scene["head_camera"].paint(36, 60)
    _pictures(commander, 1)
    yaw = _head(env)[HEAD_YAW]
    bearing = _bearing_at(36, yaw)
    assert FACE_STOP < abs(bearing) < FACE_START
    assert _twist(env)[2] == pytest.approx(math.copysign(FACE_TURN_MIN, bearing))
    assert state.get_status()["lost"] is False


def test_a_full_turn_without_the_ball_gives_up():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    _lose_the_ball(env, commander)
    _hunt_until_lost(env, commander)
    status = state.get_status()
    assert status["lost"] is True
    assert status["facing"] is False
    assert status["following"] is False
    assert status["turning"] is False
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert _head(env) == [0.0, 0.0, 0.0, 0.0]


def test_a_new_face_clears_lost():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    _lose_the_ball(env, commander)
    _hunt_until_lost(env, commander)
    env.scene["head_camera"].paint(120, 60)
    state.submit_face_ball()
    commander.tick()
    status = state.get_status()
    assert status["lost"] is False
    assert status["facing"] is True


def test_stop_clears_lost():
    env, state, commander = _facing_setup(col=156, row=60)
    _turn_head_past_start(commander)
    _lose_the_ball(env, commander)
    _hunt_until_lost(env, commander)
    state.submit_stop()
    commander.tick()
    assert state.get_status()["lost"] is False


def test_a_plain_follow_never_hunts():
    env, state, commander = _following_setup(col=156, row=60)
    _turn_head_past_start(commander)
    _lose_the_ball(env, commander)
    assert _twist(env) == [0.0, 0.0, 0.0]
    assert state.get_status()["lost"] is False


def test_a_ball_a_little_to_the_right_turns_the_body_before_the_head_gets_there():
    env, state, commander = _facing_setup(col=130, row=60)
    _pictures(commander, 1)
    yaw = _head(env)[HEAD_YAW]
    assert -FACE_STOP < yaw < 0.0
    assert _twist(env)[2] < -FACE_START
    assert state.get_status()["turning"] is True


def test_the_handover_from_the_hunt_closes_the_last_bit():
    # Col 60 puts the bearing inside the FACE_STOP..FACE_START gap, same reason as above.
    env, _state, commander = _facing_setup(col=156, row=60)
    _pictures(commander, 2)
    _lose_the_ball(env, commander)
    env.scene["head_camera"].paint(60, 60)
    _pictures(commander, 1)
    yaw = _head(env)[HEAD_YAW]
    bearing = _bearing_at(60, yaw)
    assert FACE_STOP < abs(bearing) < FACE_START
    assert _twist(env)[2] == pytest.approx(math.copysign(FACE_TURN_MIN, bearing))
