"""ViewerCommander applies bridge commands to a fake env's command terms."""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from bridge.state import BRAIN_TIMEOUT_S, BridgeState
from bridge.watchdog import BrainWatchdog
from viewer_commander import (
    GESTURE_SECONDS,
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


class _Env:
    def __init__(self):
        self.command_manager = _Manager()
        self.scene = {"robot": _Robot()}
        self.resets = 0

    def reset(self):
        self.resets += 1


def _setup(sit_session: bool = False):
    env = _Env()
    state = BridgeState(ViewerLimits(sit_session=sit_session))
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
