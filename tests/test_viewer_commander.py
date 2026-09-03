"""ViewerCommander applies bridge commands to a fake env's command terms."""
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from bridge.state import BridgeState
from viewer_commander import (
    GESTURE_SECONDS,
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


def _setup():
    env = _Env()
    state = BridgeState(ViewerLimits())
    commander = ViewerCommander(env, state, DT)
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
