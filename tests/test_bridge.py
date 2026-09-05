"""Tests for the LLM bridge: command state, skills, HTTP server."""

import contextlib
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gestures import default_gestures  # noqa: E402

from bridge.state import (  # noqa: E402
    BRAIN_TIMEOUT_S,
    GET_UP_SECONDS,
    GROUND_PICK_SECONDS,
    HEAD_PITCH_MAX,
    HEAD_YAW_MAX,
    KICK_SECONDS,
    RISE_SECONDS,
    ROLL_SECONDS,
    TRICKS,
    WALK_DEFAULT_S,
    WALK_MAX_S,
    FEET,
    BallCmd,
    BridgeState,
    GestureCmd,
    LookCmd,
    PostureCmd,
    ResetCmd,
    StopCmd,
    TrickCmd,
    WalkCmd,
    available_actions,
)

from bridge import skills  # noqa: E402
from bridge.server import start_bridge  # noqa: E402
from bridge.watchdog import BrainWatchdog  # noqa: E402

CONTROL_DT = 0.02  # 50 Hz, the rate infer_policy.py calls tick() at


class FakeGesturePlayer:
    """Mimics GesturePlayer: the real key table, one gesture at a time."""

    def __init__(self, gestures=None):
        self._gestures = default_gestures() if gestures is None else gestures
        self.active_name = None

    @property
    def is_playing(self):
        return self.active_name is not None

    def keys(self):
        return tuple(self._gestures)

    def trigger(self, key):
        cfg = self._gestures.get(key)

        if cfg is None:
            return None

        self.active_name = cfg.name
        return cfg

    def cancel(self):
        self.active_name = None


class FakePolicy:
    """Mimics the PolicyInference surface the bridge touches."""

    def __init__(
        self,
        gravity_z: float = -1.0,
        walking: bool = True,
        gestures=None,
        sit: bool = False,
        roll: bool = False,
        get_up: bool = False,
        kick_right: bool = False,
        kick_left: bool = False,
        ground_pick: bool = False,
    ):
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.head_offset = np.zeros(4, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)
        self.current_policy = "standing"
        self.gesture_player = FakeGesturePlayer(gestures)
        self.started_gestures = []
        self.command_updates = 0
        self.gravity_z = gravity_z
        self.walking_session = object() if walking else None
        self.sit_session = object() if sit else None
        self.roulade_session = object() if roll else None
        self.standup_session = object() if get_up else None
        self.kick_right_session = object() if kick_right else None
        self.kick_left_session = object() if kick_left else None
        self.ground_pick_session = object() if ground_pick else None
        self.ground_picks = 0
        self.balls = []
        self.behaviors = []
        self.sit_mode = False
        self.sit_toggles = 0
        self.switch_threshold = 0.05
        self.vel_max_x = 0.3
        self.vel_min_x = -0.3
        self.vel_max_y = 0.2
        self.vel_min_y = -0.2
        self.vel_max_ang = 1.5
        self.head_max = 1.4

    def set_vel_cmd(self, vx, vy, wz):
        self.vel_cmd = np.array([vx, vy, wz], dtype=np.float32)
        magnitude = float(np.linalg.norm(self.vel_cmd))
        self.current_policy = "walking" if magnitude > self.switch_threshold else "standing"
        self._update_command()

    def _update_command(self):
        self.command_updates += 1
        cmd = np.zeros(13, dtype=np.float32)

        # Only the walking session owns the twist slots, as in infer_policy.py.
        if self.current_policy == "walking":
            cmd[0:3] = self.vel_cmd

        # The sitstand session writes the posture flag where the twist vx lives.
        if self.current_policy == "sit":
            cmd[0] = 1.0 if self.sit_mode else 0.0

        cmd[3:7] = self.head_offset
        self.command = cmd

    def toggle_sit(self):
        self.sit_toggles += 1
        self.sit_mode = not self.sit_mode
        self.current_policy = "sit" if self.sit_mode else "walking"
        self._update_command()

    def trigger_behavior(self, name):
        """Mimics PolicyInference.trigger_behavior: a session swap with a zero command."""
        self.behaviors.append(name)
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.current_policy = name
        self._update_command()

    def trigger_ground_pick(self):
        """Mimics PolicyInference.trigger_ground_pick: session swap, phase runs from zero."""
        self.ground_picks += 1
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.current_policy = "ground_pick"
        self._update_command()

    def _place_ball(self, behavior):
        """Mimics PolicyInference._place_ball: records which kick foot asked for the ball."""
        self.balls.append(behavior)

    def start_gesture(self, key):
        cfg = self.gesture_player.trigger(key)

        if cfg is None:
            return None

        self.started_gestures.append(key)
        return cfg

    def get_projected_gravity(self):
        return np.array([0.0, 0.0, self.gravity_z], dtype=np.float32)


def _pair(**policy_kwargs):
    """One policy and the state bound to it."""
    policy = FakePolicy(**policy_kwargs)

    return BridgeState(policy), policy


def _tick_seconds(runner, seconds: float, policy_enabled: bool = True) -> None:
    """Run whole control steps worth of sim time."""
    for _ in range(int(round(seconds / CONTROL_DT))):
        runner.tick(policy_enabled)


class TestBridgeState:
    def test_walk_is_queued_and_drained_in_order(self):
        state, _ = _pair()
        state.submit_walk(0.2, 0.0, 0.0, 2.0)
        state.submit_stop()
        drained = state.drain()
        assert drained == [WalkCmd(0.2, 0.0, 0.0, 2.0), StopCmd()]
        assert state.drain() == []

    def test_walk_speeds_are_clamped_to_the_policy_envelope(self):
        state, policy = _pair()
        echo = state.submit_walk(9.0, 0.0, 0.0, 2.0)
        assert echo["vx"] == pytest.approx(policy.vel_max_x)
        assert echo["clamped"] is True
        assert state.drain() == [WalkCmd(policy.vel_max_x, 0.0, 0.0, 2.0)]

    def test_walk_lateral_is_clamped_to_zero_on_the_roller_envelope(self):
        state, policy = _pair()
        policy.vel_max_y = 0.0
        policy.vel_min_y = 0.0
        echo = state.submit_walk(0.0, 0.3, 0.0, 2.0)
        assert echo["vy"] == pytest.approx(0.0)
        assert echo["clamped"] is True

    def test_walk_backward_is_clamped_to_the_asymmetric_minimum(self):
        state, policy = _pair()
        policy.vel_min_x = -0.1
        echo = state.submit_walk(-0.9, 0.0, 0.0, 2.0)
        assert echo["vx"] == pytest.approx(-0.1)

    def test_walk_without_a_walking_policy_is_rejected(self):
        state, _ = _pair(walking=False)
        with pytest.raises(ValueError):
            state.submit_walk(0.2, 0.0, 0.0, 2.0)
        assert state.drain() == []

    def test_walk_duration_default_and_max(self):
        state, _ = _pair()
        assert state.submit_walk(0.1, 0.0, 0.0, None)["seconds"] == WALK_DEFAULT_S
        assert state.submit_walk(0.1, 0.0, 0.0, 60.0)["seconds"] == WALK_MAX_S

    def test_look_clamps_head_angles(self):
        state, _ = _pair()
        echo = state.submit_look(-9.0, 0.5)
        assert echo["pitch"] == pytest.approx(-HEAD_PITCH_MAX)
        assert echo["yaw"] == pytest.approx(0.5)
        assert state.drain() == [LookCmd(-HEAD_PITCH_MAX, 0.5)]

    def test_look_clamps_yaw_to_its_own_max(self):
        state, _ = _pair()
        echo = state.submit_look(0.0, 9.0)
        assert echo["yaw"] == pytest.approx(HEAD_YAW_MAX)
        assert state.drain() == [LookCmd(0.0, HEAD_YAW_MAX)]

    def test_look_takes_the_tighter_of_head_max_and_the_axis_cap(self):
        state, policy = _pair()
        policy.head_max = 0.4
        echo = state.submit_look(9.0, 9.0)
        assert echo["pitch"] == pytest.approx(0.4)
        assert echo["yaw"] == pytest.approx(0.4)
        assert echo["clamped"] is True

    def test_walk_with_nan_speed_raises_and_queues_nothing(self):
        state, _ = _pair()
        with pytest.raises(ValueError):
            state.submit_walk(float("nan"), 0.0, 0.0, 1.0)
        assert state.drain() == []

    def test_unknown_gesture_is_rejected_and_not_queued(self):
        state, _ = _pair()
        with pytest.raises(ValueError):
            state.submit_gesture("backflip")
        assert state.drain() == []
        state.submit_gesture("nod")
        assert state.drain() == [GestureCmd("nod")]

    def test_gesture_unbound_on_the_player_is_rejected(self):
        state, _ = _pair(gestures={})
        with pytest.raises(ValueError):
            state.submit_gesture("nod")
        assert state.drain() == []

    def test_status_roundtrip_returns_a_copy(self):
        state, _ = _pair()
        state.set_status({"policy": "walking"})
        status = state.get_status()
        status["policy"] = "mutated"
        assert state.get_status() == {"policy": "walking"}

    def test_status_poll_counts_as_a_brain_request(self):
        state, _ = _pair()
        before = state.request_count()
        state.get_status()
        assert state.request_count() == before + 1

    def test_walk_seconds_clamping_sets_clamped_flag(self):
        state, _ = _pair()
        echo = state.submit_walk(0.1, 0.0, 0.0, 60.0)
        assert echo["seconds"] == pytest.approx(WALK_MAX_S)
        assert echo["clamped"] is True

    def test_walk_seconds_none_default_not_clamped(self):
        state, _ = _pair()
        echo = state.submit_walk(0.1, 0.0, 0.0, None)
        assert echo["seconds"] == pytest.approx(WALK_DEFAULT_S)
        assert echo["clamped"] is False

    def test_reset_is_queued_and_counts_as_a_brain_request(self):
        state, _ = _pair()
        before = state.request_count()
        assert state.submit_reset() == {"reset": True}
        assert state.drain() == [ResetCmd()]
        assert state.request_count() == before + 1


class TestPostureState:
    def test_sit_and_stand_are_queued(self):
        state, _ = _pair(sit=True)
        assert state.submit_posture(True) == {"sit": True}
        assert state.submit_posture(False) == {"sit": False}
        assert state.drain() == [PostureCmd(True), PostureCmd(False)]

    def test_sit_without_a_sit_policy_is_rejected(self):
        state, _ = _pair()
        with pytest.raises(ValueError):
            state.submit_posture(True)
        assert state.drain() == []

    def test_walk_while_seated_is_rejected(self):
        state, _ = _pair(sit=True)
        state.set_status({"sitting": True, "posture": "sitting"})
        with pytest.raises(ValueError, match="stand up first"):
            state.submit_walk(0.2, 0.0, 0.0, 2.0)
        assert state.drain() == []

    def test_walk_while_rising_is_rejected(self):
        state, _ = _pair(sit=True)
        state.set_status({"sitting": False, "posture": "rising"})
        with pytest.raises(ValueError, match="stand up first"):
            state.submit_walk(0.2, 0.0, 0.0, 2.0)
        assert state.drain() == []

    def test_walk_is_allowed_again_once_standing(self):
        state, _ = _pair(sit=True)
        state.set_status({"sitting": False, "posture": "standing"})
        state.submit_walk(0.2, 0.0, 0.0, 2.0)
        assert state.drain() == [WalkCmd(0.2, 0.0, 0.0, 2.0)]

    def test_walk_with_a_sit_still_queued_is_rejected(self):
        state, _ = _pair(sit=True)
        state.set_status({"sitting": False, "posture": "standing"})
        state.submit_posture(True)
        with pytest.raises(ValueError, match="stand up first"):
            state.submit_walk(0.2, 0.0, 0.0, 2.0)
        assert state.drain() == [PostureCmd(True)]

    def test_walk_with_a_queued_stand_after_the_sit_is_allowed(self):
        state, _ = _pair(sit=True)
        state.set_status({"sitting": False, "posture": "standing"})
        state.submit_posture(True)
        state.submit_posture(False)
        state.submit_walk(0.2, 0.0, 0.0, 2.0)
        assert state.drain() == [PostureCmd(True), PostureCmd(False), WalkCmd(0.2, 0.0, 0.0, 2.0)]

    def test_look_gesture_and_reset_stay_allowed_while_seated(self):
        state, _ = _pair(sit=True)
        state.set_status({"sitting": True, "posture": "sitting"})
        state.submit_look(0.2, 0.1)
        state.submit_gesture("nod")
        state.submit_reset()
        assert state.drain() == [LookCmd(0.2, 0.1), GestureCmd("nod"), ResetCmd()]

    def test_sit_counts_as_a_brain_request(self):
        state, _ = _pair(sit=True)
        before = state.request_count()
        state.submit_posture(True)
        assert state.request_count() == before + 1


class TestTrickState:
    def test_roll_and_get_up_are_queued(self):
        state, _ = _pair(roll=True, get_up=True)
        assert state.submit_trick("roll") == {"trick": "roll"}
        assert state.drain() == [TrickCmd("roll")]

        assert state.submit_trick("get_up") == {"trick": "get_up"}
        assert state.drain() == [TrickCmd("get_up")]

    def test_unknown_trick_is_rejected(self):
        state, _ = _pair(roll=True)
        with pytest.raises(ValueError, match="unknown trick"):
            state.submit_trick("backflip")
        assert state.drain() == []

    def test_roll_without_a_roll_policy_is_rejected(self):
        state, _ = _pair(get_up=True)
        with pytest.raises(ValueError, match="no roll policy loaded"):
            state.submit_trick("roll")
        assert state.drain() == []

    def test_get_up_without_a_get_up_policy_is_rejected(self):
        state, _ = _pair(roll=True)
        with pytest.raises(ValueError, match="no get_up policy loaded"):
            state.submit_trick("get_up")
        assert state.drain() == []

    def test_roll_while_seated_is_rejected(self):
        state, _ = _pair(sit=True, roll=True)
        state.set_status({"sitting": True, "posture": "sitting"})
        with pytest.raises(ValueError, match="stand up first"):
            state.submit_trick("roll")
        assert state.drain() == []

    def test_roll_while_rising_is_rejected(self):
        state, _ = _pair(sit=True, roll=True)
        state.set_status({"sitting": False, "posture": "rising"})
        with pytest.raises(ValueError, match="stand up first"):
            state.submit_trick("roll")
        assert state.drain() == []

    def test_get_up_while_seated_is_rejected(self):
        state, _ = _pair(sit=True, get_up=True)
        state.set_status({"sitting": True, "posture": "sitting"})
        with pytest.raises(ValueError, match="use stand up"):
            state.submit_trick("get_up")
        assert state.drain() == []

    def test_trick_while_a_trick_runs_is_rejected(self):
        state, _ = _pair(roll=True, get_up=True)
        state.set_status({"trick": "rolling"})
        with pytest.raises(ValueError, match="trick is running"):
            state.submit_trick("get_up")
        assert state.drain() == []

    def test_trick_while_a_trick_is_still_queued_is_rejected(self):
        state, _ = _pair(roll=True, get_up=True)
        state.submit_trick("roll")
        with pytest.raises(ValueError, match="trick is running"):
            state.submit_trick("get_up")
        assert state.drain() == [TrickCmd("roll")]

    def test_walk_while_a_trick_runs_is_rejected(self):
        state, _ = _pair(roll=True)
        state.set_status({"trick": "rolling"})
        with pytest.raises(ValueError, match="trick is running"):
            state.submit_walk(0.2, 0.0, 0.0, 2.0)
        assert state.drain() == []

    def test_sit_while_a_trick_runs_is_rejected(self):
        state, _ = _pair(sit=True, roll=True)
        state.set_status({"trick": "rolling"})
        with pytest.raises(ValueError, match="trick is running"):
            state.submit_posture(True)
        assert state.drain() == []

    def test_walk_is_allowed_again_once_the_trick_is_over(self):
        state, _ = _pair(roll=True)
        state.set_status({"trick": "none"})
        state.submit_walk(0.2, 0.0, 0.0, 2.0)
        assert state.drain() == [WalkCmd(0.2, 0.0, 0.0, 2.0)]

    def test_trick_counts_as_a_brain_request(self):
        state, _ = _pair(roll=True)
        before = state.request_count()
        state.submit_trick("roll")
        assert state.request_count() == before + 1


class TestKickState:
    def test_kick_right_and_left_are_queued(self):
        state, _ = _pair(kick_right=True, kick_left=True)
        assert state.submit_kick("right") == {"trick": "kick_right"}
        assert state.drain() == [TrickCmd("kick_right")]
        assert state.submit_kick("left") == {"trick": "kick_left"}
        assert state.drain() == [TrickCmd("kick_left")]

    def test_kick_defaults_to_the_right_foot(self):
        state, _ = _pair(kick_right=True)
        assert state.submit_kick() == {"trick": "kick_right"}

    def test_unknown_foot_is_rejected(self):
        state, _ = _pair(kick_right=True, kick_left=True)
        with pytest.raises(ValueError, match="unknown foot"):
            state.submit_kick("middle")
        assert state.drain() == []

    def test_kick_without_its_policy_is_rejected(self):
        state, _ = _pair(kick_right=True)
        with pytest.raises(ValueError, match="no kick_left policy loaded, start the runner with --kick-left"):
            state.submit_kick("left")

    def test_kick_while_seated_is_rejected(self):
        state, _ = _pair(sit=True, kick_right=True)
        state.set_status({"sitting": True})
        with pytest.raises(ValueError, match="sitting, stand up first"):
            state.submit_kick("right")

    def test_kick_while_rising_is_rejected(self):
        state, _ = _pair(sit=True, kick_right=True)
        state.set_status({"sitting": False, "posture": "rising"})
        with pytest.raises(ValueError, match="sitting, stand up first"):
            state.submit_kick("right")

    def test_kick_while_a_trick_runs_is_rejected(self):
        state, _ = _pair(kick_right=True)
        state.set_status({"trick": "rolling"})
        with pytest.raises(ValueError, match="trick is running"):
            state.submit_kick("right")

    def test_kick_seconds_is_the_trick_timer(self):
        assert TRICKS["kick_right"].seconds == KICK_SECONDS
        assert TRICKS["kick_left"].seconds == KICK_SECONDS
        assert TRICKS["kick_right"].status == "kicking"


class TestBallState:
    def test_ball_is_queued_for_either_foot(self):
        state, _ = _pair(kick_right=True, kick_left=True)
        assert state.submit_ball("right") == {"ball": "right"}
        assert state.submit_ball("left") == {"ball": "left"}
        assert state.drain() == [BallCmd("right"), BallCmd("left")]

    def test_ball_defaults_to_the_right_foot(self):
        state, _ = _pair(kick_right=True)
        assert state.submit_ball() == {"ball": "right"}

    def test_ball_without_a_kick_policy_for_that_foot_is_rejected(self):
        state, _ = _pair(kick_right=True)
        with pytest.raises(ValueError, match="no kick_left policy loaded"):
            state.submit_ball("left")
        assert state.drain() == []

    def test_ball_with_an_unknown_foot_is_rejected(self):
        state, _ = _pair(kick_right=True)
        with pytest.raises(ValueError, match="unknown foot"):
            state.submit_ball("both")

    def test_ball_during_a_trick_is_rejected(self):
        state, _ = _pair(kick_right=True)
        state.set_status({"trick": "kicking"})
        with pytest.raises(ValueError, match="trick is running"):
            state.submit_ball("right")

    def test_ball_with_a_kick_still_queued_is_rejected(self):
        state, _ = _pair(kick_right=True)
        state.submit_kick("right")
        with pytest.raises(ValueError, match="trick is running"):
            state.submit_ball("right")

    def test_ball_counts_as_a_brain_request(self):
        state, _ = _pair(kick_right=True)
        before = state.request_count()
        state.submit_ball("right")
        assert state.request_count() == before + 1


class TestAvailableActions:
    def test_walking_policy_offers_walk_look_and_both_gestures(self):
        actions = available_actions(FakePolicy())
        assert actions["walk"] is True
        assert actions["look"] is True
        assert actions["nod"] is True
        assert actions["shake"] is True

    def test_untrained_sessions_are_unavailable(self):
        actions = available_actions(FakePolicy())
        assert actions["sit"] is False
        assert actions["stand up"] is False
        assert actions["kick right"] is False
        assert actions["kick left"] is False
        assert actions["roulade"] is False
        assert actions["ground pick"] is False

    def test_kick_actions_follow_their_own_session(self):
        actions = available_actions(FakePolicy(kick_left=True))
        assert actions["kick left"] is True
        assert actions["kick right"] is False

    def test_walk_is_unavailable_without_a_walking_policy(self):
        assert available_actions(FakePolicy(walking=False))["walk"] is False

    def test_gestures_follow_the_keys_the_player_carries(self):
        policy = FakePolicy(gestures={})
        actions = available_actions(policy)
        assert actions["nod"] is False
        assert actions["shake"] is False

    def test_a_loaded_session_makes_its_action_available(self):
        policy = FakePolicy()
        policy.sit_session = object()
        actions = available_actions(policy)
        assert actions["sit"] is True
        assert actions["stand up"] is True


class TestBrainWatchdog:
    def test_fires_once_after_the_timeout_of_silence(self):
        state = BridgeState(FakePolicy())
        watchdog = BrainWatchdog(state, CONTROL_DT)
        ticks = int(round((BRAIN_TIMEOUT_S + 1.0) / CONTROL_DT))
        assert sum(watchdog.tick() for _ in range(ticks)) == 1

    def test_stays_quiet_while_requests_keep_arriving(self):
        state = BridgeState(FakePolicy())
        watchdog = BrainWatchdog(state, CONTROL_DT)

        for _ in range(int(round((BRAIN_TIMEOUT_S + 1.0) / CONTROL_DT))):
            state.get_status()
            assert watchdog.tick() is False

    def test_rearms_after_the_brain_speaks_again(self):
        state = BridgeState(FakePolicy())
        watchdog = BrainWatchdog(state, CONTROL_DT)
        ticks = int(round((BRAIN_TIMEOUT_S + 1.0) / CONTROL_DT))

        assert sum(watchdog.tick() for _ in range(ticks)) == 1
        state.get_status()
        assert sum(watchdog.tick() for _ in range(ticks)) == 1

    def test_timeout_is_configurable(self):
        state = BridgeState(FakePolicy())
        watchdog = BrainWatchdog(state, CONTROL_DT, timeout_s=0.1)
        assert sum(watchdog.tick() for _ in range(int(round(0.1 / CONTROL_DT)))) == 1


class TestSkillsTick:
    def test_walk_applies_and_expires_in_sim_time(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.0, 0.3, 1.0)
        runner.tick()
        assert policy.vel_cmd.tolist() == pytest.approx([0.2, 0.0, 0.3])

        _tick_seconds(runner, 0.9)
        assert policy.vel_cmd[0] == pytest.approx(0.2)

        _tick_seconds(runner, 0.2)
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]

    def test_new_walk_extends_the_countdown(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.0, 0.0, 1.0)
        runner.tick()
        state.submit_walk(0.1, 0.0, 0.0, 5.0)
        _tick_seconds(runner, 2.0)
        assert policy.vel_cmd[0] == pytest.approx(0.1)

    def test_stop_zeroes_twist_head_and_gesture(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.1, 0.0, 5.0)
        state.submit_look(0.3, -0.2)
        state.submit_gesture("nod")
        runner.tick()
        state.submit_stop()
        runner.tick()
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]
        assert policy.head_offset.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert policy.gesture_player.active_name is None

    def test_reset_zeroes_twist_head_and_gesture_like_stop(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.1, 0.0, 5.0)
        state.submit_look(0.3, -0.2)
        state.submit_gesture("nod")
        runner.tick()
        state.submit_reset()
        runner.tick()
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]
        assert policy.head_offset.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert policy.gesture_player.active_name is None

    def test_look_writes_head_pitch_and_yaw(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_look(0.3, -0.4)
        runner.tick()
        assert policy.head_offset.tolist() == pytest.approx([0.0, 0.3, -0.4, 0.0])
        assert policy.command_updates > 0

    def test_gesture_maps_names_to_keys(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_gesture("nod")
        state.submit_gesture("shake")
        runner.tick()
        assert policy.started_gestures == ["n", "m"]

    def test_status_snapshot_content(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.0, 0.0, 2.0)
        runner.tick()
        status = state.get_status()
        assert status["ready"] is True
        assert status["policy"] == "walking"
        assert status["twist"] == pytest.approx([0.2, 0.0, 0.0])
        assert status["walk_seconds_left"] == pytest.approx(2.0 - CONTROL_DT)
        assert status["gesture"] is None
        assert status["fallen"] is False

    def test_status_lists_the_actions_this_policy_can_run(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        runner.tick()
        actions = state.get_status()["actions"]
        assert actions["walk"] is True
        assert actions["roulade"] is False

    def test_status_twist_is_zero_when_the_command_block_drops_it(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.01, 0.0, 0.0, 2.0)  # under the walking switch threshold
        runner.tick()
        status = state.get_status()
        assert status["policy"] == "standing"
        assert status["twist"] == pytest.approx([0.0, 0.0, 0.0])

    def test_status_reports_the_short_gesture_name(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_gesture("nod")
        runner.tick()
        assert state.get_status()["gesture"] == "nod"

    def test_paused_tick_reports_not_ready_and_holds_the_countdown(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.0, 0.0, 0.1)
        runner.tick(policy_enabled=False)
        _tick_seconds(runner, 1.0, policy_enabled=False)
        status = state.get_status()
        assert status["ready"] is False
        assert status["walk_seconds_left"] == pytest.approx(0.1)
        assert policy.vel_cmd[0] == pytest.approx(0.2)

    def test_watchdog_zeroes_twist_and_head_after_silence(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_look(0.5, 0.4)
        runner.tick()
        assert policy.head_offset[1] == pytest.approx(0.5)

        _tick_seconds(runner, BRAIN_TIMEOUT_S + 1.0)
        assert policy.head_offset.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]

    def test_watchdog_leaves_a_running_gesture_alone(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_gesture("nod")
        runner.tick()
        policy.head_offset[1] = 0.3  # a gesture writing its own head pose

        _tick_seconds(runner, BRAIN_TIMEOUT_S + 1.0)
        assert policy.head_offset[1] == pytest.approx(0.3)

    def test_status_polls_keep_the_watchdog_quiet(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_look(0.5, 0.0)
        runner.tick()

        for _ in range(int(round((BRAIN_TIMEOUT_S + 1.0) / CONTROL_DT))):
            runner.tick()
            state.get_status()

        assert policy.head_offset[1] == pytest.approx(0.5)

    def test_watchdog_releases_once_and_lets_a_later_look_stand(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_look(0.5, 0.0)
        runner.tick()
        _tick_seconds(runner, BRAIN_TIMEOUT_S + 1.0)

        state.submit_look(0.2, 0.0)
        runner.tick()
        _tick_seconds(runner, 1.0)
        assert policy.head_offset[1] == pytest.approx(0.2)

    @pytest.mark.parametrize("gravity_z", [0.0, 0.9])
    def test_fallen_is_true_near_or_past_upright(self, gravity_z):
        state, policy = _pair(gravity_z=gravity_z)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        runner.tick()
        assert state.get_status()["fallen"] is True

    def test_fallen_is_false_when_upright(self):
        state, policy = _pair(gravity_z=-1.0)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        runner.tick()
        assert state.get_status()["fallen"] is False

    def test_fallen_zeroes_the_twist_and_cancels_the_walk(self):
        state, policy = _pair(gravity_z=0.0)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.0, 0.0, 5.0)
        runner.tick()
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]
        assert state.get_status()["walk_seconds_left"] == pytest.approx(0.0)


class TestSkillsPosture:
    def test_sit_toggles_the_policy_once(self):
        state, policy = _pair(sit=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_posture(True)
        runner.tick()
        assert policy.sit_mode is True
        assert policy.sit_toggles == 1

    def test_sit_twice_leaves_the_posture_alone(self):
        state, policy = _pair(sit=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_posture(True)
        state.submit_posture(True)
        runner.tick()
        assert policy.sit_toggles == 1

    def test_stand_toggles_back(self):
        state, policy = _pair(sit=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_posture(True)
        runner.tick()
        state.submit_posture(False)
        runner.tick()
        assert policy.sit_mode is False
        assert policy.sit_toggles == 2

    def test_sitting_cancels_a_running_walk(self):
        state, policy = _pair(sit=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.0, 0.0, 5.0)
        runner.tick()
        state.submit_posture(True)
        runner.tick()
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]
        assert state.get_status()["walk_seconds_left"] == 0.0

    def test_status_reports_sitting_and_posture(self):
        state, policy = _pair(sit=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        runner.tick()
        status = state.get_status()
        assert status["sitting"] is False
        assert status["posture"] == "standing"

        state.submit_posture(True)
        runner.tick()
        status = state.get_status()
        assert status["sitting"] is True
        assert status["posture"] == "sitting"

    def test_status_twist_is_zero_while_seated(self):
        state, policy = _pair(sit=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_posture(True)
        runner.tick()

        # The seated command block carries the posture flag in the vx slot, not a speed.
        assert policy.command[0] == pytest.approx(1.0)
        assert state.get_status()["twist"] == [0.0, 0.0, 0.0]

    def test_standing_up_reports_rising_until_the_rise_is_over(self):
        state, policy = _pair(sit=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_posture(True)
        runner.tick()

        state.submit_posture(False)
        runner.tick()
        assert state.get_status()["posture"] == "rising"

        _tick_seconds(runner, RISE_SECONDS)
        assert state.get_status()["posture"] == "standing"

    def test_reset_stands_the_robot_back_up(self):
        state, policy = _pair(sit=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_posture(True)
        runner.tick()

        state.submit_reset()
        runner.tick()
        assert policy.sit_mode is False
        assert state.get_status()["posture"] == "standing"


class TestSkillsTrick:
    def test_roll_starts_the_behavior_and_reports_rolling(self):
        state, policy = _pair(roll=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_trick("roll")
        runner.tick()
        assert policy.behaviors == ["roulade"]
        assert state.get_status()["trick"] == "rolling"

    def test_get_up_starts_the_behavior_and_reports_getting_up(self):
        state, policy = _pair(get_up=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_trick("get_up")
        runner.tick()
        assert policy.behaviors == ["standup"]
        assert state.get_status()["trick"] == "getting_up"

    def test_trick_returns_to_none_when_the_timer_ends(self):
        state, policy = _pair(roll=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_trick("roll")
        runner.tick()

        _tick_seconds(runner, ROLL_SECONDS - 0.1)
        assert state.get_status()["trick"] == "rolling"

        _tick_seconds(runner, 0.2)
        assert state.get_status()["trick"] == "none"
        assert runner.trick_name() == "none"

    def test_get_up_runs_for_its_own_duration(self):
        state, policy = _pair(get_up=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_trick("get_up")
        runner.tick()

        _tick_seconds(runner, ROLL_SECONDS + 0.1)
        assert state.get_status()["trick"] == "getting_up"

        _tick_seconds(runner, GET_UP_SECONDS)
        assert state.get_status()["trick"] == "none"

    def test_rolling_cancels_a_running_walk(self):
        state, policy = _pair(roll=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_walk(0.2, 0.0, 0.0, 5.0)
        runner.tick()
        state.submit_trick("roll")
        runner.tick()
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]
        assert state.get_status()["walk_seconds_left"] == 0.0

    def test_reset_clears_a_running_trick(self):
        state, policy = _pair(roll=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_trick("roll")
        runner.tick()

        state.submit_reset()
        runner.tick()
        assert state.get_status()["trick"] == "none"

    def test_status_reports_no_trick_by_default(self):
        state, policy = _pair()
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        runner.tick()
        assert state.get_status()["trick"] == "none"

    def test_paused_tick_holds_the_trick_countdown(self):
        state, policy = _pair(roll=True)
        runner = skills.SkillRunner(policy, state, CONTROL_DT)
        state.submit_trick("roll")
        runner.tick(policy_enabled=False)
        _tick_seconds(runner, ROLL_SECONDS + 1.0, policy_enabled=False)
        assert state.get_status()["trick"] == "rolling"


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def _raw_request(url, raw: bytes) -> bytes:
    """Send handwritten request bytes so bad headers reach the server verbatim."""
    host, port = urllib.parse.urlsplit(url).netloc.split(":")

    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(raw)

        return sock.recv(4096)


@contextlib.contextmanager
def _served(policy):
    """Run the bridge on a free port for the length of the block."""
    state = BridgeState(policy)
    server = start_bridge(state, port=0)  # port 0: OS picks a free port

    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}"

    finally:
        server.shutdown()


class TestBridgeServer:
    @pytest.fixture()
    def served_state(self):
        with _served(FakePolicy()) as pair:
            yield pair

    def test_walk_roundtrip(self, served_state):
        state, url = served_state
        status, body = _post(f"{url}/walk", {"vx": 0.2, "seconds": 2})
        assert status == 200
        assert body["vx"] == pytest.approx(0.2)
        assert state.drain() == [WalkCmd(0.2, 0.0, 0.0, 2.0)]

    def test_walk_without_a_walking_policy_returns_400(self):
        with _served(FakePolicy(walking=False)) as (state, url):
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                _post(f"{url}/walk", {"vx": 0.2})
            assert excinfo.value.code == 400
            assert "error" in json.loads(excinfo.value.read())
            assert state.drain() == []

    def test_stop_roundtrip(self, served_state):
        state, url = served_state
        status, body = _post(f"{url}/stop", {})
        assert status == 200
        assert body == {"stopped": True}
        assert state.drain() == [StopCmd()]

    def test_reset_roundtrip(self, served_state):
        state, url = served_state
        status, body = _post(f"{url}/reset", {})
        assert status == 200
        assert body == {"reset": True}
        assert state.drain() == [ResetCmd()]

    def test_sit_and_stand_roundtrip(self):
        with _served(FakePolicy(sit=True)) as (state, url):
            assert _post(f"{url}/sit", {}) == (200, {"sit": True})
            assert _post(f"{url}/stand", {}) == (200, {"sit": False})
            assert state.drain() == [PostureCmd(True), PostureCmd(False)]

    def test_roll_and_get_up_roundtrip(self):
        with _served(FakePolicy(roll=True, get_up=True)) as (state, url):
            assert _post(f"{url}/roll", {}) == (200, {"trick": "roll"})
            assert state.drain() == [TrickCmd("roll")]
            assert _post(f"{url}/get_up", {}) == (200, {"trick": "get_up"})
            assert state.drain() == [TrickCmd("get_up")]

    def test_roll_without_a_roll_policy_returns_400(self, served_state):
        state, url = served_state
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"{url}/roll", {})
        assert excinfo.value.code == 400
        assert "error" in json.loads(excinfo.value.read())
        assert state.drain() == []

    def test_sit_without_a_sit_policy_returns_400(self, served_state):
        state, url = served_state
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"{url}/sit", {})
        assert excinfo.value.code == 400
        assert state.drain() == []

    def test_walk_while_seated_returns_400(self):
        with _served(FakePolicy(sit=True)) as (state, url):
            state.set_status({"sitting": True, "posture": "sitting"})
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                _post(f"{url}/walk", {"vx": 0.2})
            assert excinfo.value.code == 400
            assert state.drain() == []

    def test_look_roundtrip(self, served_state):
        state, url = served_state
        status, body = _post(f"{url}/look", {"pitch": 0.3, "yaw": -0.2})
        assert status == 200
        assert body == {"pitch": pytest.approx(0.3), "yaw": pytest.approx(-0.2), "clamped": False}
        assert state.drain() == [LookCmd(0.3, -0.2)]

    def test_status_roundtrip(self, served_state):
        state, url = served_state
        state.set_status({"policy": "standing", "fallen": False})
        with urllib.request.urlopen(f"{url}/status", timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read()) == {"policy": "standing", "fallen": False}

    def test_status_counts_as_a_brain_request_and_a_peek_does_not(self, served_state):
        state, url = served_state
        state.set_status({"policy": "walking"})

        with urllib.request.urlopen(f"{url}/status?peek=1", timeout=5) as resp:
            assert json.loads(resp.read()) == {"policy": "walking"}

        before = state.request_count()

        with urllib.request.urlopen(f"{url}/status?peek=1", timeout=5):
            pass

        assert state.request_count() == before

        with urllib.request.urlopen(f"{url}/status", timeout=5):
            pass

        assert state.request_count() == before + 1

    def test_unknown_gesture_returns_400(self, served_state):
        state, url = served_state
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"{url}/gesture", {"name": "backflip"})
        assert excinfo.value.code == 400
        assert "error" in json.loads(excinfo.value.read())
        assert state.drain() == []

    def test_bad_json_returns_400(self, served_state):
        state, url = served_state
        req = urllib.request.Request(
            f"{url}/walk", data=b"not json", method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=5)
        assert excinfo.value.code == 400

    def test_bad_content_length_returns_400(self, served_state):
        state, url = served_state
        raw = b"POST /walk HTTP/1.1\r\nHost: localhost\r\nContent-Length: abc\r\n\r\n"
        response = _raw_request(url, raw)
        assert b"400" in response.split(b"\r\n", 1)[0]
        assert state.drain() == []

    def test_unknown_route_returns_404(self, served_state):
        state, url = served_state
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"{url}/dance", {})
        assert excinfo.value.code == 404

    def test_non_object_json_body_returns_400(self, served_state):
        state, url = served_state
        req = urllib.request.Request(
            f"{url}/walk", data=b"5", method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=5)
        assert excinfo.value.code == 400
        assert "error" in json.loads(excinfo.value.read())
        assert state.drain() == []

    def test_kick_roundtrip_defaults_to_the_right_foot(self):
        with _served(FakePolicy(kick_right=True, kick_left=True)) as (state, url):
            assert _post(f"{url}/kick", {}) == (200, {"trick": "kick_right"})
            assert state.drain() == [TrickCmd("kick_right")]
            assert _post(f"{url}/kick", {"foot": "left"}) == (200, {"trick": "kick_left"})
            assert state.drain() == [TrickCmd("kick_left")]

    def test_ball_roundtrip(self):
        with _served(FakePolicy(kick_right=True, kick_left=True)) as (state, url):
            assert _post(f"{url}/ball", {}) == (200, {"ball": "right"})
            assert _post(f"{url}/ball", {"foot": "left"}) == (200, {"ball": "left"})
            assert state.drain() == [BallCmd("right"), BallCmd("left")]

    def test_kick_without_its_policy_returns_400(self, served_state):
        state, url = served_state
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"{url}/kick", {"foot": "left"})
        assert excinfo.value.code == 400
        assert "no kick_left policy loaded" in json.loads(excinfo.value.read())["error"]
        assert state.drain() == []

    def test_ball_without_a_kick_policy_returns_400(self, served_state):
        state, url = served_state
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"{url}/ball", {})
        assert excinfo.value.code == 400
        assert state.drain() == []

    def test_kick_with_an_unknown_foot_returns_400(self):
        with _served(FakePolicy(kick_right=True)) as (state, url):
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                _post(f"{url}/kick", {"foot": "middle"})
            assert excinfo.value.code == 400
            assert state.drain() == []
