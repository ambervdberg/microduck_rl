"""Tests for the LLM bridge: command state, skills, HTTP server."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bridge.state import (  # noqa: E402
    HEAD_PITCH_MAX,
    HEAD_YAW_MAX,
    VX_MAX,
    WALK_DEFAULT_S,
    WALK_MAX_S,
    BridgeState,
    GestureCmd,
    LookCmd,
    StopCmd,
    WalkCmd,
)

from bridge import skills  # noqa: E402
from bridge.server import start_bridge  # noqa: E402


class TestBridgeState:
    def test_walk_is_queued_and_drained_in_order(self):
        state = BridgeState()
        state.submit_walk(0.2, 0.0, 0.0, 2.0)
        state.submit_stop()
        drained = state.drain()
        assert drained == [WalkCmd(0.2, 0.0, 0.0, 2.0), StopCmd()]
        assert state.drain() == []

    def test_walk_speeds_are_clamped_and_reported(self):
        state = BridgeState()
        echo = state.submit_walk(9.0, 0.0, 0.0, 2.0)
        assert echo["vx"] == pytest.approx(VX_MAX)
        assert echo["clamped"] is True
        assert state.drain() == [WalkCmd(VX_MAX, 0.0, 0.0, 2.0)]

    def test_walk_duration_default_and_max(self):
        state = BridgeState()
        assert state.submit_walk(0.1, 0.0, 0.0, None)["seconds"] == WALK_DEFAULT_S
        assert state.submit_walk(0.1, 0.0, 0.0, 60.0)["seconds"] == WALK_MAX_S

    def test_look_clamps_head_angles(self):
        state = BridgeState()
        echo = state.submit_look(-9.0, 0.5)
        assert echo["pitch"] == pytest.approx(-HEAD_PITCH_MAX)
        assert echo["yaw"] == pytest.approx(0.5)
        assert state.drain() == [LookCmd(-HEAD_PITCH_MAX, 0.5)]

    def test_look_clamps_yaw_to_its_own_max(self):
        state = BridgeState()
        echo = state.submit_look(0.0, 9.0)
        assert echo["yaw"] == pytest.approx(HEAD_YAW_MAX)
        assert state.drain() == [LookCmd(0.0, HEAD_YAW_MAX)]

    def test_unknown_gesture_is_rejected_and_not_queued(self):
        state = BridgeState()
        with pytest.raises(ValueError):
            state.submit_gesture("backflip")
        assert state.drain() == []
        state.submit_gesture("nod")
        assert state.drain() == [GestureCmd("nod")]

    def test_status_roundtrip_returns_a_copy(self):
        state = BridgeState()
        state.set_status({"policy": "walking"})
        status = state.get_status()
        status["policy"] = "mutated"
        assert state.get_status() == {"policy": "walking"}

    def test_walk_seconds_clamping_sets_clamped_flag(self):
        state = BridgeState()
        echo = state.submit_walk(0.1, 0.0, 0.0, 60.0)
        assert echo["seconds"] == pytest.approx(WALK_MAX_S)
        assert echo["clamped"] is True

    def test_walk_seconds_none_default_not_clamped(self):
        state = BridgeState()
        echo = state.submit_walk(0.1, 0.0, 0.0, None)
        assert echo["seconds"] == pytest.approx(WALK_DEFAULT_S)
        assert echo["clamped"] is False


class FakeGesturePlayer:
    def __init__(self):
        self.active_name = None

    def cancel(self):
        self.active_name = None


class FakePolicy:
    """Mimics the PolicyInference surface the bridge touches."""

    def __init__(self):
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.head_offset = np.zeros(4, dtype=np.float32)
        self.current_policy = "standing"
        self.gesture_player = FakeGesturePlayer()
        self.started_gestures = []
        self.command_updates = 0

    def set_vel_cmd(self, vx, vy, wz):
        self.vel_cmd = np.array([vx, vy, wz], dtype=np.float32)
        self.current_policy = "walking" if np.linalg.norm(self.vel_cmd) > 0.05 else "standing"

    def _update_command(self):
        self.command_updates += 1

    def start_gesture(self, key):
        self.started_gestures.append(key)
        self.gesture_player.active_name = {"n": "nod yes", "m": "shake no"}[key]
        return object()

    def get_projected_gravity(self):
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)


class TestSkillsTick:
    def test_walk_applies_and_expires_to_zero(self):
        state, policy = BridgeState(), FakePolicy()
        runner = skills.SkillRunner(policy, state)
        state.submit_walk(0.2, 0.0, 0.3, 2.0)
        runner.tick(now=100.0)
        assert policy.vel_cmd.tolist() == pytest.approx([0.2, 0.0, 0.3])

        runner.tick(now=101.9)  # still walking
        assert policy.vel_cmd[0] == pytest.approx(0.2)

        runner.tick(now=102.1)  # deadline passed
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]

    def test_new_walk_extends_deadline(self):
        state, policy = BridgeState(), FakePolicy()
        runner = skills.SkillRunner(policy, state)
        state.submit_walk(0.2, 0.0, 0.0, 2.0)
        runner.tick(now=100.0)
        state.submit_walk(0.1, 0.0, 0.0, 5.0)
        runner.tick(now=101.0)
        runner.tick(now=103.0)  # old deadline passed, new one not
        assert policy.vel_cmd[0] == pytest.approx(0.1)

    def test_stop_zeroes_twist_head_and_gesture(self):
        state, policy = BridgeState(), FakePolicy()
        runner = skills.SkillRunner(policy, state)
        state.submit_walk(0.2, 0.1, 0.0, 5.0)
        state.submit_look(0.3, -0.2)
        state.submit_gesture("nod")
        runner.tick(now=100.0)
        state.submit_stop()
        runner.tick(now=100.1)
        assert policy.vel_cmd.tolist() == [0.0, 0.0, 0.0]
        assert policy.head_offset.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert policy.gesture_player.active_name is None

    def test_look_writes_head_pitch_and_yaw(self):
        state, policy = BridgeState(), FakePolicy()
        runner = skills.SkillRunner(policy, state)
        state.submit_look(0.3, -0.4)
        runner.tick(now=100.0)
        assert policy.head_offset.tolist() == pytest.approx([0.0, 0.3, -0.4, 0.0])
        assert policy.command_updates > 0

    def test_gesture_maps_names_to_keys(self):
        state, policy = BridgeState(), FakePolicy()
        runner = skills.SkillRunner(policy, state)
        state.submit_gesture("nod")
        state.submit_gesture("shake")
        runner.tick(now=100.0)
        assert policy.started_gestures == ["n", "m"]

    def test_status_snapshot_content(self):
        state, policy = BridgeState(), FakePolicy()
        runner = skills.SkillRunner(policy, state)
        state.submit_walk(0.2, 0.0, 0.0, 2.0)
        runner.tick(now=100.0)
        status = state.get_status()
        assert status["policy"] == "walking"
        assert status["twist"] == pytest.approx([0.2, 0.0, 0.0])
        assert status["walk_seconds_left"] == pytest.approx(2.0)
        assert status["gesture"] is None
        assert status["fallen"] is False


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


class TestBridgeServer:
    @pytest.fixture()
    def served_state(self):
        state = BridgeState()
        server = start_bridge(state, port=0)  # port 0: OS picks a free port
        url = f"http://127.0.0.1:{server.server_address[1]}"
        yield state, url
        server.shutdown()

    def test_walk_roundtrip(self, served_state):
        state, url = served_state
        status, body = _post(f"{url}/walk", {"vx": 0.2, "seconds": 2})
        assert status == 200
        assert body["vx"] == pytest.approx(0.2)
        assert state.drain() == [WalkCmd(0.2, 0.0, 0.0, 2.0)]

    def test_status_roundtrip(self, served_state):
        state, url = served_state
        state.set_status({"policy": "standing", "fallen": False})
        with urllib.request.urlopen(f"{url}/status", timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read()) == {"policy": "standing", "fallen": False}

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
