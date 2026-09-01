"""Tests for the LLM bridge: command state, skills, HTTP server."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bridge.state import (  # noqa: E402
    HEAD_MAX,
    VX_MAX,
    WALK_DEFAULT_S,
    WALK_MAX_S,
    BridgeState,
    GestureCmd,
    LookCmd,
    StopCmd,
    WalkCmd,
)


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
        assert echo["pitch"] == pytest.approx(-HEAD_MAX)
        assert echo["yaw"] == pytest.approx(0.5)
        assert state.drain() == [LookCmd(-HEAD_MAX, 0.5)]

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
