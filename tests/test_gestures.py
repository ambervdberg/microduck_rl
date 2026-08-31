"""Locks in the two properties that make a scripted head gesture safe to send
to a policy trained on head pose commands:

1. It starts and ends at exactly zero offset, so the head returns to whatever
   pose the operator had set and the servos get no step command at either end.
2. It never leaves the amplitude it declares, so the command stays inside the
   range the velocity env trains on (±1.4 rad).

Both are easy to break with an innocent-looking change to the windowing math,
and neither shows up as an error at runtime -- only as a twitchy head.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gestures import (  # noqa: E402
    HEAD_PITCH,
    HEAD_POSE_DIM,
    HEAD_YAW,
    NOD_YES,
    SHAKE_NO,
    GestureCfg,
    GesturePlayer,
    default_gestures,
)

CONTROL_DT = 1.0 / 50.0  # the runtime's 50 Hz control loop


def _play_to_completion(player, dt=CONTROL_DT, max_steps=10_000):
    """Return every offset the player emits until it goes idle."""
    offsets = []
    for _ in range(max_steps):
        offset = player.advance(dt)
        if offset is None:
            break
        offsets.append(offset)
    else:
        pytest.fail("gesture never finished")
    return offsets


class TestGestureCfg:
    class TestOffsetAt:
        def test_should_be_zero_at_the_start(self):
            assert NOD_YES.offset_at(0.0) == pytest.approx(np.zeros(HEAD_POSE_DIM))

        def test_should_be_zero_at_the_end(self):
            assert NOD_YES.offset_at(NOD_YES.duration_s) == pytest.approx(
                np.zeros(HEAD_POSE_DIM), abs=1e-6
            )

        def test_should_move_only_its_own_axis(self):
            offset = SHAKE_NO.offset_at(SHAKE_NO.duration_s / 2)
            others = [i for i in range(HEAD_POSE_DIM) if i != SHAKE_NO.axis]
            assert offset[SHAKE_NO.axis] != 0.0
            assert offset[others] == pytest.approx(np.zeros(len(others)))

        def test_should_stay_within_its_amplitude(self):
            cfg = GestureCfg(name="probe", axis=HEAD_PITCH, amplitude_rad=0.4,
                             period_s=0.5, cycles=3)
            steps = int(cfg.duration_s / CONTROL_DT) + 1
            peak = max(abs(cfg.offset_at(i * CONTROL_DT)[cfg.axis]) for i in range(steps))
            assert peak <= cfg.amplitude_rad + 1e-6

        def test_should_reach_close_to_its_amplitude(self):
            # A window that damped the whole gesture would pass the bound above
            # while producing a gesture nobody can see.
            steps = int(NOD_YES.duration_s / CONTROL_DT) + 1
            peak = max(abs(NOD_YES.offset_at(i * CONTROL_DT)[NOD_YES.axis])
                       for i in range(steps))
            assert peak > 0.8 * NOD_YES.amplitude_rad


class TestGesturePlayer:
    class TestTrigger:
        def test_should_return_none_when_the_key_is_unbound(self):
            player = GesturePlayer(default_gestures())
            assert player.trigger("!") is None
            assert not player.is_playing

        def test_should_start_playing_when_the_key_is_bound(self):
            player = GesturePlayer(default_gestures())
            assert player.trigger("n") is NOD_YES
            assert player.is_playing
            assert player.active_name == "nod yes"

        def test_should_restart_a_gesture_that_is_already_playing(self):
            player = GesturePlayer(default_gestures())
            player.trigger("n")
            player.advance(NOD_YES.duration_s * 0.9)
            player.trigger("n")
            offsets = _play_to_completion(player)
            assert len(offsets) == pytest.approx(NOD_YES.duration_s / CONTROL_DT, rel=0.05)

    class TestAdvance:
        def test_should_return_none_while_idle(self):
            player = GesturePlayer(default_gestures())
            assert player.advance(CONTROL_DT) is None

        def test_should_end_on_a_zero_offset(self):
            player = GesturePlayer(default_gestures())
            player.trigger("m")
            offsets = _play_to_completion(player)
            assert offsets[-1] == pytest.approx(np.zeros(HEAD_POSE_DIM))

        def test_should_go_idle_after_its_duration(self):
            player = GesturePlayer(default_gestures())
            player.trigger("m")
            _play_to_completion(player)
            assert not player.is_playing
            assert player.advance(CONTROL_DT) is None

        def test_should_survive_an_irregular_frame_time(self):
            # actual_dt in infer_policy.py is wall clock, not a fixed step.
            player = GesturePlayer(default_gestures())
            player.trigger("n")
            offsets = _play_to_completion(player, dt=CONTROL_DT * 3.7)
            assert offsets[-1] == pytest.approx(np.zeros(HEAD_POSE_DIM))

    class TestCancel:
        def test_should_stop_a_playing_gesture(self):
            player = GesturePlayer(default_gestures())
            player.trigger("n")
            player.cancel()
            assert not player.is_playing
            assert player.advance(CONTROL_DT) is None


class TestDefaultGestures:
    def test_should_bind_nod_and_shake_to_distinct_axes(self):
        gestures = default_gestures()
        assert gestures["n"].axis == HEAD_PITCH
        assert gestures["m"].axis == HEAD_YAW

    def test_should_stay_inside_the_trained_head_command_range(self):
        trained_max_rad = 1.4
        for cfg in default_gestures().values():
            assert cfg.amplitude_rad < trained_max_rad

    def test_should_not_collide_with_existing_infer_policy_keys(self):
        taken = set("tgklryhbpaezs ")
        assert taken.isdisjoint(default_gestures().keys())
