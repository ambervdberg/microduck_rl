"""The follower turns ball sightings into the next head pose, and sweeps when the ball is lost."""
import math
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from bridge.ball_finder import BallSighting
from bridge.follower import (
    DEAD_BAND,
    FACE_GAIN,
    FACE_START,
    FACE_STOP,
    GAIN,
    HUNT_RATE,
    LOST_AFTER_S,
    SEARCH_PITCH,
    SEARCH_RATE,
    SEARCH_YAW_MAX,
    BallFollower,
    BallHunt,
    BodyTurner,
)
from bridge.state import HEAD_PITCH_MAX, HEAD_YAW_MAX

DT = 0.1
LOST_UPDATES = int(round(LOST_AFTER_S / DT))


def _seen(x=0.0, y=0.0, size=40):
    return BallSighting(x, y, size)


def _unseen_for(follower, updates, pitch=0.0, yaw=0.0):
    for _ in range(updates):
        pitch, yaw = follower.update(None, pitch, yaw)
    return pitch, yaw


def test_a_ball_on_the_right_turns_the_head_right():
    pitch, yaw = BallFollower(DT).update(_seen(x=0.5), 0.0, 0.0)
    assert yaw == pytest.approx(-GAIN * 0.5)
    assert pitch == 0.0


def test_a_ball_on_the_left_turns_the_head_left():
    _pitch, yaw = BallFollower(DT).update(_seen(x=-0.5), 0.0, 0.0)
    assert yaw == pytest.approx(GAIN * 0.5)


def test_a_ball_low_in_the_picture_tilts_the_head_down():
    pitch, yaw = BallFollower(DT).update(_seen(y=0.5), 0.0, 0.0)
    assert pitch == pytest.approx(GAIN * 0.5)
    assert yaw == 0.0


def test_a_centred_ball_holds_the_head():
    pitch, yaw = BallFollower(DT).update(_seen(x=DEAD_BAND / 2, y=-DEAD_BAND / 2), 0.2, -0.3)
    assert (pitch, yaw) == (0.2, -0.3)


def test_the_head_never_passes_the_trained_caps():
    follower = BallFollower(DT)
    pitch, yaw = 0.0, 0.0
    for _ in range(100):
        pitch, yaw = follower.update(_seen(x=-1.0, y=1.0), pitch, yaw)
    assert yaw == HEAD_YAW_MAX
    assert pitch == HEAD_PITCH_MAX


def test_tighter_caps_are_respected():
    follower = BallFollower(DT, pitch_max=0.2, yaw_max=0.3)
    pitch, yaw = 0.0, 0.0
    for _ in range(20):
        pitch, yaw = follower.update(_seen(x=1.0, y=1.0), pitch, yaw)
    assert yaw == -0.3
    assert pitch == 0.2


def test_a_short_loss_keeps_the_pose():
    follower = BallFollower(DT)
    pitch, yaw = _unseen_for(follower, LOST_UPDATES - 1, pitch=0.2, yaw=0.4)
    assert (pitch, yaw) == (0.2, 0.4)
    assert follower.searching is False


def test_a_full_second_without_a_ball_starts_the_search():
    follower = BallFollower(DT)
    pitch, yaw = _unseen_for(follower, LOST_UPDATES, pitch=0.2, yaw=0.4)
    assert follower.searching is True
    assert pitch == SEARCH_PITCH
    assert yaw == pytest.approx(0.4 + SEARCH_RATE * DT)


def test_the_sweep_turns_around_at_its_limit():
    follower = BallFollower(DT)
    _unseen_for(follower, LOST_UPDATES - 1)
    pitch, yaw = follower.update(None, 0.0, SEARCH_YAW_MAX - 0.05)
    assert yaw == SEARCH_YAW_MAX
    pitch, yaw = follower.update(None, pitch, yaw)
    assert yaw == pytest.approx(SEARCH_YAW_MAX - SEARCH_RATE * DT)


def test_the_sweep_covers_both_sides():
    follower = BallFollower(DT)
    _unseen_for(follower, LOST_UPDATES)
    yaws = []
    pitch, yaw = 0.0, 0.0
    for _ in range(int(4 * SEARCH_YAW_MAX / (SEARCH_RATE * DT))):
        pitch, yaw = follower.update(None, pitch, yaw)
        yaws.append(yaw)
    assert min(yaws) == pytest.approx(-SEARCH_YAW_MAX, abs=1e-6)
    assert max(yaws) == pytest.approx(SEARCH_YAW_MAX, abs=1e-6)


def test_a_sighting_ends_the_search():
    follower = BallFollower(DT)
    pitch, yaw = _unseen_for(follower, LOST_UPDATES + 3)
    assert follower.searching is True
    pitch, yaw = follower.update(_seen(x=0.5), pitch, yaw)
    assert follower.searching is False
    assert follower.sighting == _seen(x=0.5)


def test_the_last_sighting_is_kept_and_cleared():
    follower = BallFollower(DT)
    follower.update(_seen(x=0.1), 0.0, 0.0)
    assert follower.sighting == _seen(x=0.1)
    follower.update(None, 0.0, 0.0)
    assert follower.sighting is None


TURN_MAX = 1.0


def test_a_small_head_yaw_asks_for_no_turn():
    turner = BodyTurner(TURN_MAX)
    assert turner.update(FACE_STOP / 2, searching=False) == 0.0
    assert turner.turning is False


def test_a_head_far_to_the_left_turns_the_body_left():
    turner = BodyTurner(TURN_MAX)
    assert turner.update(0.5, searching=False) == pytest.approx(FACE_GAIN * 0.5)
    assert turner.turning is True


def test_a_head_far_to_the_right_turns_the_body_right():
    turner = BodyTurner(TURN_MAX)
    assert turner.update(-0.5, searching=False) == pytest.approx(-FACE_GAIN * 0.5)


def test_the_turn_stays_under_the_cap():
    turner = BodyTurner(TURN_MAX)
    assert turner.update(1.4, searching=False) == TURN_MAX
    assert turner.update(-1.4, searching=False) == -TURN_MAX


def test_the_turn_keeps_going_between_start_and_stop():
    turner = BodyTurner(TURN_MAX)
    turner.update(FACE_START + 0.05, searching=False)
    assert turner.update(0.2, searching=False) == pytest.approx(FACE_GAIN * 0.2)
    assert turner.turning is True


def test_the_turn_ends_under_the_stop_yaw():
    turner = BodyTurner(TURN_MAX)
    turner.update(0.5, searching=False)
    assert turner.update(FACE_STOP - 0.01, searching=False) == 0.0
    assert turner.turning is False


def test_a_turn_does_not_start_between_stop_and_start():
    turner = BodyTurner(TURN_MAX)
    assert turner.update(0.2, searching=False) == 0.0
    assert turner.turning is False


def test_searching_holds_the_body_still():
    turner = BodyTurner(TURN_MAX)
    turner.update(0.5, searching=False)
    assert turner.update(1.0, searching=True) == 0.0
    assert turner.turning is False


def test_the_hunt_turns_the_way_the_head_looked():
    assert BallHunt(direction=-0.4).update(0.0) == -HUNT_RATE
    assert BallHunt(direction=0.4).update(0.0) == HUNT_RATE


def test_the_hunt_turns_left_when_the_head_was_straight():
    assert BallHunt(direction=0.0).update(0.0) == HUNT_RATE


def test_the_hunt_keeps_going_before_a_full_turn():
    hunt = BallHunt(direction=1.0)
    for yaw in (0.0, 1.5, 3.0):
        assert hunt.update(yaw) == HUNT_RATE
    assert hunt.gave_up is False


def test_the_hunt_gives_up_after_a_full_turn():
    hunt = BallHunt(direction=1.0)
    for step in range(8):
        hunt.update(step * 1.0)
    assert hunt.gave_up is True
    assert hunt.update(8.0) == 0.0


def test_the_turn_count_crosses_the_pi_seam():
    hunt = BallHunt(direction=1.0)
    hunt.update(3.0)
    hunt.update(-3.0)
    assert hunt.turned == pytest.approx(2 * math.pi - 6.0)


def test_turning_the_wrong_way_counts_down():
    hunt = BallHunt(direction=1.0)
    hunt.update(0.0)
    hunt.update(-1.0)
    assert hunt.turned == pytest.approx(-1.0)
    assert hunt.gave_up is False
