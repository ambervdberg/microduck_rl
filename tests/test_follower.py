"""The follower turns ball sightings into the next head pose, and sweeps when the ball is lost."""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from bridge.ball_finder import BallSighting
from bridge.follower import (
    DEAD_BAND,
    GAIN,
    LOST_AFTER_S,
    SEARCH_PITCH,
    SEARCH_RATE,
    SEARCH_YAW_MAX,
    BallFollower,
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
