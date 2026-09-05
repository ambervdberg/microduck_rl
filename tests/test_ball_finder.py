"""The ball finder turns one camera picture into where the orange ball is."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from bridge.ball_finder import MIN_PIXELS, BallSighting, find_ball, orange_mask

WIDTH = 160
HEIGHT = 120
ORANGE = (255, 140, 0)


def _picture():
    """A black picture the size of the head camera."""
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def _dot(picture, col, row, radius=3, colour=ORANGE):
    """Paint a square dot of one colour, centred on a pixel."""
    picture[row - radius:row + radius, col - radius:col + radius] = colour
    return picture


def test_a_black_picture_has_no_ball():
    assert find_ball(_picture()) is None


def test_a_dot_left_of_the_middle_reads_as_negative_x():
    sighting = find_ball(_dot(_picture(), 40, 60))
    assert sighting.x < -0.4
    assert sighting.x > -0.6
    assert sighting.y == pytest.approx(0.0, abs=0.02)
    assert sighting.size == 36


def test_a_dot_at_the_right_edge_reads_as_x_near_one():
    sighting = find_ball(_dot(_picture(), 157, 60))
    assert sighting.x > 0.9


def test_a_dot_low_in_the_picture_reads_as_positive_y():
    sighting = find_ball(_dot(_picture(), 80, 100))
    assert sighting.y > 0.6
    assert sighting.x == pytest.approx(0.0, abs=0.02)


def test_two_dots_read_as_the_point_between_them():
    picture = _dot(_dot(_picture(), 40, 60), 120, 60)
    sighting = find_ball(picture)
    assert sighting.x == pytest.approx(0.0, abs=0.02)
    assert sighting.size == 72


def test_a_dark_orange_dot_still_counts():
    sighting = find_ball(_dot(_picture(), 80, 60, colour=(140, 70, 10)))
    assert sighting is not None


def test_a_red_dot_does_not_count():
    assert find_ball(_dot(_picture(), 80, 60, colour=(255, 0, 0))) is None


def test_the_floor_and_its_white_dots_do_not_count():
    picture = _picture()
    picture[:] = (36, 51, 80)
    picture = _dot(picture, 80, 60, colour=(255, 255, 255))
    assert find_ball(picture) is None


def test_fewer_than_min_pixels_is_noise():
    picture = _picture()
    picture[60, 80:80 + MIN_PIXELS - 1] = ORANGE
    assert find_ball(picture) is None
    picture[60, 80:80 + MIN_PIXELS] = ORANGE
    assert find_ball(picture) is not None


def test_the_mask_is_true_only_on_the_dot():
    mask = orange_mask(_dot(_picture(), 80, 60))
    assert mask.shape == (HEIGHT, WIDTH)
    assert mask.sum() == 36


def test_a_sighting_is_a_plain_value():
    assert BallSighting(0.1, -0.2, 9) == BallSighting(0.1, -0.2, 9)
