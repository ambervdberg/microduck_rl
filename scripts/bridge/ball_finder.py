"""Finds the orange ball in one camera picture.

A picture is an array of height by width by 3 (red, green, blue, 0 to 255).
The answer is where the orange pixels sit, as a fraction of the picture from
its middle, plus how many there are. Pure numpy, no sim.
"""

from dataclasses import dataclass

import numpy as np

# Orange is red 255, green 140, blue 0 in the sim. Light shades it, so the rule is ratios.
RED_MIN = 120              # darker than this is shadow, not ball
BLUE_MAX_OF_RED = 0.5      # blue at most half of red
GREEN_MIN_OF_RED = 1 / 3   # green between a third and three quarters of red
GREEN_MAX_OF_RED = 3 / 4
MIN_PIXELS = 4             # fewer orange pixels than this is noise


@dataclass(frozen=True)
class BallSighting:
    """Where the ball is in the picture and how big it looks."""

    x: float   # -1 left edge, 0 middle, 1 right edge
    y: float   # -1 top edge, 0 middle, 1 bottom edge
    size: int  # orange pixel count


def orange_mask(rgb: np.ndarray) -> np.ndarray:
    """True where a pixel has the ball's colour."""
    red = rgb[..., 0].astype(np.int32)
    green = rgb[..., 1].astype(np.int32)
    blue = rgb[..., 2].astype(np.int32)

    bright = red >= RED_MIN
    little_blue = blue <= BLUE_MAX_OF_RED * red
    some_green = (green >= GREEN_MIN_OF_RED * red) & (green <= GREEN_MAX_OF_RED * red)

    return bright & little_blue & some_green


def find_ball(rgb: np.ndarray) -> BallSighting | None:
    """Where the ball is in the picture, or None when there is no ball."""
    mask = orange_mask(rgb)
    rows, cols = np.nonzero(mask)

    if len(cols) < MIN_PIXELS:
        return None

    height, width = mask.shape
    x = 2.0 * (cols.mean() + 0.5) / width - 1.0
    y = 2.0 * (rows.mean() + 0.5) / height - 1.0

    return BallSighting(float(x), float(y), int(len(cols)))
