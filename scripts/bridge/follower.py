"""Turns ball sightings into the next head pose.

Two states. Following: each picture nudges the head toward the ball and holds
it at the middle. Searching, after LOST_AFTER_S without a ball: the head looks
a bit down and sweeps left and right until the ball shows up again. No sim here.
Head signs: positive pitch looks down, positive yaw looks left.
"""

from bridge.ball_finder import BallSighting
from bridge.state import HEAD_PITCH_MAX, HEAD_YAW_MAX

GAIN = 0.15           # rad of head turn per unit of picture offset, per picture
DEAD_BAND = 0.05      # an offset under this holds the head, no jitter at the middle
LOST_AFTER_S = 1.0    # this long without a ball starts the search
SEARCH_PITCH = 0.3    # rad down while searching, where a ball on the floor would be
SEARCH_YAW_MAX = 1.2  # rad, the sweep turns around here
SEARCH_RATE = 1.0     # rad/s of sweep


class BallFollower:
    """Next head pose from the latest sighting. One update per picture."""

    def __init__(self, dt: float, pitch_max: float = HEAD_PITCH_MAX, yaw_max: float = HEAD_YAW_MAX):
        self._dt = float(dt)
        self._pitch_max = float(pitch_max)
        self._yaw_max = float(yaw_max)
        self._lost_after = int(round(LOST_AFTER_S / self._dt))
        self._unseen_updates = 0
        self._sweep_dir = 1.0
        self.sighting: BallSighting | None = None

    @property
    def searching(self) -> bool:
        """True once the ball has been out of view for LOST_AFTER_S."""
        return self._unseen_updates >= self._lost_after

    def update(self, sighting: BallSighting | None, pitch: float, yaw: float) -> tuple[float, float]:
        """The head pose to hold until the next picture."""
        self.sighting = sighting

        if sighting is None:
            return self._unseen(pitch, yaw)

        self._unseen_updates = 0

        return self._follow(sighting, pitch, yaw)

    def _follow(self, sighting: BallSighting, pitch: float, yaw: float) -> tuple[float, float]:
        """Nudge the head toward the ball. Ball on the right means yaw down, ball low means pitch up."""
        if abs(sighting.x) > DEAD_BAND:
            yaw -= GAIN * sighting.x

        if abs(sighting.y) > DEAD_BAND:
            pitch += GAIN * sighting.y

        return self._clamped(pitch, yaw)

    def _unseen(self, pitch: float, yaw: float) -> tuple[float, float]:
        """Hold the pose for a while, then sweep."""
        self._unseen_updates += 1

        if not self.searching:
            return pitch, yaw

        return self._sweep(yaw)

    def _sweep(self, yaw: float) -> tuple[float, float]:
        """Triangle wave on yaw between the sweep limits, pitch a bit down."""
        yaw += self._sweep_dir * SEARCH_RATE * self._dt

        if abs(yaw) >= SEARCH_YAW_MAX:
            yaw = max(-SEARCH_YAW_MAX, min(SEARCH_YAW_MAX, yaw))
            self._sweep_dir = -self._sweep_dir

        return self._clamped(SEARCH_PITCH, yaw)

    def _clamped(self, pitch: float, yaw: float) -> tuple[float, float]:
        """Inside the trained head ranges."""
        pitch = max(-self._pitch_max, min(self._pitch_max, pitch))
        yaw = max(-self._yaw_max, min(self._yaw_max, yaw))

        return pitch, yaw
