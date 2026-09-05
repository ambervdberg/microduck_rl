"""Turns ball sightings into the next head pose.

Two states. Following: each picture nudges the head toward the ball and holds
it at the middle. Searching, after LOST_AFTER_S without a ball: the head looks
a bit down and sweeps left and right until the ball shows up again. No sim here.
Head signs: positive pitch looks down, positive yaw looks left.
BodyTurner turns the ball's bearing into a body turn rate for the face ball skill.
BallHunt turns the body a full circle while the ball is out of view, then gives up.
BallApproach picks the forward speed for the go to ball skill and says when the robot is there.
"""

import math

from bridge.ball_finder import BallSighting
from bridge.state import HEAD_PITCH_MAX, HEAD_YAW_MAX

GAIN = 0.15           # rad of head turn per unit of picture offset, per picture
DEAD_BAND = 0.05      # an offset under this holds the head, no jitter at the middle
LOST_AFTER_S = 1.0    # this long without a ball starts the search
SEARCH_PITCH = 0.3    # rad down while searching, where a ball on the floor would be
SEARCH_YAW_MAX = 1.2  # rad, the sweep turns around here
SEARCH_RATE = 1.0     # rad/s of sweep

FACE_START = 0.3      # rad of ball bearing that starts a body turn
FACE_STOP = 0.1       # rad of ball bearing that ends it, the gap stops flip-flopping
FACE_GAIN = 1.0       # rad/s of body turn per rad of ball bearing
FACE_TURN_MIN = 0.3   # rad/s, the smallest turn the walk policy actually makes

APPROACH_SPEED = 0.15  # m/s, the slow walk that is measured to work
ARRIVE_PITCH = 0.73    # rad of head pitch with the ball one foot length ahead, measured in sim
ARRIVE_SIZE = 580      # orange pixels at that distance, the backup when the pitch cap comes first
GIVE_UP_S = 20.0       # walking longer than this means the ball is not reachable

HUNT_RATE = 0.6               # rad/s of body turn while the ball is out of view
HUNT_FULL_TURN = 2 * math.pi  # rad of body turn without a sighting, then the hunt gives up

CAMERA_FOVY = 75.0           # deg, vertical field of view. Same as HEAD_CAMERA_CFG in viewer_bridge.py.
CAMERA_ASPECT = 160 / 120    # picture width over height. Same as HEAD_CAMERA_CFG.
TAN_HALF_WIDTH = math.tan(math.radians(CAMERA_FOVY / 2)) * CAMERA_ASPECT


def wrap_angle(angle: float) -> float:
    """Into -pi..pi."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def ball_bearing(yaw: float, sighting: BallSighting | None) -> float:
    """Angle from the body's forward direction to the ball, positive left. The head yaw when nothing is seen."""
    if sighting is None:
        return yaw

    return yaw - math.atan(sighting.x * TAN_HALF_WIDTH)


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


class BodyTurner:
    """Turn rate for the body from the ball's bearing. Same sign: ball left, body left."""

    def __init__(self, turn_max: float):
        self._turn_max = float(turn_max)
        self._turning = False

    @property
    def turning(self) -> bool:
        """True while the body is catching up with the ball's bearing."""
        return self._turning

    def engage(self) -> None:
        """Start turning now, whatever the bearing. The turn still ends under FACE_STOP."""
        self._turning = True

    def update(self, bearing: float, searching: bool) -> float:
        """Turn rate in rad/s. Zero while searching, or while the ball is near straight ahead.
        While turning the rate is at least FACE_TURN_MIN, so the walker moves.
        """
        if searching:
            self._turning = False
            return 0.0

        if abs(bearing) >= FACE_START:
            self._turning = True
        elif abs(bearing) <= FACE_STOP:
            self._turning = False

        if not self._turning:
            return 0.0

        rate = math.copysign(max(abs(FACE_GAIN * bearing), FACE_TURN_MIN), bearing)

        return max(-self._turn_max, min(self._turn_max, rate))


class BallHunt:
    """Body turn while the ball is out of view. One full turn without a sighting gives up."""

    def __init__(self, direction: float):
        self._direction = -1.0 if direction < 0.0 else 1.0
        self._turned = 0.0
        self._last_body_yaw: float | None = None

    @property
    def turned(self) -> float:
        """Rad turned in the hunt direction so far, from the measured heading."""
        return self._turned

    @property
    def gave_up(self) -> bool:
        """True once the body has turned HUNT_FULL_TURN without a sighting."""
        return self._turned >= HUNT_FULL_TURN

    def update(self, body_yaw: float) -> float:
        """Turn rate for this picture. body_yaw is the measured heading in rad."""
        if self._last_body_yaw is not None:
            self._turned += self._direction * wrap_angle(body_yaw - self._last_body_yaw)
        self._last_body_yaw = body_yaw

        if self.gave_up:
            return 0.0

        return self._direction * HUNT_RATE


class BallApproach:
    """Forward speed toward the ball. Arrived and gave up are final until a new approach starts."""

    def __init__(self, dt: float):
        self._give_up_after = int(round(GIVE_UP_S / float(dt)))
        self._updates = 0
        self.state = "lost"

    def update(self, sighting, pitch: float, searching: bool, turning: bool) -> float:
        """Speed to walk at until the next picture."""
        if self.state in ("arrived", "gave_up"):
            return 0.0

        self._updates += 1

        if self._updates >= self._give_up_after:
            self.state = "gave_up"
            return 0.0

        if sighting is None or searching:
            self.state = "lost"
            return 0.0

        if pitch >= ARRIVE_PITCH or sighting.size >= ARRIVE_SIZE:
            self.state = "arrived"
            return 0.0

        if turning:
            self.state = "turning"
            return 0.0

        self.state = "walking"
        return APPROACH_SPEED
