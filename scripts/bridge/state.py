"""Thread-safe command state for the LLM bridge.

The HTTP thread submits commands. The sim thread drains and applies them.
Every value is clamped to the envelope the policy itself was given, so the
bridge can never ask for more than the keyboard can.
"""

import math
import threading
from collections import deque
from dataclasses import dataclass

# scripts/ is sys.path[0] when infer_policy.py runs.
from gestures import default_gestures

HEAD_PITCH_MAX = 1.1   # rad, trained neck/head_pitch cap
HEAD_YAW_MAX = 1.4     # rad, trained head_yaw cap

WALK_DEFAULT_S = 3.0
WALK_MAX_S = 10.0

# Seconds of silence from the brain before the bridge releases twist and head.
BRAIN_TIMEOUT_S = 10.0

# How long the stand up takes: the trained flag flip is about a 2 s glide.
RISE_SECONDS = 2.5

# How long each episodic trick runs before control goes back to walking.
ROLL_SECONDS = 2.0
GET_UP_SECONDS = 3.0

# What /status reports while no trick is running.
NO_TRICK = "none"

_GESTURE_TABLE = default_gestures()

# Short API name -> keyboard key that starts the gesture.
GESTURE_KEYS = {cfg.name.split()[0]: key for key, cfg in _GESTURE_TABLE.items()}

# Player gesture name -> the short name /gesture and /status both speak.
GESTURE_SHORT_NAMES = {cfg.name: cfg.name.split()[0] for cfg in _GESTURE_TABLE.values()}

GESTURES = tuple(GESTURE_KEYS)


@dataclass(frozen=True)
class WalkCmd:
    vx: float
    vy: float
    wz: float
    seconds: float


@dataclass(frozen=True)
class LookCmd:
    pitch: float
    yaw: float


@dataclass(frozen=True)
class GestureCmd:
    name: str


@dataclass(frozen=True)
class PostureCmd:
    """Sit down or stand back up. The sitstand policy does the move itself."""

    sit: bool


@dataclass(frozen=True)
class TrickCmd:
    """Run one episodic trick. The trick policy takes over until its timer ends."""

    name: str


@dataclass(frozen=True)
class Trick:
    """One trick: the policy that runs it, how long it takes, what /status calls it."""

    session: str
    behavior: str
    seconds: float
    status: str
    flag: str


# Trick name the API speaks -> everything the bridge needs to run and report it.
TRICKS = {
    "roll": Trick("roulade_session", "roulade", ROLL_SECONDS, "rolling", "--roulade"),
    "get_up": Trick("standup_session", "standup", GET_UP_SECONDS, "getting_up", "--standup"),
}

TRICK_NAMES = tuple(TRICKS)


@dataclass(frozen=True)
class StopCmd:
    pass


@dataclass(frozen=True)
class ResetCmd:
    pass


# What the robot could do, and what the loaded policy needs for it.
# None is always available, "gesture:<name>" needs the key bound, anything else is a session attribute.
ACTIONS = (
    ("walk", "walking_session"),
    ("look", None),
    ("nod", "gesture:nod"),
    ("shake", "gesture:shake"),
    ("sit", "sit_session"),
    ("stand up", "sit_session"),
    ("pick up", "ground_pick_session"),
    ("kick", "kick_session"),
    ("roulade", "roulade_session"),
    ("get up off the floor", "standup_session"),
)


def available_actions(policy) -> dict[str, bool]:
    """Which catalog actions the loaded policy can run."""
    return {name: _action_available(policy, needs) for name, needs in ACTIONS}


def _action_available(policy, needs: str | None) -> bool:
    """One catalog entry: nothing needed, a bound gesture, or a loaded session."""
    if needs is None:
        return True

    if needs.startswith("gesture:"):
        return _gesture_bound(policy, needs.split(":", 1)[1])

    return bool(getattr(policy, needs, None))


def _gesture_bound(policy, name: str) -> bool:
    """True when the gesture player carries the key this gesture plays on."""
    key = GESTURE_KEYS.get(name)

    return key is not None and key in policy.gesture_player.keys()


@dataclass(frozen=True)
class Envelope:
    """Speed and head limits the bridge clamps against."""

    vx_min: float
    vx_max: float
    vy_min: float
    vy_max: float
    wz_max: float
    head_pitch_max: float
    head_yaw_max: float


def policy_envelope(policy) -> Envelope:
    """Read the envelope main() installed on the policy, tightened per head axis."""
    head_max = float(policy.head_max)

    return Envelope(
        vx_min=float(policy.vel_min_x),
        vx_max=float(policy.vel_max_x),
        vy_min=float(policy.vel_min_y),
        vy_max=float(policy.vel_max_y),
        wz_max=float(policy.vel_max_ang),
        head_pitch_max=min(head_max, HEAD_PITCH_MAX),
        head_yaw_max=min(head_max, HEAD_YAW_MAX),
    )


def _clamp(value, low: float, high: float) -> float:
    """Clamp a value into [low, high]. Non-finite input is rejected."""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError("value must be a finite number")

    return max(low, min(high, value))


def _clamp_seconds(seconds) -> float:
    """Fill in the default walk duration and clamp it to the bridge max."""
    if seconds is None:
        seconds = WALK_DEFAULT_S

    return max(0.0, min(float(seconds), WALK_MAX_S))


def _walk_clamped(vx, vy, wz, seconds, cvx, cvy, cwz, cseconds) -> bool:
    """True if clamping changed any submitted walk value."""
    speeds_changed = (cvx, cvy, cwz) != (float(vx), float(vy), float(wz))
    seconds_changed = (seconds is not None) and (cseconds != float(seconds))

    return speeds_changed or seconds_changed


class BridgeState:
    """Command queue plus the status snapshot and the brain liveness counter."""

    def __init__(self, policy):
        self._policy = policy
        self._lock = threading.Lock()
        self._pending: deque = deque()
        self._status: dict = {"ready": False}
        self._request_count = 0

    def submit_walk(self, vx, vy, wz, seconds) -> dict:
        """Clamp a walk to the policy envelope, then queue it and echo what will run."""
        self._note_request()
        self._require_walking_policy()
        self._require_not_seated("sitting, stand up first")
        self._require_no_trick()

        envelope = policy_envelope(self._policy)
        cvx = _clamp(vx, envelope.vx_min, envelope.vx_max)
        cvy = _clamp(vy, envelope.vy_min, envelope.vy_max)
        cwz = _clamp(wz, -envelope.wz_max, envelope.wz_max)
        cseconds = _clamp_seconds(seconds)
        clamped = _walk_clamped(vx, vy, wz, seconds, cvx, cvy, cwz, cseconds)

        self._enqueue(WalkCmd(cvx, cvy, cwz, cseconds))

        return {"vx": cvx, "vy": cvy, "wz": cwz, "seconds": cseconds, "clamped": clamped}

    def submit_look(self, pitch, yaw) -> dict:
        """Clamp a head pose, then queue it and echo back what will run."""
        self._note_request()

        envelope = policy_envelope(self._policy)
        cp = _clamp(pitch, -envelope.head_pitch_max, envelope.head_pitch_max)
        cy = _clamp(yaw, -envelope.head_yaw_max, envelope.head_yaw_max)
        clamped = (cp, cy) != (float(pitch), float(yaw))

        self._enqueue(LookCmd(cp, cy))

        return {"pitch": cp, "yaw": cy, "clamped": clamped}

    def submit_gesture(self, name) -> dict:
        """Queue a named gesture. An unknown name or an unbound key is rejected."""
        self._note_request()

        key = GESTURE_KEYS.get(name)

        if key is None:
            raise ValueError(f"unknown gesture {name!r}, expected one of {GESTURES}")

        if key not in self._policy.gesture_player.keys():
            raise ValueError(f"gesture {name!r} is not bound on this policy")

        self._enqueue(GestureCmd(name))

        return {"gesture": name}

    def submit_posture(self, sit: bool) -> dict:
        """Queue a sit or a stand up. Head commands stay allowed in both postures."""
        self._note_request()
        self._require_sit_policy()

        sit = bool(sit)

        if sit:
            self._require_no_trick()

        self._enqueue(PostureCmd(sit))

        return {"sit": sit}

    def submit_trick(self, name) -> dict:
        """Queue one episodic trick. The robot must be standing and free."""
        self._note_request()

        trick = TRICKS.get(name)

        if trick is None:
            raise ValueError(f"unknown trick {name!r}, expected one of {TRICK_NAMES}")

        self._require_trick_policy(name, trick)
        self._require_not_seated(self._seated_message(name))
        self._require_no_trick()

        self._enqueue(TrickCmd(name))

        return {"trick": name}

    def submit_stop(self) -> dict:
        """Queue an immediate stop."""
        self._note_request()
        self._enqueue(StopCmd())

        return {"stopped": True}

    def submit_reset(self) -> dict:
        """Queue a stop plus, where the sim supports it, a respawn."""
        self._note_request()
        self._enqueue(ResetCmd())

        return {"reset": True}

    def policy(self):
        """The policy or limits object the bridge clamps against. Set once at construction."""
        return self._policy

    def drain(self) -> list:
        """Return and clear all pending commands, oldest first."""
        with self._lock:
            drained = list(self._pending)
            self._pending.clear()

        return drained

    def set_status(self, status: dict) -> None:
        """Replace the published status snapshot. Not a brain request."""
        with self._lock:
            self._status = dict(status)

    def get_status(self) -> dict:
        """Return a copy of the latest published status. Counts as a brain request."""
        with self._lock:
            self._request_count += 1

            return dict(self._status)

    def peek_status(self) -> dict:
        """Return a copy of the latest published status without feeding the watchdog."""
        with self._lock:
            return dict(self._status)

    def request_count(self) -> int:
        """Number of brain requests served so far, status polls included."""
        with self._lock:
            return self._request_count

    def _note_request(self) -> None:
        """Mark the brain as alive. The watchdog reads this count."""
        with self._lock:
            self._request_count += 1

    def _require_walking_policy(self) -> None:
        """Reject walk commands the command block would silently drop."""
        if not self._policy.walking_session:
            raise ValueError("no walking policy loaded, start infer_policy.py with --walking")

    def _require_sit_policy(self) -> None:
        """Reject posture commands no loaded policy can run."""
        if not getattr(self._policy, "sit_session", None):
            raise ValueError("no sit policy loaded, start the runner with --sitstand")

    def _require_trick_policy(self, name: str, trick: Trick) -> None:
        """Reject a trick no loaded policy can run."""
        if not getattr(self._policy, trick.session, None):
            raise ValueError(f"no {name} policy loaded, start the runner with {trick.flag}")

    @staticmethod
    def _seated_message(name: str) -> str:
        """Why a seated robot cannot run this trick. Get up is the wrong tool for a sit."""
        if name == "get_up":
            return "sitting, use stand up"

        return "sitting, stand up first"

    def _require_not_seated(self, message: str) -> None:
        """Reject a command while the robot is seated, still getting up, or a sit is queued."""
        status = self.peek_status()
        seated = status.get("sitting") or status.get("posture") == "rising"

        if seated or self._sit_queued():
            raise ValueError(message)

    def _require_no_trick(self) -> None:
        """Reject a command while a trick is running or still waiting to start."""
        running = self.peek_status().get("trick", NO_TRICK) != NO_TRICK

        if running or self._trick_queued():
            raise ValueError("a trick is running, wait for it")

    def _trick_queued(self) -> bool:
        """True when a trick is still waiting to be drained."""
        with self._lock:
            return any(isinstance(cmd, TrickCmd) for cmd in self._pending)

    def _sit_queued(self) -> bool:
        """True when the last posture command still waiting to be drained is a sit."""
        with self._lock:
            postures = [cmd.sit for cmd in self._pending if isinstance(cmd, PostureCmd)]

        return bool(postures) and postures[-1]

    def _enqueue(self, cmd) -> None:
        """Append one command to the pending queue."""
        with self._lock:
            self._pending.append(cmd)
