"""LangChain tools wrapping the Microduck bridge HTTP API.

Caps and safety live in the bridge, not here: these are thin clients.
"""

import json
import os
import time

import requests
from langchain_core.tools import tool


def _bridge_url() -> str:
    """Base URL of the bridge, from the environment or the sim default."""
    return os.environ.get("BRIDGE_URL", "http://127.0.0.1:8630")


def _request(method: str, path: str, body: dict | None = None) -> str:
    """Call the bridge, returning an error string instead of raising.

    A tool must never crash the agent: unreachable bridge, a non-JSON
    reply, or a non-2xx status all come back as {"error": ...}.
    """
    url = f"{_bridge_url()}{path}"

    try:
        resp = requests.request(method, url, json=body, timeout=5)
        data = resp.json()

    # Bridge replied, but the body was not valid JSON.
    except requests.exceptions.JSONDecodeError as exc:
        return json.dumps({"error": f"bad reply from {url} (status {resp.status_code}): {exc}"})

    # Bridge down, refused connection, or the request timed out.
    except requests.RequestException as exc:
        return json.dumps({"error": f"bridge unreachable at {url}: {exc}"})

    if not 200 <= resp.status_code < 300:
        error = data.get("error") if isinstance(data, dict) else None
        return json.dumps({"error": error or f"bridge returned status {resp.status_code}"})

    return json.dumps(data)


def _post(path: str, body: dict) -> str:
    """POST a command body to the bridge."""
    return _request("POST", path, body)


def _get(path: str) -> str:
    """GET a bridge route with no body."""
    return _request("GET", path)


IDLE_POLL_S = 0.25

# How long to wait for the sim to pick the command up before watching for the end of it.
START_WAIT_S = 1.0

# Wait cap for a walk of N seconds. Sim time in the viewer runs slower than wall time.
WALK_WAIT_FACTOR = 2.0
WALK_WAIT_MARGIN_S = 2.0

GESTURE_MAX_WAIT_S = 6.0

# Wait cap for a sit or a stand up. Sim time in the viewer runs slower than wall time.
POSTURE_MAX_WAIT_S = 8.0

# How long the sit itself takes once the status says sitting: about a 2 s glide in sim time.
SIT_SETTLE_S = 2.5

# How long a look takes to settle, so two looks in a row are both visible.
LOOK_SETTLE_S = 1.0

# How long each trick runs. A bridge kick adds a 0.4 s lead to bring the head
# home first, the margin below covers it.
ROLL_SECONDS = 2.0
GET_UP_SECONDS = 3.0
KICK_SECONDS = 1.5
GROUND_PICK_SECONDS = 4.0

# How far the ball must travel for a kick to count as landed, in metres.
KICK_TRAVEL_M = 0.3

# How long a kick looks for the ball: the head camera needs a few pictures to find it.
LOOK_FOR_BALL_S = 3.0

# Extra wait on top of the trick seconds. Sim time in the viewer runs slower than wall time.
TRICK_WAIT_MARGIN_S = 4.0

# Wait cap for a face: a full hunt circle at HUNT_RATE plus the turn, in slow viewer time.
FACE_MAX_WAIT_S = 25.0

# Wait cap for a walk to the ball: the bridge gives up at 20 s of sim time, the viewer runs slower.
GO_TO_BALL_MAX_WAIT_S = 30.0

# Approach values that mean the walk to the ball is over.
APPROACH_OVER = ("arrived", "gave_up", "none")

# What /status reports while no trick is running.
NO_TRICK = "none"


def _is_idle(status: dict) -> bool:
    """True when the robot has no walk or gesture running."""
    twist = status.get("twist") or [0.0, 0.0, 0.0]
    return all(float(v) == 0.0 for v in twist) and not status.get("gesture")


def _is_running(status: dict) -> bool:
    """True when the status shows a walk or a gesture under way."""
    twist = status.get("twist") or [0.0, 0.0, 0.0]
    walking = float(status.get("walk_seconds_left") or 0.0) > 0.0 or any(float(v) != 0.0 for v in twist)
    return walking or bool(status.get("gesture"))


def _poll_status() -> dict:
    """Sleep one poll interval, then read the status. A bridge failure carries an "error" key."""
    time.sleep(IDLE_POLL_S)
    return json.loads(_get("/status"))


def _wait_until_started() -> None:
    """Wait for the status to show the command running, or give up after the start window.

    A paused or slow sim has not drained the command yet, and its stale idle
    status would otherwise end the wait immediately.
    """
    deadline = time.monotonic() + START_WAIT_S
    while time.monotonic() < deadline:
        status = _poll_status()
        if "error" in status or _is_running(status):
            return


def _wait_until_idle(max_wait_s: float) -> None:
    """Block until the bridge reports the robot idle, or the wait cap passes.

    Tools return only when their action is over, so the agent's next tool
    call starts after the previous one finished, and a sequence of commands
    plays out in order.
    """
    _wait_until_started()

    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        status = _poll_status()
        if "error" in status or _is_idle(status):
            return


def _wait_for_status(field: str, target: str | bool, max_wait_s: float) -> None:
    """Block until a status field reaches its target, or the wait cap passes.

    The start window comes first: a stale snapshot from before the sim drained
    the command would otherwise end the wait at once.
    """
    time.sleep(START_WAIT_S)
    deadline = time.monotonic() + max_wait_s

    while time.monotonic() < deadline:
        status = _poll_status()

        if "error" in status or status.get(field) == target:
            return


def _walk_max_wait(reply: dict) -> float:
    """Wait cap for a walk, from the seconds the bridge echoed back."""
    seconds = float(reply.get("seconds") or 0.0)
    return WALK_WAIT_FACTOR * seconds + WALK_WAIT_MARGIN_S


def _gesture_max_wait(_reply: dict) -> float:
    """Wait cap for a gesture. Every gesture is about as long as the next."""
    return GESTURE_MAX_WAIT_S


def _post_and_wait(path: str, body: dict, max_wait) -> str:
    """POST a command, then wait for it to finish. Errors return at once.

    max_wait reads the reply, so a walk sizes its cap from the duration the
    bridge accepted rather than the one that was asked for.
    """
    reply = _post(path, body)
    echo = json.loads(reply)
    if "error" not in echo:
        _wait_until_idle(max_wait(echo))
    return reply


def _post_and_watch(path: str, field: str, target: str | bool, max_wait_s: float, settle_s: float = 0.0,
                    body: dict | None = None) -> str:
    """POST a command, then wait for a status field to reach its target. Errors return at once.

    settle_s covers a move the status cannot time: the sit flag flips at the
    start of the glide, the stand up has its own rising posture.
    """
    reply = _post(path, body or {})
    echo = json.loads(reply)

    if "error" not in echo:
        _wait_for_status(field, target, max_wait_s)
        time.sleep(settle_s)

    return reply


def _post_and_wait_for(path: str, field: str, targets: tuple, max_wait_s: float) -> str:
    """POST a command, wait for a status field to reach one of the targets, return that field.

    A bridge error returns at once. A wait that runs out returns the last value seen.
    The reply also carries lost and at_ball from the last status read.
    A "none" before the walk has started is the old status and is skipped.
    """
    reply = _post(path, {})
    echo = json.loads(reply)

    if "error" in echo:
        return reply

    time.sleep(START_WAIT_S)
    deadline = time.monotonic() + max_wait_s
    value = None
    status = {}
    started = False

    while time.monotonic() < deadline:
        status = _poll_status()

        if "error" in status:
            return json.dumps(status)

        value = status.get(field)

        if value != "none":
            started = True

        if value in targets and (value != "none" or started):
            break

    return json.dumps({field: value, "lost": bool(status.get("lost", False)),
                        "at_ball": bool(status.get("at_ball", False))})


def _post_posture(path: str, target: str, settle_s: float = 0.0) -> str:
    """POST a sit or a stand up, then wait for the robot to reach that posture."""
    return _post_and_watch(path, "posture", target, POSTURE_MAX_WAIT_S, settle_s)


def _post_trick(path: str, seconds: float, body: dict | None = None) -> str:
    """POST a trick, then wait for the status to say no trick is running any more."""
    return _post_and_watch(path, "trick", NO_TRICK, seconds + TRICK_WAIT_MARGIN_S, body=body)


def _post_and_settle(path: str, body: dict) -> str:
    """POST a command, then wait a fixed settle time. Errors return at once."""
    reply = _post(path, body)
    echo = json.loads(reply)
    if "error" not in echo:
        time.sleep(LOOK_SETTLE_S)
    return reply


@tool
def walk(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, seconds: float = 3.0) -> str:
    """Walk, and return when the walk is over (about `seconds` seconds later).

    vx: forward speed in m/s (max 0.4, negative walks backward).
    vy: sideways speed in m/s (max 0.3, positive is left).
    wz: turn speed in rad/s (max 1.0, positive turns left).
    seconds: how long to walk (max 10). For longer walks, call again.
    """
    return _post_and_wait("/walk", {"vx": vx, "vy": vy, "wz": wz, "seconds": seconds}, _walk_max_wait)


@tool
def stop() -> str:
    """Stop immediately: zero speed, head to neutral, cancel gestures."""
    return _post("/stop", {})


@tool
def look(pitch: float = 0.0, yaw: float = 0.0) -> str:
    """Point the head and hold it there. Returns after the head has moved.

    The head stays there until the next look or stop, so call look(0, 0) to
    look straight ahead again. Radians.

    pitch: positive looks DOWN, negative looks up, max 1.1. yaw: positive looks left, max 1.4.
    """
    return _post_and_settle("/look", {"pitch": pitch, "yaw": yaw})


@tool
def gesture(name: str) -> str:
    """Play a head gesture: 'nod' (yes) or 'shake' (no). Returns when it is done."""
    return _post_and_wait("/gesture", {"name": name}, _gesture_max_wait)


@tool
def sit() -> str:
    """Sit down on the floor. Returns once the robot is sitting.

    The robot cannot walk while sitting. Call stand_up first.
    """
    return _post_posture("/sit", "sitting", SIT_SETTLE_S)


@tool
def stand_up() -> str:
    """Stand back up out of a sit. Returns once the robot is standing."""
    return _post_posture("/stand", "standing")


@tool
def roll() -> str:
    """Do one forward roll, a roulade. Returns once the roll is over.

    A trick to play when someone asks for it. The robot must be standing and
    free: it cannot roll while sitting or during another trick.
    """
    return _post_trick("/roll", ROLL_SECONDS)


@tool
def get_up() -> str:
    """Get up off the floor after a fall. Returns once the robot is back up.

    Use this when the robot fell over or the user says it is lying on the
    floor. Not for a sit: to leave a sit, call stand_up.
    """
    return _post_trick("/get_up", GET_UP_SECONDS)


@tool
def kick(foot: str = "auto") -> str:
    """Kick the ball, looking for it and walking up to it first when it is not already at a foot.

    foot: 'auto' lets the robot kick with the foot the ball is on, 'left' or
    'right' pick one. The robot must be standing and free. The reply says foot,
    kicked true only when the ball moved, travel in metres, walked_up true when
    the robot walked to the ball first, and looked true when it turned the head
    camera on to find the ball. Do not call new_ball before a kick on your own,
    the user asks for a ball.
    """
    status, error = _status_or_error()
    if error:
        return error
    looked = False

    if not status.get("ball_seen"):
        looked = _look_for_ball()
        status, error = _status_or_error()
        if error:
            return error

    walked_up = False

    if _ball_far(status):
        approach = _walk_up()
        if "error" in approach:
            return json.dumps(approach)
        walked_up = True

    reply = _post_trick("/kick", KICK_SECONDS, {"foot": foot})
    echo = json.loads(reply)
    if "error" in echo:
        return reply

    status, error = _status_or_error()
    if error:
        return error

    return _kick_reply(echo, status, walked_up, looked)


def _status_or_error() -> tuple[dict, str | None]:
    """Read /status. On a bridge error, return the reply to return early instead."""
    status = json.loads(_get("/status"))
    if "error" in status:
        return status, json.dumps(status)
    return status, None


def _look_for_ball() -> bool:
    """Turn the head camera on and wait for the ball to show up. True when the follow started.

    The bridge only knows where the ball is while a follow runs, so a kick out
    of the blue has to start one.
    """
    if "error" in json.loads(_post("/follow_ball", {})):
        return False

    deadline = time.monotonic() + LOOK_FOR_BALL_S

    while time.monotonic() < deadline:
        if _poll_status().get("ball_seen"):
            break

    return True


def _ball_far(status: dict) -> bool:
    """True when the ball is in view but out of kick reach."""
    return bool(status.get("ball_seen")) and not status.get("ball_close")


def _walk_up() -> dict:
    """Walk to the ball, the same wait go_to_ball makes, parsed."""
    return json.loads(_post_and_wait_for("/go_to_ball", "approach", APPROACH_OVER, GO_TO_BALL_MAX_WAIT_S))


def _kick_reply(echo: dict, status: dict, walked_up: bool, looked: bool) -> str:
    """Build the kick reply from the trick the bridge ran and the status after it."""
    travel = float(status.get("last_kick_travel") or 0.0)
    foot = str(echo.get("trick") or "").removeprefix("kick_")

    return json.dumps({"foot": foot, "kicked": travel > KICK_TRAVEL_M, "travel": travel, "walked_up": walked_up,
                       "looked": looked})


@tool
def new_ball(foot: str = "right") -> str:
    """Put a new ball in front of one foot, 'right' or 'left'. Returns at once.

    There is one ball. It moves to the kick spot of that foot. Walking can
    push it away, a kick sends it rolling. The bridge never reports where it is.
    """
    return _post("/ball", {"foot": foot})


@tool
def ground_pick() -> str:
    """Bow the beak down to the floor and back up, one 4 s cycle. Returns when it is over.

    Nothing is grabbed, it is the bow only. The robot must be standing and free.
    """
    return _post_trick("/ground_pick", GROUND_PICK_SECONDS)


@tool
def follow_ball() -> str:
    """Turn the head to keep the ball in view. Returns at once, the follow keeps running.

    It stays on until stop, sit, look, a gesture or a trick. When the ball is out
    of view the head sweeps left and right until it shows up again. Walking is
    allowed while following. Read status for ball_seen to answer "do you see it".
    """
    return _post("/follow_ball", {})


@tool
def face_ball() -> str:
    """Turn on the spot until the ball is straight ahead. Returns once the body stops turning.

    Starts follow_ball too, so the head keeps the ball in view afterwards. The
    robot must be standing and free. While the ball is out of view the body
    turns a full circle to look for it. If the reply has lost true, the ball
    was not found anywhere and the robot stopped, tell the user so.
    """
    return _with_face_outcome(_post_and_watch("/face_ball", "turning", False, FACE_MAX_WAIT_S))


def _with_face_outcome(reply: str) -> str:
    """Add the bridge's lost and turning flags to a face reply. Errors pass through."""
    echo = json.loads(reply)
    if "error" in echo:
        return reply

    status = json.loads(_get("/status"))
    echo["lost"] = bool(status.get("lost", False))
    echo["turning"] = bool(status.get("turning", False))

    return json.dumps(echo)


@tool
def go_to_ball() -> str:
    """Walk to the ball and stop one foot length in front of it. Returns when it is there or gave up.

    Faces the ball first, walks at the slow speed, keeps the head on the ball.
    The reply says approach: arrived, gave_up (20 s without getting there) or
    walking (still going when the wait ran out). The reply also says lost true
    when the ball was never found, and at_ball true when a kick can reach it.
    """
    return _post_and_wait_for("/go_to_ball", "approach", APPROACH_OVER, GO_TO_BALL_MAX_WAIT_S)


@tool
def status() -> str:
    """Current robot state: active policy, speeds, head pose, fallen or not."""
    return _get("/status")


ALL_TOOLS = [
    walk, stop, look, gesture, sit, stand_up, roll, get_up, kick, new_ball, ground_pick, follow_ball, face_ball,
    go_to_ball, status,
]
