"""HTTP front door for the bridge. Localhost only, stdlib only.

The server never touches the policy: it queues commands on BridgeState
and reads the status snapshot. The sim thread applies them through
SkillRunner, so a dead server can never take the sim down.

Routes:
    GET  /status    the latest status snapshot, and a sign the brain is alive
    GET  /status?peek=1   the same snapshot, without counting as a brain request
    POST /walk      {"vx", "vy", "wz", "seconds"}
    POST /look      {"pitch", "yaw"}
    POST /gesture   {"name"}
    POST /sit       sit down, no body
    POST /stand     stand back up, no body
    POST /roll      one forward roll, no body
    POST /get_up    get up off the floor, no body
    POST /kick      {"foot": "right" | "left"}, kick with that foot, right by default
    POST /ball      {"foot": "right" | "left"}, new ball in front of that foot
    POST /ground_pick   beak to the floor and back up, no body
    POST /stop      zero twist, head and gesture
    POST /reset     stop, and respawn the robot where the sim supports it
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from bridge.state import BridgeState


def start_bridge(state: BridgeState, port: int) -> ThreadingHTTPServer:
    """Serve the bridge API on 127.0.0.1:port in a daemon thread."""

    # Bind to loopback only. The bridge is never reachable from the network.
    server = BridgeServer(state, port)

    # Daemon thread: the process exits with the sim, no shutdown call needed.
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="bridge-http")
    thread.start()

    return server


# Carries the shared state so every request handler can reach it.
class BridgeServer(ThreadingHTTPServer):

    def __init__(self, state: BridgeState, port: int):
        super().__init__(("127.0.0.1", port), BridgeHandler)
        self.state = state


# Any request problem, with the HTTP code it should answer with.
class HttpError(Exception):

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _peek_wanted(query: str) -> bool:
    """True for ?peek=1. Any other value, and a missing flag, count as a brain request."""
    return parse_qs(query).get("peek", [""])[0] == "1"


class BridgeHandler(BaseHTTPRequestHandler):

    # POST path to handler method name. Each method takes the body and returns the echo.
    POST_ROUTES = {
        "/walk": "_walk",
        "/stop": "_stop",
        "/look": "_look",
        "/gesture": "_gesture",
        "/sit": "_sit",
        "/stand": "_stand",
        "/roll": "_roll",
        "/get_up": "_get_up",
        "/kick": "_kick",
        "/ball": "_ball",
        "/ground_pick": "_ground_pick",
        "/reset": "_reset",
    }

    # Silence the default per-request log line so the sim console stays clean.
    def log_message(self, *_):
        pass

    # GET has one route: the latest status snapshot.
    def do_GET(self):
        url = urlsplit(self.path)

        if url.path != "/status":
            self._reply_error(HttpError(404, f"unknown route {self.path}"))
            return

        self._reply(200, self._status(url.query))

    # POST carries a command: parse the body, route it, answer with the echo.
    def do_POST(self):
        try:
            body = self._parse_body()
            echo = self._route(body)

        except HttpError as err:
            self._reply_error(err)

        else:
            self._reply(200, echo)

    @property
    def _state(self) -> BridgeState:
        return self.server.state

    # A peek reads the snapshot only. Anything else marks the brain as alive.
    def _status(self, query: str) -> dict:
        if _peek_wanted(query):
            return self._state.peek_status()

        return self._state.get_status()

    # Reads the request body as a JSON object. Empty body means empty command.
    def _parse_body(self) -> dict:
        length = self._content_length()
        raw = self.rfile.read(length) if length else b"{}"

        try:
            body = json.loads(raw) if raw.strip() else {}

        except json.JSONDecodeError:
            raise HttpError(400, "body is not valid JSON")

        # A bare number or list parses fine but has no fields to read.
        if not isinstance(body, dict):
            raise HttpError(400, "body must be a JSON object")

        return body

    # Body length from the header. A non-numeric or negative value is a 400.
    def _content_length(self) -> int:
        raw = self.headers.get("Content-Length")

        if not raw:
            return 0

        try:
            length = int(raw)

        except ValueError:
            raise HttpError(400, "Content-Length is not a number")

        if length < 0:
            raise HttpError(400, "Content-Length is negative")

        return length

    # Looks up the route for this path and runs it. Bad values become a 400.
    def _route(self, body: dict) -> dict:
        method_name = self.POST_ROUTES.get(self.path)

        if method_name is None:
            raise HttpError(404, f"unknown route {self.path}")

        # submit_* raise ValueError or TypeError on bad values. Nothing is queued then.
        try:
            return getattr(self, method_name)(body)

        except (ValueError, TypeError) as exc:
            raise HttpError(400, str(exc))

    # Walk for a while. Missing speeds default to zero, missing seconds to the bridge default.
    def _walk(self, body: dict) -> dict:
        return self._state.submit_walk(
            body.get("vx", 0.0),
            body.get("vy", 0.0),
            body.get("wz", 0.0),
            body.get("seconds"),
        )

    # Sit down. The sitstand policy runs the move, the walk routes stay closed until it stands.
    def _sit(self, _body: dict) -> dict:
        return self._state.submit_posture(True)

    # Stand back up out of a sit.
    def _stand(self, _body: dict) -> dict:
        return self._state.submit_posture(False)

    # One forward roll. The roulade policy owns the robot until its timer ends.
    def _roll(self, _body: dict) -> dict:
        return self._state.submit_trick("roll")

    # Get up off the floor. For a seated robot /stand is the route instead.
    def _get_up(self, _body: dict) -> dict:
        return self._state.submit_trick("get_up")

    # Kick with one foot, right by default. Nothing checks for a ball.
    def _kick(self, body: dict) -> dict:
        return self._state.submit_kick(body.get("foot", "right"))

    # Put a new ball in front of one foot, right by default.
    def _ball(self, body: dict) -> dict:
        return self._state.submit_ball(body.get("foot", "right"))

    # One ground pick cycle. Nothing is grabbed, the sim has no mouth.
    def _ground_pick(self, _body: dict) -> dict:
        return self._state.submit_ground_pick()

    # Zero everything now.
    def _stop(self, _body: dict) -> dict:
        return self._state.submit_stop()

    # Zero everything and put the robot back at its spawn.
    def _reset(self, _body: dict) -> dict:
        return self._state.submit_reset()

    # Hold a head pose until the next look or stop.
    def _look(self, body: dict) -> dict:
        return self._state.submit_look(body.get("pitch", 0.0), body.get("yaw", 0.0))

    # Play a named head gesture. An unknown name raises and becomes a 400.
    def _gesture(self, body: dict) -> dict:
        return self._state.submit_gesture(body.get("name"))

    def _reply_error(self, err: HttpError):
        self._reply(err.code, {"error": err.message})

    # Every response is JSON with an explicit length.
    def _reply(self, code: int, body: dict):
        payload = json.dumps(body).encode()

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()

        self.wfile.write(payload)
