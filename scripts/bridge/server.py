"""HTTP front door for the bridge. Localhost only, stdlib only.

The server never touches the policy: it queues commands on BridgeState
and reads the status snapshot. The sim thread applies them through
SkillRunner, so a dead server can never take the sim down.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bridge.state import BridgeState


def start_bridge(state: BridgeState, port: int) -> ThreadingHTTPServer:
    """Serve the bridge API on 127.0.0.1:port in a daemon thread."""

    # Bind to loopback only; the bridge is never reachable from the network.
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


class BridgeHandler(BaseHTTPRequestHandler):

    # POST path to handler method name; each method takes the body and returns the echo.
    POST_ROUTES = {
        "/walk": "_walk",
        "/stop": "_stop",
        "/look": "_look",
        "/gesture": "_gesture",
    }

    # Silence the default per-request log line so the sim console stays clean.
    def log_message(self, *_):
        pass

    # GET has one route: the latest status snapshot.
    def do_GET(self):
        if self.path != "/status":
            self._reply_error(HttpError(404, f"unknown route {self.path}"))
            return

        self._reply(200, self._state.get_status())

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

    # Reads the request body as a JSON object; empty body means empty command.
    def _parse_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"

        try:
            body = json.loads(raw) if raw.strip() else {}

        except json.JSONDecodeError:
            raise HttpError(400, "body is not valid JSON")

        # A bare number or list parses fine but has no fields to read.
        if not isinstance(body, dict):
            raise HttpError(400, "body must be a JSON object")

        return body

    # Looks up the route for this path and runs it; bad values become a 400.
    def _route(self, body: dict) -> dict:
        method_name = self.POST_ROUTES.get(self.path)

        if method_name is None:
            raise HttpError(404, f"unknown route {self.path}")

        # submit_* raise ValueError or TypeError on bad values; nothing is queued then.
        try:
            return getattr(self, method_name)(body)

        except (ValueError, TypeError) as exc:
            raise HttpError(400, str(exc))

    # Walk for a while; missing speeds default to zero, missing seconds to the bridge default.
    def _walk(self, body: dict) -> dict:
        return self._state.submit_walk(
            body.get("vx", 0.0),
            body.get("vy", 0.0),
            body.get("wz", 0.0),
            body.get("seconds"),
        )

    # Zero everything now.
    def _stop(self, _body: dict) -> dict:
        return self._state.submit_stop()

    # Hold a head pose until the next look or stop.
    def _look(self, body: dict) -> dict:
        return self._state.submit_look(body.get("pitch", 0.0), body.get("yaw", 0.0))

    # Play a named head gesture; an unknown name raises and becomes a 400.
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
