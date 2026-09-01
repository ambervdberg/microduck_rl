"""HTTP front door for the bridge. Localhost only, stdlib only.

The server never touches the policy: it queues commands on BridgeState
and reads the status snapshot. The sim thread applies them via
skills.tick, so a dead server can never take the sim down.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bridge.state import BridgeState


def start_bridge(state: BridgeState, port: int) -> ThreadingHTTPServer:
    """Serve the bridge API on 127.0.0.1:port in a daemon thread."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the sim console clean
            pass

        def _reply(self, code: int, body: dict):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/status":
                self._reply(200, state.get_status())
            else:
                self._reply(404, {"error": f"unknown route {self.path}"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                self._reply(400, {"error": "body is not valid JSON"})
                return
            try:
                if self.path == "/walk":
                    self._reply(200, state.submit_walk(
                        body.get("vx", 0.0), body.get("vy", 0.0),
                        body.get("wz", 0.0), body.get("seconds"),
                    ))
                elif self.path == "/stop":
                    self._reply(200, state.submit_stop())
                elif self.path == "/look":
                    self._reply(200, state.submit_look(
                        body.get("pitch", 0.0), body.get("yaw", 0.0),
                    ))
                elif self.path == "/gesture":
                    self._reply(200, state.submit_gesture(body.get("name")))
                else:
                    self._reply(404, {"error": f"unknown route {self.path}"})
            except (ValueError, TypeError) as exc:
                self._reply(400, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="bridge-http")
    thread.start()
    return server
