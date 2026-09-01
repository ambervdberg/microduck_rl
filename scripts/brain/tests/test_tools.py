"""Tool tests against a stub bridge server (stdlib only, no LLM)."""

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def stub_bridge(monkeypatch):
    """Records requests, replies 200 with an echo of path and body."""
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _handle(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else {}
            received.append((self.command, self.path, body))
            payload = json.dumps({"path": self.path, "echo": body}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _handle
        do_POST = _handle

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("BRIDGE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield received
    server.shutdown()


def test_walk_posts_clamped_arguments(stub_bridge):
    import tools
    result = tools.walk.invoke({"vx": 0.2, "vy": 0.0, "wz": 0.1, "seconds": 2.0})
    assert stub_bridge == [("POST", "/walk", {"vx": 0.2, "vy": 0.0, "wz": 0.1, "seconds": 2.0})]
    assert "echo" in result


def test_status_gets(stub_bridge):
    import tools
    tools.status.invoke({})
    assert stub_bridge[0][:2] == ("GET", "/status")


def test_gesture_and_stop_and_look(stub_bridge):
    import tools
    tools.gesture.invoke({"name": "nod"})
    tools.stop.invoke({})
    tools.look.invoke({"pitch": 0.2, "yaw": -0.3})
    paths = [r[1] for r in stub_bridge]
    assert paths == ["/gesture", "/stop", "/look"]


def test_status_reports_error_when_bridge_unreachable(monkeypatch):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()
    monkeypatch.setenv("BRIDGE_URL", f"http://127.0.0.1:{closed_port}")

    import tools
    result = tools.status.invoke({})

    assert "error" in json.loads(result)
