"""Tool tests against a stub bridge server (stdlib only, no LLM)."""

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools


def _commands(received):
    """The recorded requests without the idle polls walk and gesture make."""
    return [r for r in received if r[1] != "/status"]


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
    monkeypatch.setattr(tools, "IDLE_POLL_S", 0.01)
    yield received
    server.shutdown()


def test_walk_posts_clamped_arguments(stub_bridge):
    import tools
    result = tools.walk.invoke({"vx": 0.2, "vy": 0.0, "wz": 0.1, "seconds": 2.0})
    assert _commands(stub_bridge) == [("POST", "/walk", {"vx": 0.2, "vy": 0.0, "wz": 0.1, "seconds": 2.0})]
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
    paths = [r[1] for r in _commands(stub_bridge)]
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


@pytest.fixture()
def stub_bridge_non_json(monkeypatch):
    """Replies 200 with a body that is not JSON."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            payload = b"not json"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("BRIDGE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield
    server.shutdown()


def test_status_reports_bad_reply_with_status_code_on_non_json_body(stub_bridge_non_json):
    import tools
    result = json.loads(tools.status.invoke({}))

    assert "bad reply" in result["error"]
    assert "200" in result["error"]


@pytest.fixture()
def stub_bridge_error_status(monkeypatch):
    """Replies 400 with a JSON body carrying an error key."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            payload = json.dumps({"error": "speed too high"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("BRIDGE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield
    server.shutdown()


def test_walk_surfaces_bridge_error_body_on_non_2xx_status(stub_bridge_error_status):
    import tools
    result = json.loads(tools.walk.invoke({"vx": 5.0}))

    assert result["error"] == "speed too high"


@pytest.fixture
def busy_then_idle_bridge(monkeypatch):
    """Status says walking for two polls, then idle. Records every request."""
    received = []
    polls = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _handle(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else {}
            received.append((self.command, self.path, body))
            if self.path == "/status":
                polls["n"] += 1
                twist = [0.3, 0.0, 0.0] if polls["n"] <= 2 else [0.0, 0.0, 0.0]
                payload = json.dumps({"ready": True, "twist": twist, "gesture": None}).encode()
            else:
                payload = json.dumps({"echo": body}).encode()
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
    monkeypatch.setattr(tools, "IDLE_POLL_S", 0.01)
    yield received
    server.shutdown()


def test_walk_waits_until_the_bridge_reports_idle(busy_then_idle_bridge):
    tools.walk.invoke({"vx": 0.3, "seconds": 1.0})
    status_polls = [r for r in busy_then_idle_bridge if r[1] == "/status"]
    assert len(status_polls) == 3
    assert busy_then_idle_bridge[0][1] == "/walk"


def test_wait_gives_up_after_the_cap(busy_then_idle_bridge, monkeypatch):
    monkeypatch.setattr(tools, "IDLE_MAX_WAIT_S", 0.015)
    before = len(busy_then_idle_bridge)
    tools.walk.invoke({"vx": 0.3, "seconds": 1.0})
    status_polls = [r for r in busy_then_idle_bridge[before:] if r[1] == "/status"]
    assert 1 <= len(status_polls) <= 2
