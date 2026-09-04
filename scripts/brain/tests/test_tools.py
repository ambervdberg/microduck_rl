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
    monkeypatch.setattr(tools, "START_WAIT_S", 0.02)
    monkeypatch.setattr(tools, "LOOK_SETTLE_S", 0.0)
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


class _FakeClock:
    """Stands in for tools.time: sleeping moves the clock, nothing blocks."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _walking(seconds_left=1.0):
    return {"ready": True, "twist": [0.3, 0.0, 0.0], "walk_seconds_left": seconds_left, "gesture": None}


def _idle():
    return {"ready": True, "twist": [0.0, 0.0, 0.0], "walk_seconds_left": 0.0, "gesture": None}


def _scripted_bridge(monkeypatch, statuses):
    """Serves the given status bodies in order, the last one forever. Records every request."""
    received = []
    remaining = list(statuses)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _handle(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else {}
            received.append((self.command, self.path, body))
            if self.path == "/status":
                status = remaining.pop(0) if len(remaining) > 1 else remaining[0]
                payload = json.dumps(status).encode()
            else:
                payload = json.dumps({"echo": body, "seconds": body.get("seconds")}).encode()
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
    monkeypatch.setattr(tools, "time", _FakeClock())
    return received, server


def _polls(received):
    """Only the status polls, in the order the tool made them."""
    return [r for r in received if r[1] == "/status"]


def test_walk_waits_until_the_bridge_reports_idle(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_walking(), _walking(), _idle()])
    try:
        tools.walk.invoke({"vx": 0.3, "seconds": 1.0})
    finally:
        server.shutdown()

    # The walk goes out first, then one poll while it runs, then the idle one that ends the wait.
    assert [r[1] for r in received] == ["/walk", "/status", "/status", "/status"]


def test_walk_ignores_a_stale_idle_status_before_the_sim_picks_the_command_up(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_idle(), _walking(), _walking(), _idle()])
    try:
        tools.walk.invoke({"vx": 0.3, "seconds": 1.0})
    finally:
        server.shutdown()

    # The stale idle does not end the wait: the tool keeps polling to the real idle.
    assert len(_polls(received)) == 4


def test_wait_gives_up_after_the_cap_derived_from_the_walk(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_walking()])
    try:
        tools.walk.invoke({"vx": 0.3, "seconds": 1.0})
    finally:
        server.shutdown()

    cap = tools.WALK_WAIT_FACTOR * 1.0 + tools.WALK_WAIT_MARGIN_S
    assert len(_polls(received)) == 1 + int(cap / tools.IDLE_POLL_S)


def test_gesture_gives_up_after_its_own_cap(monkeypatch):
    gesturing = {"ready": True, "twist": [0.0, 0.0, 0.0], "gesture": "nod"}
    received, server = _scripted_bridge(monkeypatch, [gesturing])
    try:
        tools.gesture.invoke({"name": "nod"})
    finally:
        server.shutdown()

    assert len(_polls(received)) == 1 + int(tools.GESTURE_MAX_WAIT_S / tools.IDLE_POLL_S)


def test_look_settles_on_success_and_returns_at_once_on_an_error(monkeypatch):
    clock = _FakeClock()
    _, server = _scripted_bridge(monkeypatch, [_idle()])
    monkeypatch.setattr(tools, "time", clock)
    try:
        tools.look.invoke({"pitch": 0.2, "yaw": 0.0})
        assert clock.now == tools.LOOK_SETTLE_S

        # A rejected look has nothing to settle: the head never moved.
        monkeypatch.setattr(tools, "_post", lambda path, body: json.dumps({"error": "head angle too big"}))
        tools.look.invoke({"pitch": 9.0, "yaw": 0.0})
        assert clock.now == tools.LOOK_SETTLE_S
    finally:
        server.shutdown()


def _seated():
    return {"ready": True, "twist": [0.0, 0.0, 0.0], "posture": "sitting", "sitting": True}


def _rising():
    return {"ready": True, "twist": [0.0, 0.0, 0.0], "posture": "rising", "sitting": False}


def _standing():
    return {"ready": True, "twist": [0.0, 0.0, 0.0], "posture": "standing", "sitting": False}


def test_sit_posts_and_waits_for_the_sitting_posture(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_standing(), _seated()])
    try:
        tools.sit.invoke({})
    finally:
        server.shutdown()

    assert [r[1] for r in received] == ["/sit", "/status", "/status"]
    assert _commands(received) == [("POST", "/sit", {})]


def test_stand_up_waits_through_the_rise(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_seated(), _rising(), _standing()])
    try:
        tools.stand_up.invoke({})
    finally:
        server.shutdown()

    assert [r[1] for r in received] == ["/stand", "/status", "/status", "/status"]


def test_posture_wait_gives_up_after_its_own_cap(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_standing()])
    try:
        tools.sit.invoke({})
    finally:
        server.shutdown()

    assert len(_polls(received)) == int(tools.POSTURE_MAX_WAIT_S / tools.IDLE_POLL_S)


def test_sit_returns_at_once_when_the_bridge_rejects_it(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_standing()])
    monkeypatch.setattr(tools, "_post", lambda path, body: json.dumps({"error": "no sit policy loaded"}))
    try:
        result = json.loads(tools.sit.invoke({}))
    finally:
        server.shutdown()

    assert "error" in result
    assert _polls(received) == []


def test_sit_and_stand_up_are_registered_tools():
    names = [t.name for t in tools.ALL_TOOLS]
    assert "sit" in names
    assert "stand_up" in names


def test_sit_waits_the_start_window_then_settles(monkeypatch):
    clock = _FakeClock()
    _received, server = _scripted_bridge(monkeypatch, [_seated()])
    monkeypatch.setattr(tools, "time", clock)
    try:
        tools.sit.invoke({})
    finally:
        server.shutdown()

    # Start window, one poll that reads the target, then the sit settle time.
    assert clock.now == pytest.approx(tools.START_WAIT_S + tools.IDLE_POLL_S + tools.SIT_SETTLE_S)


def test_stand_up_waits_the_start_window_and_does_not_settle(monkeypatch):
    clock = _FakeClock()
    _received, server = _scripted_bridge(monkeypatch, [_standing()])
    monkeypatch.setattr(tools, "time", clock)
    try:
        tools.stand_up.invoke({})
    finally:
        server.shutdown()

    assert clock.now == pytest.approx(tools.START_WAIT_S + tools.IDLE_POLL_S)


def _rolling():
    return {"ready": True, "twist": [0.0, 0.0, 0.0], "posture": "standing", "trick": "rolling"}


def _getting_up():
    return {"ready": True, "twist": [0.0, 0.0, 0.0], "posture": "standing", "trick": "getting_up"}


def _no_trick():
    return {"ready": True, "twist": [0.0, 0.0, 0.0], "posture": "standing", "trick": "none"}


def test_roll_posts_and_waits_until_the_trick_is_over(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_rolling(), _rolling(), _no_trick()])
    try:
        tools.roll.invoke({})
    finally:
        server.shutdown()

    assert [r[1] for r in received] == ["/roll", "/status", "/status", "/status"]
    assert _commands(received) == [("POST", "/roll", {})]


def test_get_up_posts_and_waits_until_the_trick_is_over(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_getting_up(), _no_trick()])
    try:
        tools.get_up.invoke({})
    finally:
        server.shutdown()

    assert [r[1] for r in received] == ["/get_up", "/status", "/status"]
    assert _commands(received) == [("POST", "/get_up", {})]


def test_roll_returns_at_once_when_the_bridge_rejects_it(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_no_trick()])
    monkeypatch.setattr(tools, "_post", lambda path, body: json.dumps({"error": "a trick is running, wait for it"}))
    try:
        result = json.loads(tools.roll.invoke({}))
    finally:
        server.shutdown()

    assert "error" in result
    assert _polls(received) == []


def test_roll_wait_gives_up_after_the_cap_from_the_roll_seconds(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_rolling()])
    try:
        tools.roll.invoke({})
    finally:
        server.shutdown()

    cap = tools.ROLL_SECONDS + tools.TRICK_WAIT_MARGIN_S
    assert len(_polls(received)) == int(cap / tools.IDLE_POLL_S)


def test_get_up_wait_gives_up_after_the_cap_from_the_get_up_seconds(monkeypatch):
    received, server = _scripted_bridge(monkeypatch, [_getting_up()])
    try:
        tools.get_up.invoke({})
    finally:
        server.shutdown()

    cap = tools.GET_UP_SECONDS + tools.TRICK_WAIT_MARGIN_S
    assert len(_polls(received)) == int(cap / tools.IDLE_POLL_S)


def test_roll_and_get_up_are_registered_tools():
    names = [t.name for t in tools.ALL_TOOLS]
    assert "roll" in names
    assert "get_up" in names
