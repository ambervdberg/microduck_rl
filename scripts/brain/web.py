"""Chat page for the brain. Type in the browser, the agent drives the bridge.

    UV_PROJECT_ENVIRONMENT=~/.venvs/microduck_brain uv run --project scripts/brain \\
        python scripts/brain/web.py [--port 8631] [--host 0.0.0.0] [--viewer-port 8632]

Routes (matched on the path suffix, so the page works behind a path prefix too):
    GET  .../          the chat page
    GET  .../status    the bridge status, passed through
    GET  .../config    {"viewer_port": N}, the port the page loads the viewer from
    POST .../chat      {"text": "..."} -> {"reply": "..."}
    POST .../clear     forget the conversation and reset the sim
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import _check_env, make_agent, run_turn
from tools import _bridge_url

CHAT_PAGE = Path(__file__).with_name("chat.html")

VIEWER_PORT = 8632

# The page itself is served here: at the root locally, at the mount tailscale serve maps.
PAGE_PATHS = ("", "/chat")


class ChatBrain:
    """One conversation, one turn at a time."""

    def __init__(self, agent):
        self._agent = agent
        self._messages: list = []
        self._lock = threading.Lock()

    def say(self, text: str) -> str:
        with self._lock:
            self._messages.append({"role": "user", "content": text})
            self._messages, reply = run_turn(self._agent, self._messages)
        return reply

    def clear(self) -> None:
        """Forget the conversation so the next message starts fresh."""
        with self._lock:
            self._messages = []


def bridge_status() -> dict:
    """The bridge's /status, or a not-ready stub when the sim is down."""
    try:
        return requests.get(_bridge_url() + "/status", timeout=2).json()
    except requests.RequestException as exc:
        return {"ready": False, "error": str(exc)}


def reset_bridge() -> str | None:
    """Put the robot back at its spawn. Returns the failure text, or None when it worked."""
    try:
        requests.post(_bridge_url() + "/reset", json={}, timeout=2)
        return None
    except requests.RequestException as exc:
        return str(exc)


class ChatServer(ThreadingHTTPServer):
    def __init__(self, brain: ChatBrain, host: str, port: int, viewer_port: int = VIEWER_PORT):
        super().__init__((host, port), ChatHandler)
        self.brain = brain
        self.viewer_port = viewer_port


class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        path = self.path.rstrip("/")
        if path.endswith("/status"):
            self._reply_json(200, bridge_status())
        elif path.endswith("/config"):
            self._reply_json(200, {"viewer_port": self.server.viewer_port})
        elif path in PAGE_PATHS:
            self._reply_bytes(200, "text/html; charset=utf-8", CHAT_PAGE.read_bytes())
        else:
            self._reply_json(404, {"error": f"unknown route {self.path}"})

    def do_POST(self):
        path = self.path.rstrip("/")
        raw = self._read_body()
        if raw is None:
            self._reply_json(400, {"error": "Content-Length is not a length"})
        elif path.endswith("/clear"):
            self._reply_json(200, self._clear())
        elif path.endswith("/chat"):
            self._say(raw)
        else:
            self._reply_json(404, {"error": f"unknown route {self.path}"})

    def _clear(self) -> dict:
        """Forget the conversation and reset the sim. A dead bridge is reported, not raised."""
        self.server.brain.clear()
        error = reset_bridge()
        return {"cleared": True, "bridge_error": error} if error else {"cleared": True}

    def _say(self, raw: bytes) -> None:
        """Run one chat turn on the posted text."""
        text = _chat_text(raw)
        if text is None:
            self._reply_json(400, {"error": "body must be JSON with a non-empty 'text'"})
            return
        self._reply_json(200, {"reply": self.server.brain.say(text)})

    def _read_body(self) -> bytes | None:
        """The request body. None when the Content-Length header is not a length."""
        raw = self.headers.get("Content-Length")
        if not raw:
            return b""
        try:
            length = int(raw)
        except ValueError:
            return None
        if length < 0:
            return None
        return self.rfile.read(length) if length else b""

    def _reply_json(self, code: int, payload: dict) -> None:
        self._reply_bytes(code, "application/json", json.dumps(payload).encode())

    def _reply_bytes(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _chat_text(raw: bytes) -> str | None:
    """The 'text' field of a chat body, or None when the body cannot carry one."""
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return None
    text = body.get("text") if isinstance(body, dict) else None
    return text.strip() if isinstance(text, str) and text.strip() else None


def serve(brain: ChatBrain, host: str, port: int, viewer_port: int = VIEWER_PORT) -> ChatServer:
    server = ChatServer(brain, host, port, viewer_port)
    threading.Thread(target=server.serve_forever, daemon=True, name="chat-http").start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8631)
    parser.add_argument("--viewer-port", type=int, default=VIEWER_PORT, help="port the viser viewer serves on")
    args = parser.parse_args()

    _check_env()
    server = serve(ChatBrain(make_agent()), args.host, args.port, args.viewer_port)
    print(f"Chat page on http://localhost:{args.port}  (bridge at {_bridge_url()})")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
