"""Chat page for the brain. Type in the browser, the agent drives the bridge.

    UV_PROJECT_ENVIRONMENT=~/.venvs/microduck_brain uv run --project scripts/brain \\
        python scripts/brain/web.py [--port 8631] [--host 0.0.0.0]

Routes (matched on the path suffix, so the page works behind a path prefix too):
    GET  .../          the chat page
    GET  .../status    the bridge status, passed through
    POST .../chat      {"text": "..."} -> {"reply": "..."}
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


def bridge_status() -> dict:
    """The bridge's /status, or a not-ready stub when the sim is down."""
    try:
        return requests.get(_bridge_url() + "/status", timeout=2).json()
    except requests.RequestException as exc:
        return {"ready": False, "error": str(exc)}


class ChatServer(ThreadingHTTPServer):
    def __init__(self, brain: ChatBrain, host: str, port: int):
        super().__init__((host, port), ChatHandler)
        self.brain = brain


class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/status"):
            self._reply_json(200, bridge_status())
        else:
            self._reply_bytes(200, "text/html; charset=utf-8", CHAT_PAGE.read_bytes())

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat"):
            self._reply_json(404, {"error": f"unknown route {self.path}"})
            return
        text = self._read_text()
        if text is None:
            self._reply_json(400, {"error": "body must be JSON with a non-empty 'text'"})
            return
        self._reply_json(200, {"reply": self.server.brain.say(text)})

    def _read_text(self) -> str | None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return None
        text = body.get("text") if isinstance(body, dict) else None
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _reply_json(self, code: int, payload: dict) -> None:
        self._reply_bytes(code, "application/json", json.dumps(payload).encode())

    def _reply_bytes(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(brain: ChatBrain, host: str, port: int) -> ChatServer:
    server = ChatServer(brain, host, port)
    threading.Thread(target=server.serve_forever, daemon=True, name="chat-http").start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8631)
    args = parser.parse_args()

    _check_env()
    server = serve(ChatBrain(make_agent()), args.host, args.port)
    print(f"Chat page on http://localhost:{args.port}  (bridge at {_bridge_url()})")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
