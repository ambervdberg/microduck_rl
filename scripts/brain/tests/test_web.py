"""Chat server with a fake agent: no Azure, no sim."""
import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web


class _EchoAgent:
    def invoke(self, payload):
        text = payload["messages"][-1]["content"]
        return {"messages": payload["messages"] + [SimpleNamespace(text=f"ok: {text}")]}


class _FailingAgent:
    def invoke(self, payload):
        raise RuntimeError("azure down")


def _start(agent):
    server = web.serve(web.ChatBrain(agent), "127.0.0.1", 0)
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post(url, body: bytes):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_chat_returns_reply_and_keeps_history():
    server, url = _start(_EchoAgent())
    try:
        code, data = _post(url + "/chat", json.dumps({"text": "walk"}).encode())
        assert code == 200
        assert data["reply"] == "ok: walk"
        assert len(server.brain._messages) == 2
    finally:
        server.shutdown()


def test_chat_behind_path_prefix():
    server, url = _start(_EchoAgent())
    try:
        code, data = _post(url + "/chat/chat", json.dumps({"text": "nod"}).encode())
        assert code == 200
        assert data["reply"] == "ok: nod"
    finally:
        server.shutdown()


def test_chat_error_drops_pending_message():
    server, url = _start(_FailingAgent())
    try:
        code, data = _post(url + "/chat", json.dumps({"text": "walk"}).encode())
        assert code == 200
        assert data["reply"].startswith("Error:")
        assert server.brain._messages == []
    finally:
        server.shutdown()


def test_bad_body_is_400():
    server, url = _start(_EchoAgent())
    try:
        code, _ = _post(url + "/chat", b"not json")
        assert code == 400
        code, _ = _post(url + "/chat", json.dumps({"text": "  "}).encode())
        assert code == 400
    finally:
        server.shutdown()


def test_root_serves_the_page():
    server, url = _start(_EchoAgent())
    try:
        with urllib.request.urlopen(url + "/") as resp:
            assert resp.status == 200
            assert b"Jeeves" in resp.read()
    finally:
        server.shutdown()
