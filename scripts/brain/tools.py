"""LangChain tools wrapping the Microduck bridge HTTP API.

Caps and safety live in the bridge, not here: these are thin clients.
"""

import json
import os

import requests
from langchain_core.tools import tool


def _bridge_url() -> str:
    return os.environ.get("BRIDGE_URL", "http://127.0.0.1:8630")


def _request(method: str, path: str, body: dict | None = None) -> str:
    """Call the bridge, returning an error string instead of raising.

    A tool must never crash the agent: unreachable bridge or a bad
    response both come back as {"error": ...} for the model to read.
    """
    url = f"{_bridge_url()}{path}"
    try:
        resp = requests.request(method, url, json=body, timeout=5)
        return json.dumps(resp.json())
    except requests.RequestException as exc:
        return json.dumps({"error": f"bridge unreachable at {url}: {exc}"})
    except ValueError as exc:
        return json.dumps({"error": f"bad response from {url}: {exc}"})


def _post(path: str, body: dict) -> str:
    return _request("POST", path, body)


def _get(path: str) -> str:
    return _request("GET", path)


@tool
def walk(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, seconds: float = 3.0) -> str:
    """Walk for a few seconds, then stop automatically.

    vx: forward speed in m/s (max 0.4, negative walks backward).
    vy: sideways speed in m/s (max 0.3, positive is left).
    wz: turn speed in rad/s (max 1.0, positive turns left).
    seconds: how long to walk (max 10). For longer walks, call again.
    """
    return _post("/walk", {"vx": vx, "vy": vy, "wz": wz, "seconds": seconds})


@tool
def stop() -> str:
    """Stop immediately: zero speed, head to neutral, cancel gestures."""
    return _post("/stop", {})


@tool
def look(pitch: float = 0.0, yaw: float = 0.0) -> str:
    """Point the head and hold it there. Radians, max 1.4.

    pitch: positive looks up. yaw: positive looks left.
    """
    return _post("/look", {"pitch": pitch, "yaw": yaw})


@tool
def gesture(name: str) -> str:
    """Play a head gesture: 'nod' (yes) or 'shake' (no)."""
    return _post("/gesture", {"name": name})


@tool
def status() -> str:
    """Current robot state: active policy, speeds, head pose, fallen or not."""
    return _get("/status")


ALL_TOOLS = [walk, stop, look, gesture, status]
