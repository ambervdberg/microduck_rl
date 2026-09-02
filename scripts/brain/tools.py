"""LangChain tools wrapping the Microduck bridge HTTP API.

Caps and safety live in the bridge, not here: these are thin clients.
"""

import json
import os

import requests
from langchain_core.tools import tool


def _bridge_url() -> str:
    """Base URL of the bridge, from the environment or the sim default."""
    return os.environ.get("BRIDGE_URL", "http://127.0.0.1:8630")


def _request(method: str, path: str, body: dict | None = None) -> str:
    """Call the bridge, returning an error string instead of raising.

    A tool must never crash the agent: unreachable bridge, a non-JSON
    reply, or a non-2xx status all come back as {"error": ...}.
    """
    url = f"{_bridge_url()}{path}"

    try:
        resp = requests.request(method, url, json=body, timeout=5)
        data = resp.json()

    # Bridge replied, but the body was not valid JSON.
    except requests.exceptions.JSONDecodeError as exc:
        return json.dumps({"error": f"bad reply from {url} (status {resp.status_code}): {exc}"})

    # Bridge down, refused connection, or the request timed out.
    except requests.RequestException as exc:
        return json.dumps({"error": f"bridge unreachable at {url}: {exc}"})

    if not 200 <= resp.status_code < 300:
        error = data.get("error") if isinstance(data, dict) else None
        return json.dumps({"error": error or f"bridge returned status {resp.status_code}"})

    return json.dumps(data)


def _post(path: str, body: dict) -> str:
    """POST a command body to the bridge."""
    return _request("POST", path, body)


def _get(path: str) -> str:
    """GET a bridge route with no body."""
    return _request("GET", path)


@tool
def walk(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0, seconds: float = 3.0) -> str:
    """Start walking. Returns right away, the robot keeps walking for `seconds` seconds.

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
    """Point the head and hold it there. Radians.

    pitch: positive looks DOWN, negative looks up, max 1.1. yaw: positive looks left, max 1.4.
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
