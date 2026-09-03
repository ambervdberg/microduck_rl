"""Viser viewer plus the bridge API, so a brain can drive the robot you watch in the browser.

    uv run scripts/viewer_bridge.py --policy walk_lowspeed-range.onnx [--bridge-port 8630]

Then open http://localhost:8080 for the viewer. The bridge listens on 127.0.0.1:<bridge-port>
with the same routes infer_policy.py --bridge serves (/walk, /look, /gesture, /stop, /status),
so scripts/brain works unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge.server import start_bridge
from bridge.state import BridgeState
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer
from viewer_commander import ViewerCommander, ViewerLimits

import mjlab_microduck.tasks  # noqa: F401  registers the tasks

TASK = "Mjlab-Velocity-Flat-MicroDuck"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="ONNX walking policy")
    parser.add_argument("--bridge-port", type=int, default=8630)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_env(device: str) -> RslRlVecEnvWrapper:
    """One quiet env: no pushes, no obs noise, play config."""
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.events.pop("push_robot", None)
    for term in cfg.observations["actor"].terms.values():
        term.noise = None
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    return RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)


class OnnxPolicy:
    """Policy callable for the viewer. Ticks the commander before every action."""

    def __init__(self, path: str, device: str, commander: ViewerCommander):
        self._session = ort.InferenceSession(path)
        self._input = self._session.get_inputs()[0].name
        self._device = device
        self._commander = commander

    def __call__(self, obs) -> torch.Tensor:
        self._commander.tick()
        actor = obs["actor"] if hasattr(obs, "keys") else obs
        action = self._session.run(None, {self._input: actor.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.as_tensor(action, device=self._device)


def main() -> None:
    args = parse_args()
    env = build_env(args.device)
    unwrapped = env.unwrapped
    unwrapped.reset()

    state = BridgeState(ViewerLimits())
    commander = ViewerCommander(unwrapped, state, unwrapped.step_dt)
    start_bridge(state, args.bridge_port)
    print(f"[bridge] listening on 127.0.0.1:{args.bridge_port}")

    policy = OnnxPolicy(args.policy, args.device, commander)
    ViserPlayViewer(env, policy).run()
    env.close()


if __name__ == "__main__":
    main()
