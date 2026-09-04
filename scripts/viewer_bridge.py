"""Viser viewer plus the bridge API, so a brain can drive the robot you watch in the browser.

    uv run scripts/viewer_bridge.py --policy walk_lowspeed-range.onnx [--sitstand sitstand.onnx]
        [--roulade roulade.onnx] [--standup standup.onnx]
        [--bridge-port 8630] [--viewer-port 8632]

Then open http://localhost:8632 for the viewer. The bridge listens on 127.0.0.1:<bridge-port>
with the same routes infer_policy.py --bridge serves (/walk, /look, /gesture, /sit, /stand,
/roll, /get_up, /stop, /reset, /status), so scripts/brain works unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import viser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge.server import start_bridge
from bridge.state import BridgeState
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer
from viewer_commander import ViewerCommander, ViewerLimits

import mjlab_microduck.tasks  # noqa: F401  registers the tasks
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

TASK = "Mjlab-Velocity-Flat-MicroDuck"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="ONNX walking policy")
    parser.add_argument("--sitstand", help="ONNX sitstand policy, unlocks /sit and /stand")
    parser.add_argument("--roulade", help="ONNX roulade policy, unlocks /roll")
    parser.add_argument("--standup", help="ONNX standup policy, unlocks /get_up")
    parser.add_argument("--bridge-port", type=int, default=8630)
    parser.add_argument("--viewer-port", type=int, default=8632)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_env(device: str, sitstand: bool = False) -> RslRlVecEnvWrapper:
    """One quiet env: no pushes, no obs noise, play config.

    sitstand also stands for roll and get up: all three need the ground-contact model.
    """
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1

    # A robot on the ground falls through the walk model's floor, only the feet collide.
    if sitstand:
        cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    cfg.events.pop("push_robot", None)
    for term in cfg.observations["actor"].terms.values():
        term.noise = None

    # Front three-quarter view at robot height.
    cfg.viewer.distance = 1.24
    cfg.viewer.azimuth = 160.0
    cfg.viewer.elevation = 17.0
    cfg.viewer.lookat = (0.0, 0.0, 0.0)

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    return RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)


class OnnxPolicy:
    """Policy callable for the viewer. Ticks the commander, then runs the session it asks for.

    Every session takes the same 61D actor obs, so a posture change is a session swap.
    """

    def __init__(self, paths: dict[str, str], device: str, commander: ViewerCommander):
        self._sessions = {name: ort.InferenceSession(path) for name, path in paths.items()}
        self._inputs = {name: s.get_inputs()[0].name for name, s in self._sessions.items()}
        self._device = device
        self._commander = commander

    def __call__(self, obs) -> torch.Tensor:
        self._commander.tick()
        actor = obs["actor"] if hasattr(obs, "keys") else obs
        session, input_name = self._active_session()
        action = session.run(None, {input_name: actor.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.as_tensor(action, device=self._device)

    def _active_session(self):
        """The session for the current posture, falling back to walking when sit is not loaded."""
        name = self._commander.active_policy()

        if name not in self._sessions:
            name = "walking"

        return self._sessions[name], self._inputs[name]


def policy_paths(args: argparse.Namespace) -> dict[str, str]:
    """The ONNX files to load, keyed by the name the commander uses."""
    paths = {"walking": args.policy}

    for name, path in (("sit", args.sitstand), ("roll", args.roulade), ("get_up", args.standup)):
        if path:
            paths[name] = path

    return paths


def bridge_limits(args: argparse.Namespace) -> ViewerLimits:
    """The envelope plus which sessions the flags loaded, so the bridge unlocks those routes."""
    return ViewerLimits(
        sit_session=bool(args.sitstand),
        roulade_session=bool(args.roulade),
        standup_session=bool(args.standup),
    )


def needs_ground_contact(args: argparse.Namespace) -> bool:
    """True when a loaded policy puts the robot on the ground and needs its collisions."""
    return bool(args.sitstand or args.roulade or args.standup)


def main() -> None:
    args = parse_args()
    env = build_env(args.device, needs_ground_contact(args))
    unwrapped = env.unwrapped
    unwrapped.reset()

    state = BridgeState(bridge_limits(args))
    commander = ViewerCommander(unwrapped, state, unwrapped.step_dt)
    start_bridge(state, args.bridge_port)
    print(f"[bridge] listening on 127.0.0.1:{args.bridge_port}")

    policy = OnnxPolicy(policy_paths(args), args.device, commander)
    viser_server = viser.ViserServer(port=args.viewer_port)
    print(f"[viewer] http://localhost:{args.viewer_port}")

    ViserPlayViewer(env, policy, viser_server=viser_server).run()
    env.close()


if __name__ == "__main__":
    main()
