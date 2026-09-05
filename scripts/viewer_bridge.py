"""Viser viewer plus the bridge API, so a brain can drive the robot you watch in the browser.

    uv run scripts/viewer_bridge.py --policy walk_lowspeed-range.onnx [--sitstand sitstand.onnx]
        [--roulade roulade.onnx] [--standup standup.onnx]
        [--kick-right kick_right.onnx] [--kick-left kick_left.onnx] [--ground-pick ground_pick.onnx]
        [--follow-ball]
        [--bridge-port 8630] [--viewer-port 8632]

Then open http://localhost:8632 for the viewer. The bridge listens on 127.0.0.1:<bridge-port>
with the same routes infer_policy.py --bridge serves (/walk, /look, /gesture, /sit, /stand,
/roll, /get_up, /kick, /ball, /ground_pick, /follow_ball, /face_ball, /go_to_ball, /stop, /reset,
/status), so scripts/brain works unchanged.
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
from mjlab.sensor.camera_sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer
from viewer_commander import HEAD_CAMERA, ViewerCommander, ViewerLimits

import mjlab_microduck.tasks  # noqa: F401  registers the tasks
from mjlab_microduck.robot.microduck_constants import MICRODUCK_BALL_CFG, MICRODUCK_STANDUP_ROBOT_CFG

TASK = "Mjlab-Velocity-Flat-MicroDuck"
LONG_EPISODE_S = 3600.0

# Contact headroom for the ball, the same value the kick task uses.
BALL_NCONMAX = 50

# The head_camera exported into the model faces backward and lies on its side.
# The viewer adds its own on the head, looking ahead and 25 degrees down.
HEAD_CAMERA_CFG = CameraSensorCfg(
    name=HEAD_CAMERA,
    parent_body="robot/jaw_soft",
    pos=(0.0155, -0.0000914, -0.0733),
    quat=(0.6903, -0.153, 0.153, -0.6903),
    fovy=75.0,
    width=160,
    height=120,
    data_types=("rgb",),
    use_shadows=False,
    use_textures=True,
    # Group 0 only (floor and ball): the collision meshes are orange-brown and
    # would fool the ball finder, the visual meshes block the view from inside the head.
    enabled_geom_groups=(0,),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="ONNX walking policy")
    parser.add_argument("--sitstand", help="ONNX sitstand policy, unlocks /sit and /stand")
    parser.add_argument("--roulade", help="ONNX roulade policy, unlocks /roll")
    parser.add_argument("--standup", help="ONNX standup policy, unlocks /get_up")
    parser.add_argument("--kick-right", help="ONNX right foot kick policy, unlocks /kick and /ball for that foot")
    parser.add_argument("--kick-left", help="ONNX left foot kick policy, unlocks /kick and /ball for that foot")
    parser.add_argument("--ground-pick", help="ONNX ground pick policy, unlocks /ground_pick")
    parser.add_argument("--follow-ball", action="store_true",
                        help="render a head camera, unlocks follow, face and go to ball, brings its own ball")
    parser.add_argument("--bridge-port", type=int, default=8630)
    parser.add_argument("--viewer-port", type=int, default=8632)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def keep_alive(cfg) -> None:
    """Never respawn on a fall. A roll passes 177 deg and get up needs a fallen robot.

    nan_state stays. time_out stays as a term, pushed so far out it never fires.
    """
    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.episode_length_s = LONG_EPISODE_S


def viewer_cfg(sitstand: bool = False, ball: bool = False, camera: bool = False):
    """One quiet play cfg: no pushes, no obs noise, never respawns.

    sitstand also stands for roll, get up, kick and ground pick: they all need the
    ground-contact model. ball adds the kick ball as a second entity, robot first.
    camera adds the head camera sensor the follow ball skill reads.
    """
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    keep_alive(cfg)

    # A robot on the ground falls through the walk model's floor, only the feet collide.
    if sitstand:
        cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    # The robot stays first: reset events write its root state at qpos[:, 0:7].
    if ball:
        cfg.scene.entities = {**cfg.scene.entities, "ball": MICRODUCK_BALL_CFG}
        cfg.sim.nconmax = BALL_NCONMAX

    if camera:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (HEAD_CAMERA_CFG,)

    cfg.events.pop("push_robot", None)
    for term in cfg.observations["actor"].terms.values():
        term.noise = None

    # Front three-quarter view at robot height.
    cfg.viewer.distance = 1.24
    cfg.viewer.azimuth = 160.0
    cfg.viewer.elevation = 17.0
    cfg.viewer.lookat = (0.0, 0.0, 0.0)

    return cfg


def build_env(device: str, sitstand: bool = False, ball: bool = False, camera: bool = False) -> RslRlVecEnvWrapper:
    """The viewer env built from viewer_cfg."""
    env = ManagerBasedRlEnv(cfg=viewer_cfg(sitstand, ball, camera), device=device)
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

    for name, path in (
        ("sit", args.sitstand),
        ("roll", args.roulade),
        ("get_up", args.standup),
        ("kick_right", args.kick_right),
        ("kick_left", args.kick_left),
        ("ground_pick", args.ground_pick),
    ):
        if path:
            paths[name] = path

    return paths


def bridge_limits(args: argparse.Namespace) -> ViewerLimits:
    """The envelope plus which sessions the flags loaded, so the bridge unlocks those routes."""
    return ViewerLimits(
        sit_session=bool(args.sitstand),
        roulade_session=bool(args.roulade),
        standup_session=bool(args.standup),
        kick_right_session=bool(args.kick_right),
        kick_left_session=bool(args.kick_left),
        ground_pick_session=bool(args.ground_pick),
        camera=bool(args.follow_ball),
    )


def needs_ground_contact(args: argparse.Namespace) -> bool:
    """True when a loaded policy puts the robot on the ground and needs its collisions."""
    return bool(args.sitstand or args.roulade or args.standup or args.kick_right or args.kick_left or args.ground_pick)


def needs_ball(args: argparse.Namespace) -> bool:
    """True when a kick policy is loaded, or the follow ball skill wants one to follow."""
    return bool(args.kick_right or args.kick_left or args.follow_ball)


def needs_camera(args: argparse.Namespace) -> bool:
    """True when the follow ball skill is on, the only reader of the head camera."""
    return bool(args.follow_ball)


def main() -> None:
    args = parse_args()
    env = build_env(args.device, needs_ground_contact(args), needs_ball(args), needs_camera(args))
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
