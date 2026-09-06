"""viewer_bridge turns its policy flags into sessions, limits and a robot model."""
import os
import sys
from argparse import Namespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from viewer_bridge import (
    BALL_ROLLING_FRICTION,
    HEAD_CAMERA_CFG,
    bridge_limits,
    keep_alive,
    needs_ball,
    needs_camera,
    needs_ground_contact,
    parse_args,
    policy_paths,
    rolling_ball_spec,
    viewer_cfg,
)
from viewer_commander import HEAD_CAMERA


def _args(sitstand=None, roulade=None, standup=None, kick_right=None, kick_left=None, ground_pick=None,
          follow_ball=False) -> Namespace:
    return Namespace(
        policy="walk.onnx",
        sitstand=sitstand,
        roulade=roulade,
        standup=standup,
        kick_right=kick_right,
        kick_left=kick_left,
        ground_pick=ground_pick,
        follow_ball=follow_ball,
    )


def test_the_walking_policy_is_the_only_one_loaded_by_default():
    assert policy_paths(_args()) == {"walking": "walk.onnx"}


def test_every_policy_flag_loads_a_session_under_its_own_name():
    paths = policy_paths(_args(sitstand="sit.onnx", roulade="roll.onnx", standup="up.onnx"))
    assert paths == {
        "walking": "walk.onnx",
        "sit": "sit.onnx",
        "roll": "roll.onnx",
        "get_up": "up.onnx",
    }


def test_trick_flags_unlock_only_their_own_session():
    limits = bridge_limits(_args(roulade="roll.onnx"))
    assert limits.roulade_session is True
    assert limits.standup_session is False
    assert limits.sit_session is False

    limits = bridge_limits(_args(standup="up.onnx"))
    assert limits.standup_session is True
    assert limits.roulade_session is False


def test_any_ground_policy_asks_for_the_ground_contact_model():
    assert needs_ground_contact(_args()) is False
    assert needs_ground_contact(_args(sitstand="sit.onnx")) is True
    assert needs_ground_contact(_args(roulade="roll.onnx")) is True
    assert needs_ground_contact(_args(standup="up.onnx")) is True


class _Cfg(Namespace):
    """Stands in for the play cfg where keep_alive only touches terminations and length."""

    def __init__(self):
        super().__init__(
            terminations={"fell_over": object(), "out_of_terrain_bounds": object(), "time_out": object(),
                          "nan_state": object()},
            episode_length_s=20.0,
        )


def test_keep_alive_drops_the_fall_and_bounds_terminations():
    cfg = _Cfg()
    keep_alive(cfg)
    assert "fell_over" not in cfg.terminations
    assert "out_of_terrain_bounds" not in cfg.terminations


def test_keep_alive_keeps_the_nan_guard_and_pushes_the_time_out_away():
    cfg = _Cfg()
    keep_alive(cfg)
    assert "nan_state" in cfg.terminations
    assert "time_out" in cfg.terminations
    assert cfg.episode_length_s == 3600.0


def test_keep_alive_survives_a_cfg_without_those_terminations():
    cfg = _Cfg()
    cfg.terminations = {}
    keep_alive(cfg)
    assert cfg.terminations == {}


def test_kick_and_ground_pick_flags_load_sessions_under_the_commander_names():
    paths = policy_paths(_args(kick_right="kr.onnx", kick_left="kl.onnx", ground_pick="gp.onnx"))
    assert paths == {
        "walking": "walk.onnx",
        "kick_right": "kr.onnx",
        "kick_left": "kl.onnx",
        "ground_pick": "gp.onnx",
    }


def test_kick_and_ground_pick_flags_ask_for_the_ground_contact_model():
    assert needs_ground_contact(_args(kick_right="kr.onnx")) is True
    assert needs_ground_contact(_args(kick_left="kl.onnx")) is True
    assert needs_ground_contact(_args(ground_pick="gp.onnx")) is True


def test_no_flags_means_no_ball():
    assert needs_ball(_args()) is False
    assert needs_ball(_args(ground_pick="gp.onnx")) is False


def test_a_kick_flag_or_follow_ball_asks_for_the_ball():
    assert needs_ball(_args(kick_right="kr.onnx")) is True
    assert needs_ball(_args(kick_left="kl.onnx")) is True
    assert needs_ball(_args(follow_ball=True)) is True


def test_the_ball_joins_the_scene_second_with_contact_headroom():
    cfg = viewer_cfg(sitstand=True, ball=True)
    assert list(cfg.scene.entities) == ["robot", "ball"]
    assert cfg.sim.nconmax == 50


def test_the_ball_rolls_to_a_stop_instead_of_rolling_for_ever():
    spec = rolling_ball_spec()
    friction = spec.geom("ball_geom").friction
    assert friction[2] == pytest.approx(BALL_ROLLING_FRICTION)
    assert friction[0] == pytest.approx(0.5)


def test_the_viewer_ball_uses_the_rolling_friction_spec():
    cfg = viewer_cfg(ball=True)
    assert cfg.scene.entities["ball"].spec_fn is rolling_ball_spec


def test_no_kick_flag_means_no_ball():
    cfg = viewer_cfg(sitstand=True)
    assert list(cfg.scene.entities) == ["robot"]


def test_kick_and_ground_pick_flags_unlock_only_their_own_session():
    limits = bridge_limits(_args(kick_right="kr.onnx"))
    assert limits.kick_right_session is True
    assert limits.kick_left_session is False
    assert limits.ground_pick_session is False

    limits = bridge_limits(_args(ground_pick="gp.onnx"))
    assert limits.ground_pick_session is True
    assert limits.kick_right_session is False


def test_only_the_follow_ball_flag_asks_for_the_camera():
    assert needs_camera(_args(kick_right="kr.onnx")) is False
    assert needs_camera(_args(kick_right="kr.onnx", follow_ball=True)) is True


def test_the_follow_ball_flag_unlocks_the_camera_route():
    assert bridge_limits(_args(kick_right="kr.onnx")).camera is False
    assert bridge_limits(_args(kick_right="kr.onnx", follow_ball=True)).camera is True


def test_the_camera_joins_the_scene_as_a_sensor_on_the_head():
    cfg = viewer_cfg(sitstand=True, ball=True, camera=True)
    cameras = [s for s in cfg.scene.sensors if s.name == HEAD_CAMERA]
    assert cameras == [HEAD_CAMERA_CFG]
    assert HEAD_CAMERA_CFG.parent_body == "robot/jaw_soft"
    assert HEAD_CAMERA_CFG.enabled_geom_groups == (0,)
    assert (HEAD_CAMERA_CFG.width, HEAD_CAMERA_CFG.height) == (160, 120)


def test_no_follow_ball_flag_means_no_camera():
    cfg = viewer_cfg(sitstand=True, ball=True)
    assert all(s.name != HEAD_CAMERA for s in (cfg.scene.sensors or ()))


def test_follow_ball_alone_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["viewer_bridge.py", "--policy", "walk.onnx", "--follow-ball"])
    args = parse_args()
    assert args.follow_ball is True
