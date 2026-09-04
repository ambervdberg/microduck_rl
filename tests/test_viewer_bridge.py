"""viewer_bridge turns its policy flags into sessions, limits and a robot model."""
import os
import sys
from argparse import Namespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from viewer_bridge import bridge_limits, keep_alive, needs_ground_contact, policy_paths


def _args(sitstand=None, roulade=None, standup=None) -> Namespace:
    return Namespace(policy="walk.onnx", sitstand=sitstand, roulade=roulade, standup=standup)


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
