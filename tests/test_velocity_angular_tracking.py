"""Angular-velocity tracking: split instantaneous / EMA.

A walking microduck sways at ~1.0 rad/s in yaw. Scored instantaneously against
a commanded 0, that pays a frozen policy the maximum (+1.999/s measured) and a
slow walk +0.831/s, so standing outbids walking at small commands. The EMA
cancels the sway and charges only the sustained rate, so a real turn and a
heading drift still cost.
"""

import math

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _Data:
    def __init__(self, n):
        self.root_link_ang_vel_b = torch.zeros(n, 3)


class _Asset:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, asset):
        self._asset = asset

    def __getitem__(self, _):
        return self._asset


class _Cmd:
    def __init__(self, n):
        self.cmd = torch.zeros(n, 3)

    def get_command(self, _):
        return self.cmd


class _Env:
    def __init__(self, n):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.episode_length_buf = torch.full((n,), 10, dtype=torch.long)
        self.scene = _Scene(_Asset(_Data(n)))
        self.command_manager = _Cmd(n)


STD = math.sqrt(0.05)
TAU = 0.5


def _run(env, steps, std=STD, tau_s=TAU):
    out = None
    for _ in range(steps):
        out = microduck_mdp.track_angular_velocity_ema(
            env, std=std, command_name="twist", tau_s=tau_s
        )
    return out


def test_alpha_matches_the_env_step_dt():
    # One step from a zeroed buffer must land exactly on alpha * yaw rate,
    # with alpha = 1 - exp(-dt/tau) read off env.step_dt (not hardcoded).
    for dt in (0.02, 0.005):
        env = _Env(1)
        env.step_dt = dt
        env.scene._asset.data.root_link_ang_vel_b[:, 2] = 1.0
        out = _run(env, 1)
        alpha = 1.0 - math.exp(-dt / TAU)
        assert abs(env._ang_vel_ema[0, 2].item() - alpha) < 1e-9
        expected = math.exp(-(alpha**2) / STD**2)
        assert abs(out[0].item() - expected) < 1e-6


def test_gait_sway_around_zero_scores_near_one():
    # +-1.0 rad/s per-step yaw sway, zero mean, zero command: the whole point
    # of the term. Instantaneous scoring collapses on the same signal.
    env = _Env(1)
    yaw = env.scene._asset.data.root_link_ang_vel_b
    out = None
    for i in range(500):
        yaw[:, 2] = 1.0 if i % 2 else -1.0
        out = microduck_mdp.track_angular_velocity_ema(
            env, std=STD, command_name="twist", tau_s=TAU
        )
    assert out[0].item() > 0.95
    instantaneous = math.exp(-(1.0**2) / 0.5)
    assert instantaneous < 0.15


def test_sustained_turn_of_the_same_magnitude_scores_low():
    # Same 1.0 rad/s magnitude, but sustained instead of swaying, against a
    # zero command: an uncommanded heading drift must still be charged.
    env = _Env(1)
    env.scene._asset.data.root_link_ang_vel_b[:, 2] = 1.0
    out = _run(env, 500)
    assert out[0].item() < 1e-6


def test_commanded_turn_is_paid():
    env = _Env(1)
    env.command_manager.cmd[:, 2] = 1.0
    env.scene._asset.data.root_link_ang_vel_b[:, 2] = 1.0
    out = _run(env, 500)
    assert out[0].item() > 0.99


def test_roll_pitch_rates_are_scored_like_the_instantaneous_term():
    # mjlab's track_angular_velocity adds the xy rates to the z error; the EMA
    # companion forms the same error, so swaying roll/pitch cancels but a
    # sustained roll rate is charged.
    env = _Env(1)
    env.scene._asset.data.root_link_ang_vel_b[:, 0] = 0.5
    out = _run(env, 500)
    assert abs(out[0].item() - math.exp(-0.25 / 0.05)) < 1e-3


def test_reset_clears_the_ema():
    env = _Env(2)
    env.command_manager.cmd[:, 2] = 1.0
    env.scene._asset.data.root_link_ang_vel_b[:, 2] = 1.0
    _run(env, 500)
    env.episode_length_buf[0] = 1  # env 0 just reset
    _run(env, 1)
    alpha = 1.0 - math.exp(-0.02 / TAU)
    assert abs(env._ang_vel_ema[0, 2].item() - alpha) < 1e-6
    assert env._ang_vel_ema[1, 2].item() > 0.99


def test_velocity_cfg_wiring():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg()

    # Linear tracking untouched: run 7's setting, the only accurate one.
    lin = cfg.rewards["track_linear_velocity"]
    assert lin.weight == 2.0
    assert lin.params["std"] == math.sqrt(0.04)

    # Instantaneous angular term kept, but too weak to outbid walking.
    inst = cfg.rewards["track_angular_velocity"]
    assert inst.weight == 0.5
    assert inst.params["std"] == math.sqrt(0.5)

    # EMA angular term carries the mass, tight because sway is removed.
    ema = cfg.rewards["track_angular_velocity_ema"]
    assert ema.func is microduck_mdp.track_angular_velocity_ema
    assert ema.weight == 2.0
    assert ema.params["std"] == math.sqrt(0.05)
    assert ema.params["tau_s"] == 0.5
    assert ema.params["command_name"] == "twist"

    # Total angular mass stays close to the 2.0 it replaces.
    assert inst.weight + ema.weight == 2.5
