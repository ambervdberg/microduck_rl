"""Linear-velocity tracking: tight std on the instantaneous velocity.

The tight instantaneous term gave the best speed accuracy measured (92-99% of
commanded between 0.2 and 0.45 m/s) and is what the cfg wires. Its deadzone
below ~0.18 m/s is a coverage problem, fixed by the small-command sampling
buckets, not by loosening the term.

``track_linear_velocity_ema`` is a recorded negative result: measuring against
a 1 s EMA removed the deadzone but smeared credit assignment into a 40%
overshoot. The function and its unit tests stay; the cfg no longer wires it.
"""

import math

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp


class _Data:
    def __init__(self, n):
        self.root_link_lin_vel_b = torch.zeros(n, 3)


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


STD = math.sqrt(0.02)


def _run(env, steps, std=STD, tau_s=1.0):
    out = None
    for _ in range(steps):
        out = microduck_mdp.track_linear_velocity_ema(
            env, std=std, command_name="twist", tau_s=tau_s
        )
    return out


def test_alpha_matches_the_env_step_dt():
    # One step from a zeroed buffer must land exactly on alpha * velocity,
    # with alpha = 1 - exp(-dt/tau) read off env.step_dt (not hardcoded).
    for dt in (0.02, 0.005):
        env = _Env(1)
        env.step_dt = dt
        env.scene._asset.data.root_link_lin_vel_b[:, 0] = 0.5
        out = _run(env, 1)
        alpha = 1.0 - math.exp(-dt / 1.0)
        assert abs(env._lin_vel_ema[0, 0].item() - alpha * 0.5) < 1e-9
        expected = math.exp(-((0.0 - alpha * 0.5) ** 2) / STD**2)
        assert abs(out[0].item() - expected) < 1e-6


def test_ema_settles_on_the_true_average_speed():
    env = _Env(1)
    env.command_manager.cmd[:, 0] = 0.15
    env.scene._asset.data.root_link_lin_vel_b[:, 0] = 0.15
    out = _run(env, 500)  # 10 s
    assert out[0].item() > 0.99


def test_gait_sway_does_not_lower_the_reward():
    # Mean speed on command, +-0.3 m/s per-step sway: the instantaneous term
    # collapses, the EMA term stays near 1.
    env = _Env(1)
    env.command_manager.cmd[:, 0] = 0.15
    vel = env.scene._asset.data.root_link_lin_vel_b
    out = None
    for i in range(500):
        vel[:, 0] = 0.15 + (0.3 if i % 2 else -0.3)
        out = microduck_mdp.track_linear_velocity_ema(
            env, std=STD, command_name="twist", tau_s=1.0
        )
    assert out[0].item() > 0.95
    instantaneous = math.exp(-(0.3**2) / STD**2)
    assert instantaneous < 0.02


def test_sustained_shortfall_is_charged():
    # Walking 25% slow at 0.4 m/s commanded: 0.1 m/s DC error under
    # std^2 = 0.02 costs ~40% of the term.
    env = _Env(1)
    env.command_manager.cmd[:, 0] = 0.4
    env.scene._asset.data.root_link_lin_vel_b[:, 0] = 0.3
    out = _run(env, 500)
    assert abs(out[0].item() - math.exp(-0.01 / 0.02)) < 0.01


def test_reset_clears_the_ema():
    env = _Env(2)
    env.command_manager.cmd[:, 0] = 0.4
    env.scene._asset.data.root_link_lin_vel_b[:, 0] = 0.4
    _run(env, 500)
    env.episode_length_buf[0] = 1  # env 0 just reset
    _run(env, 1)
    alpha = 1.0 - math.exp(-0.02)
    assert abs(env._lin_vel_ema[0, 0].item() - alpha * 0.4) < 1e-6
    assert env._lin_vel_ema[1, 0].item() > 0.39


def test_velocity_cfg_wiring():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg()

    # Tight instantaneous term: the run that tracked speed best.
    inst = cfg.rewards["track_linear_velocity"]
    assert inst.weight == 2.0
    assert inst.params["std"] == math.sqrt(0.04)

    # EMA term unwired — kept in mdp.py as a documented negative result only.
    assert "track_linear_velocity_ema" not in cfg.rewards

    # Angular tracking is split; see tests/test_velocity_angular_tracking.py.
    ang = cfg.rewards["track_angular_velocity"]
    assert ang.weight == 0.5
    assert ang.params["std"] == math.sqrt(0.5)
