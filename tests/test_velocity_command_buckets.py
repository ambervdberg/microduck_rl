"""Explicit sampling buckets for the rare command regions.

Uniform sampling makes "walk slowly", "turn slowly" and "spin on the spot"
roughly 2% of the draw each, so the policy averages them into "stand still"
and they never train. One draw per resample partitions the three buckets, so
an env lands in at most one of them, and every bucketed env is un-marked as
standing (a standing env would have its command zeroed again).
"""

from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    LOW_SPEED_FRACTION,
    LOW_SPEED_X_RANGE,
    SLOW_TURN_ANG_RANGE,
    SLOW_TURN_FRACTION,
    TURN_IN_PLACE_FRACTION,
    make_microduck_velocity_env_cfg,
)

NUM_ENVS = 20000


def _sampler():
    """A VelocityCommandCommandOnly with only the fields the buckets touch."""
    sampler = object.__new__(microduck_mdp.VelocityCommandCommandOnly)
    sampler._env = SimpleNamespace(device="cpu")  # backs the read-only .device
    sampler.cfg = SimpleNamespace(
        ranges=SimpleNamespace(ang_vel_z=(-1.0, 1.0)),
        rel_turn_in_place_envs=TURN_IN_PLACE_FRACTION,
        rel_low_speed_envs=LOW_SPEED_FRACTION,
        low_speed_x_range=LOW_SPEED_X_RANGE,
        rel_slow_turn_envs=SLOW_TURN_FRACTION,
        slow_turn_ang_range=SLOW_TURN_ANG_RANGE,
    )
    sampler.vel_command_b = torch.zeros(NUM_ENVS, 3)
    sampler.vel_command_w = torch.zeros(NUM_ENVS, 3)
    sampler.is_standing_env = torch.ones(NUM_ENVS, dtype=torch.bool)
    return sampler


def _bucketed_ids(sampler, env_ids):
    """Run one resample, recording which ids each bucket claimed."""
    claimed = {}

    for name in ("turn_in_place", "low_speed", "slow_turn"):
        method = getattr(sampler, f"_command_{name}")

        def record(ids, name=name, method=method):
            claimed[name] = ids
            method(ids)

        setattr(sampler, f"_command_{name}", record)

    sampler._apply_command_buckets(env_ids)
    return claimed


def test_fractions_are_present_and_in_range():
    for fraction in (TURN_IN_PLACE_FRACTION, LOW_SPEED_FRACTION, SLOW_TURN_FRACTION):
        assert 0.0 < fraction < 1.0
    # The buckets share one draw, so they only partition if they fit inside it.
    assert TURN_IN_PLACE_FRACTION + LOW_SPEED_FRACTION + SLOW_TURN_FRACTION < 1.0


def test_buckets_are_disjoint_and_sized_by_their_fractions():
    sampler = _sampler()
    env_ids = torch.arange(NUM_ENVS)
    claimed = _bucketed_ids(sampler, env_ids)

    sets = {name: set(ids.tolist()) for name, ids in claimed.items()}
    assert sets["turn_in_place"] & sets["low_speed"] == set()
    assert sets["turn_in_place"] & sets["slow_turn"] == set()
    assert sets["low_speed"] & sets["slow_turn"] == set()

    expected = {
        "turn_in_place": TURN_IN_PLACE_FRACTION,
        "low_speed": LOW_SPEED_FRACTION,
        "slow_turn": SLOW_TURN_FRACTION,
    }
    for name, fraction in expected.items():
        assert abs(len(sets[name]) / NUM_ENVS - fraction) < 0.02, name


def test_low_speed_bucket_samples_the_configured_range():
    sampler = _sampler()
    ids = torch.arange(NUM_ENVS)
    sampler._command_low_speed(ids)

    speed = sampler.vel_command_b[ids, 0].abs()
    lo, hi = LOW_SPEED_X_RANGE
    assert speed.min() >= lo
    assert speed.max() <= hi
    # Both directions must be practised.
    forward = (sampler.vel_command_b[ids, 0] > 0).float().mean().item()
    assert 0.45 < forward < 0.55
    # Slow walking is straight: no lateral, no yaw.
    assert torch.all(sampler.vel_command_b[ids, 1] == 0.0)
    assert torch.all(sampler.vel_command_b[ids, 2] == 0.0)


def test_slow_turn_bucket_samples_the_configured_range():
    sampler = _sampler()
    ids = torch.arange(NUM_ENVS)
    sampler._command_slow_turn(ids)

    rate = sampler.vel_command_b[ids, 2].abs()
    lo, hi = SLOW_TURN_ANG_RANGE
    assert rate.min() >= lo
    assert rate.max() <= hi
    left = (sampler.vel_command_b[ids, 2] > 0).float().mean().item()
    assert 0.45 < left < 0.55
    # Turning in place: no linear command.
    assert torch.all(sampler.vel_command_b[ids, :2] == 0.0)


def test_turn_in_place_bucket_keeps_its_fast_range():
    sampler = _sampler()
    ids = torch.arange(NUM_ENVS)
    sampler._command_turn_in_place(ids)

    rate = sampler.vel_command_b[ids, 2].abs()
    assert rate.min() >= 0.4  # 0.4 * max, with ang_vel_z capped at 1.0
    assert rate.max() <= 1.0
    assert torch.all(sampler.vel_command_b[ids, :2] == 0.0)


def test_bucketed_envs_are_no_longer_standing():
    sampler = _sampler()
    env_ids = torch.arange(NUM_ENVS)
    claimed = _bucketed_ids(sampler, env_ids)

    for name, ids in claimed.items():
        assert not sampler.is_standing_env[ids].any(), name
        # The world-frame copy must carry the same command.
        assert torch.equal(sampler.vel_command_w[ids], sampler.vel_command_b[ids])


def test_cfg_wires_all_three_buckets():
    cfg = make_microduck_velocity_env_cfg()
    command = cfg.commands["twist"]

    assert isinstance(command, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert command.rel_turn_in_place_envs == TURN_IN_PLACE_FRACTION
    assert command.rel_low_speed_envs == LOW_SPEED_FRACTION
    assert command.low_speed_x_range == LOW_SPEED_X_RANGE
    assert command.rel_slow_turn_envs == SLOW_TURN_FRACTION
    assert command.slow_turn_ang_range == SLOW_TURN_ANG_RANGE

    # The low-speed bucket must cover the deadzone, which sat below ~0.18 m/s.
    assert LOW_SPEED_X_RANGE[1] <= command.ranges.lin_vel_x[1]
    # The slow-turn bucket must sit strictly below the turn-in-place bucket.
    assert SLOW_TURN_ANG_RANGE[1] <= 0.4 * command.ranges.ang_vel_z[1]
