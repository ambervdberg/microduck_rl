"""Linear-velocity tracking: tight std on the instantaneous velocity.

The tight instantaneous term gave the best speed accuracy measured (92-99% of
commanded between 0.2 and 0.45 m/s). Its deadzone below ~0.18 m/s is a
coverage problem, fixed by the small-command sampling buckets, not by
loosening the term.
"""

import math

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_velocity_cfg_wiring():
    cfg = make_microduck_velocity_env_cfg()

    # Tight instantaneous term: the std that tracked speed best. Tighter lost
    # the low-speed range for no speed gain.
    inst = cfg.rewards["track_linear_velocity"]
    assert inst.weight == 2.0
    assert inst.params["std"] == math.sqrt(0.04)

    # Angular tracking is split; see tests/test_velocity_angular_tracking.py.
    ang = cfg.rewards["track_angular_velocity"]
    assert ang.weight == 0.5
    assert ang.params["std"] == math.sqrt(0.5)
