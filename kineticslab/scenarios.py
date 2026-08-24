"""Seeded labeled transient generator for the trip-logic evaluation.

Each scenario is a reactivity input built from a seeded RNG, simulated with
the Crank-Nicolson solver, then labeled from the *simulated ground truth*
trace: should-trip if the fine-grid trace actually violates a trip
condition (overpower or sustained short period), should-not otherwise.
Labeling from the physics rather than from the construction recipe keeps
the labels honest for borderline cases.
"""

import random

from . import constants as K
from .reactivity import Step, Ramp, Piecewise
from .solver import simulate
from .protection import true_condition_times

PCM = K.PCM


def make_scenarios(n_scenarios=120, seed=12345):
    """Build a deterministic list of scenario dicts (no simulation yet)."""
    rng = random.Random(seed)
    out = []
    for i in range(n_scenarios):
        kind = rng.randrange(6)
        if kind == 0:
            # large positive step: expected trip
            rho = rng.uniform(200, 550) * PCM
            fn = Step(rho, t0=1.0)
            desc = "step +%.0f pcm" % (rho / PCM)
        elif kind == 1:
            # small positive step: usually no overpower within the window,
            # possibly a period trip if large enough
            rho = rng.uniform(5, 60) * PCM
            fn = Step(rho, t0=1.0)
            desc = "step +%.0f pcm" % (rho / PCM)
        elif kind == 2:
            # negative step: benign shutdown-direction transient
            rho = -rng.uniform(50, 800) * PCM
            fn = Step(rho, t0=1.0)
            desc = "step %.0f pcm" % (rho / PCM)
        elif kind == 3:
            # fast positive ramp, clamped: expected trip
            rate = rng.uniform(20, 80) * PCM
            t1 = 1.0 + rng.uniform(3.0, 8.0)
            fn = Ramp(rate, t0=1.0, t1=t1)
            desc = "ramp +%.1f pcm/s to t=%.1f s" % (rate / PCM, t1)
        elif kind == 4:
            # slow ramp, clamped low: benign
            rate = rng.uniform(0.5, 4.0) * PCM
            t1 = 1.0 + rng.uniform(5.0, 15.0)
            fn = Ramp(rate, t0=1.0, t1=t1)
            desc = "ramp +%.1f pcm/s to t=%.1f s" % (rate / PCM, t1)
        else:
            # zigzag maneuver, capped at a modest level: benign
            peak = rng.uniform(10, 70) * PCM
            fn = Piecewise([(0.0, 0.0), (2.0, peak), (6.0, -0.5 * peak),
                            (10.0, 0.25 * peak), (15.0, 0.0)])
            desc = "zigzag peak %.0f pcm" % (peak / PCM)
        out.append({"index": i, "desc": desc, "rho_fn": fn,
                    "noise_seed": rng.randrange(1 << 30)})
    return out


def label_scenario(sc, t_end=30.0, h=1e-3,
                   overpower_setpoint=1.18, period_setpoint=10.0):
    """Simulate and attach ground truth: trace + should_trip + true time."""
    times, powers, _, steps = simulate(sc["rho_fn"], t_end, h)
    t_true = true_condition_times(times, powers,
                                  overpower_setpoint=overpower_setpoint,
                                  period_setpoint=period_setpoint)
    sc = dict(sc)
    sc["times"] = times
    sc["powers"] = powers
    sc["steps"] = steps
    sc["should_trip"] = t_true is not None
    sc["true_time"] = t_true
    return sc


def gaussian_noise(seed, sigma):
    """Multiplicative 1 + N(0, sigma) noise stream, seeded."""
    rng = random.Random(seed)

    def f(_k):
        return 1.0 + rng.gauss(0.0, sigma)

    return f
