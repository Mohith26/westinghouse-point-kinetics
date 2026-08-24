"""Validation B mirrors: own solver vs independent scipy stiff integrators.

numpy/scipy are allowed here (oracle only), never inside kineticslab.
"""

import numpy as np
from scipy.integrate import solve_ivp

from kineticslab import constants as K
from kineticslab.reactivity import Step, Ramp, Piecewise
from kineticslab.solver import simulate, equilibrium_state

PCM = K.PCM
BETA = np.array(K.GROUP_FRACTIONS)
LAM = np.array(K.DECAY_CONSTANTS)


def rhs(t, y, fn):
    n, c = y[0], y[1:]
    dn = (fn(t) - BETA.sum()) / K.LAMBDA_GEN * n + float(LAM @ c)
    dc = BETA / K.LAMBDA_GEN * n - LAM * c
    return np.concatenate(([dn], dc))


def oracle_power(fn, t_end, method):
    y0 = np.array(equilibrium_state(1.0))
    sol = solve_ivp(rhs, (0.0, t_end), y0, method=method, args=(fn,),
                    rtol=1e-11, atol=1e-13, max_step=0.5,
                    t_eval=[t_end])
    assert sol.success
    return float(sol.y[0][-1])


def test_step_vs_lsoda():
    fn = Step(100 * PCM, t0=0.0)
    _, _, y, _ = simulate(fn, 5.0, 1e-3)
    ref = oracle_power(fn, 5.0, "LSODA")
    assert abs(y[0] - ref) / ref < 1e-8


def test_negative_step_vs_lsoda():
    fn = Step(-500 * PCM, t0=0.0)
    _, _, y, _ = simulate(fn, 5.0, 1e-3)
    ref = oracle_power(fn, 5.0, "LSODA")
    assert abs(y[0] - ref) / ref < 1e-7


def test_ramp_vs_bdf():
    fn = Ramp(10 * PCM, t0=0.0, t1=4.0)
    _, _, y, _ = simulate(fn, 5.0, 1e-3)
    ref = oracle_power(fn, 5.0, "BDF")
    assert abs(y[0] - ref) / ref < 1e-7


def test_zigzag_vs_lsoda():
    fn = Piecewise([(0, 0), (1, 80 * PCM), (2, -120 * PCM), (4, 0)])
    _, _, y, _ = simulate(fn, 5.0, 1e-3)
    ref = oracle_power(fn, 5.0, "LSODA")
    assert abs(y[0] - ref) / ref < 1e-6


def test_lsoda_and_bdf_agree_with_each_other():
    fn = Step(200 * PCM, t0=0.0)
    a = oracle_power(fn, 3.0, "LSODA")
    b = oracle_power(fn, 3.0, "BDF")
    assert abs(a - b) / a < 1e-8
