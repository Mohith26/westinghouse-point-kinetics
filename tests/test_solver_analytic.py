"""Validation A mirrors: own solver vs closed-form step solutions."""

import math

from kineticslab import constants as K
from kineticslab.inhour import (step_solution, asymptotic_period_from_inhour,
                                prompt_jump_ratio)
from kineticslab.reactivity import Step
from kineticslab.solver import simulate, asymptotic_period

PCM = K.PCM


def test_sim_matches_exact_plus_100pcm():
    rho = 100 * PCM
    _, _, y, _ = simulate(Step(rho, t0=0.0), 2.0, 1e-3)
    exact = step_solution(rho, 2.0)
    assert abs(y[0] - exact) / exact < 1e-8


def test_sim_matches_exact_minus_500pcm():
    rho = -500 * PCM
    _, _, y, _ = simulate(Step(rho, t0=0.0), 1.0, 1e-3)
    exact = step_solution(rho, 1.0)
    assert abs(y[0] - exact) / exact < 1e-4


def test_sim_matches_exact_plus_300pcm():
    rho = 300 * PCM
    _, _, y, _ = simulate(Step(rho, t0=0.0), 3.0, 1e-4)
    exact = step_solution(rho, 3.0)
    assert abs(y[0] - exact) / exact < 1e-6


def test_asymptotic_period_matches_inhour():
    rho = 200 * PCM
    T_in = asymptotic_period_from_inhour(rho)
    ts, ns, _, _ = simulate(Step(rho, t0=0.0), 300.0, 5e-3, record_every=50)
    T_sim = asymptotic_period(ts, ns, tail_frac=0.2)
    assert abs(T_sim - T_in) / T_in < 1e-6


def test_prompt_jump_within_one_percent():
    rho = 100 * PCM
    tau_p = K.LAMBDA_GEN / (K.BETA_TOTAL - rho)
    _, _, y, _ = simulate(Step(rho, t0=0.0), 5.0 * tau_p, 1e-5)
    assert abs(y[0] - prompt_jump_ratio(rho)) / prompt_jump_ratio(rho) < 0.01


def test_second_order_convergence():
    rho = 100 * PCM
    exact = step_solution(rho, 5.0)
    errs = []
    for h in (3e-2, 1e-2):
        _, _, y, _ = simulate(Step(rho, t0=0.0), 5.0, h)
        errs.append(abs(y[0] - exact) / exact)
    order = math.log(errs[0] / errs[1]) / math.log(3.0)
    assert 1.7 < order < 2.3


def test_determinism():
    rho = 150 * PCM
    r1 = simulate(Step(rho, t0=0.0), 3.0, 1e-3)
    r2 = simulate(Step(rho, t0=0.0), 3.0, 1e-3)
    assert r1[2] == r2[2] and r1[1] == r2[1]


def test_stiff_large_negative_step_stable_at_coarse_h():
    rho = -800 * PCM
    _, ns, y, _ = simulate(Step(rho, t0=0.0), 10.0, 0.1)
    assert all(p > 0.0 for p in ns)
    exact = step_solution(rho, 10.0)
    assert abs(y[0] - exact) / exact < 1e-2


def test_final_time_exact_when_not_multiple_of_h():
    ts, _, _, _ = simulate(Step(50 * PCM, t0=0.0), 1.05, 0.1)
    assert abs(ts[-1] - 1.05) < 1e-9


def test_power_positive_throughout_large_positive_step():
    _, ns, _, _ = simulate(Step(500 * PCM, t0=0.0), 5.0, 1e-3,
                           record_every=10)
    assert all(p > 0.0 for p in ns)
