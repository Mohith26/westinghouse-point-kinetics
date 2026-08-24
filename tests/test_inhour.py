import math

import pytest

from kineticslab import constants as K
from kineticslab.inhour import (inhour_rho, inhour_roots,
                                asymptotic_period_from_inhour,
                                prompt_jump_ratio, step_solution,
                                step_solution_modes)

PCM = K.PCM


def test_seven_roots_positive_rho():
    assert len(inhour_roots(100 * PCM)) == 7


def test_seven_roots_negative_rho():
    assert len(inhour_roots(-300 * PCM)) == 7


def test_roots_satisfy_inhour_equation():
    rho = 100 * PCM
    for w in inhour_roots(rho):
        assert abs(inhour_rho(w) - rho) < 1e-9


def test_largest_root_sign_matches_rho():
    assert inhour_roots(50 * PCM)[0] > 0
    assert inhour_roots(-50 * PCM)[0] < 0


def test_prompt_root_is_large_negative():
    roots = inhour_roots(100 * PCM)
    assert roots[-1] < -(K.BETA_TOTAL - 100 * PCM) / K.LAMBDA_GEN * 0.5


def test_prompt_critical_raises():
    with pytest.raises(ValueError):
        inhour_roots(K.BETA_TOTAL)


def test_zero_rho_raises():
    with pytest.raises(ValueError):
        inhour_roots(0.0)


def test_step_solution_at_zero_is_one():
    assert abs(step_solution(100 * PCM, 0.0) - 1.0) < 1e-9
    assert abs(step_solution(-500 * PCM, 0.0) - 1.0) < 1e-9


def test_asymptotic_period_plus_100pcm_range():
    T = asymptotic_period_from_inhour(100 * PCM)
    assert 50.0 < T < 60.0


def test_prompt_jump_ratio_value():
    assert abs(prompt_jump_ratio(100 * PCM) - 0.0065 / 0.0055) < 1e-12


def test_slow_mode_sum_close_to_prompt_jump():
    rho = 100 * PCM
    modes = step_solution_modes(rho)
    wp = min(w for w, _ in modes)
    slow = sum(a for w, a in modes if w != wp)
    assert abs(slow - prompt_jump_ratio(rho)) / prompt_jump_ratio(rho) < 0.01


def test_step_solution_monotone_growth_late():
    rho = 100 * PCM
    assert step_solution(rho, 10.0) > step_solution(rho, 5.0) > 1.0
