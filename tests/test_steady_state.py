from kineticslab import constants as K
from kineticslab.linalg import matvec
from kineticslab.reactivity import Constant
from kineticslab.solver import (kinetics_matrix, equilibrium_state, simulate)


def test_equilibrium_precursor_formula():
    y = equilibrium_state(2.0)
    for i, (bi, li) in enumerate(zip(K.GROUP_FRACTIONS, K.DECAY_CONSTANTS)):
        assert abs(y[1 + i] - bi * 2.0 / (K.LAMBDA_GEN * li)) < 1e-12


def test_equilibrium_is_stationary_point():
    a = kinetics_matrix(0.0)
    dy = matvec(a, equilibrium_state(1.0))
    assert max(abs(v) for v in dy) < 1e-10


def test_zero_reactivity_stays_flat_1e10():
    ts, ns, _, _ = simulate(Constant(0.0), 20.0, 1e-3, record_every=100)
    assert max(abs(p - 1.0) for p in ns) < 1e-10


def test_zero_reactivity_precursors_flat():
    y0 = equilibrium_state(1.0)
    _, _, y, _ = simulate(Constant(0.0), 5.0, 1e-3)
    assert max(abs(a - b) / b for a, b in zip(y, y0)) < 1e-10
