from kineticslab import constants as K


def test_relative_yields_sum_to_one():
    assert abs(sum(K.RELATIVE_YIELDS) - 1.0) < 1e-12


def test_group_fractions_sum_to_beta():
    assert abs(sum(K.GROUP_FRACTIONS) - K.BETA_TOTAL) < 1e-15


def test_decay_constants_positive_and_ascending():
    lam = K.DECAY_CONSTANTS
    assert all(x > 0 for x in lam)
    assert list(lam) == sorted(lam)


def test_check_consistency_small():
    assert K.check_consistency() < 1e-12


def test_citation_present():
    assert "Duderstadt" in K.__doc__ and "Keepin" in K.__doc__
