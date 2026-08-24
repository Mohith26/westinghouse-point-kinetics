from kineticslab.reactivity import Step, Ramp, Piecewise, Constant


def test_step_before_and_after():
    fn = Step(1e-3, t0=2.0)
    assert fn(1.9) == 0.0
    assert fn(2.0) == 1e-3
    assert fn(100.0) == 1e-3


def test_ramp_clamps_at_t1():
    fn = Ramp(1e-4, t0=1.0, t1=3.0)
    assert fn(0.5) == 0.0
    assert abs(fn(2.0) - 1e-4) < 1e-18
    assert abs(fn(10.0) - 2e-4) < 1e-18


def test_ramp_unclamped():
    fn = Ramp(2e-5, t0=0.0)
    assert abs(fn(5.0) - 1e-4) < 1e-18


def test_piecewise_interpolates():
    fn = Piecewise([(0.0, 0.0), (2.0, 2e-4)])
    assert abs(fn(1.0) - 1e-4) < 1e-18


def test_piecewise_flat_outside():
    fn = Piecewise([(1.0, 5e-5), (2.0, 1e-4)])
    assert fn(0.0) == 5e-5
    assert fn(3.0) == 1e-4


def test_constant():
    assert Constant(3e-4)(123.0) == 3e-4
