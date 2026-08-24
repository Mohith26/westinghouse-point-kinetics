import math

from kineticslab import constants as K
from kineticslab.protection import TripSystem, true_condition_times, evaluate
from kineticslab.reactivity import Step, Ramp
from kineticslab.solver import simulate
from kineticslab.scenarios import gaussian_noise

PCM = K.PCM


def _feed(sys, times, powers):
    for t, n in zip(times, powers):
        if sys.update(t, n):
            break
    return sys


def test_overpower_trip_fires_fast():
    fn = Step(400 * PCM, t0=1.0)
    times, powers, _, _ = simulate(fn, 5.0, 1e-3)
    tripped, t_trip, cause = evaluate(times, powers, sample_dt=0.01)
    assert tripped and cause == "overpower"
    t_true = true_condition_times(times, powers)
    assert 0.0 <= t_trip - t_true < 0.1


def test_trip_latches():
    sys = TripSystem()
    ts = [i * 0.01 for i in range(300)]
    ns = [1.5 if t < 1.0 else 0.5 for t in ts]
    _feed(sys, ts, ns)
    assert sys.tripped
    assert sys.update(10.0, 0.1) is True
    assert sys.trip_cause == "overpower"


def test_no_trip_on_flat_power():
    sys = TripSystem()
    ts = [i * 0.01 for i in range(1000)]
    _feed(sys, ts, [1.0] * len(ts))
    assert not sys.tripped


def test_no_trip_on_small_benign_step():
    # this exact shape used to spuriously trip the period channel
    fn = Step(16 * PCM, t0=1.0)
    times, powers, _, _ = simulate(fn, 20.0, 1e-3)
    assert true_condition_times(times, powers) is None
    tripped, _, _ = evaluate(times, powers, sample_dt=0.01)
    assert not tripped


def test_period_trip_on_fast_ramp():
    fn = Ramp(60 * PCM, t0=1.0, t1=6.0)
    times, powers, _, _ = simulate(fn, 8.0, 1e-3)
    tripped, t_trip, cause = evaluate(times, powers, sample_dt=0.01)
    assert tripped and cause in ("short_period", "overpower")
    t_true = true_condition_times(times, powers)
    assert t_true is not None and t_trip - t_true < 2.0


def test_period_comparator_resets_without_confirmation():
    # confirmation set far higher than the transient can sustain: the
    # comparator must arm on a brief fast rise, then disarm via hysteresis
    sys = TripSystem(confirm_samples=100000)
    t = 0.0
    n = 0.5  # start low so the overpower channel stays clear of its setpoint
    for _ in range(20):  # 0.2 s rise at inverse period 1.5 1/s
        t += 0.01
        n *= math.exp(1.5 * 0.01)
        sys.update(t, n)
    armed = sys._pd_state
    for _ in range(600):  # 6 s flat: filtered omega decays below reset
        t += 0.01
        sys.update(t, n)
    assert armed
    assert not sys.tripped
    assert not sys._pd_state
    assert sys._pd_count == 0


def test_true_condition_overpower_time():
    ts = [i * 0.001 for i in range(3000)]
    ns = [1.0 + 0.1 * t for t in ts]  # crosses 1.18 at t = 1.8
    t_true = true_condition_times(ts, ns)
    assert t_true is not None and abs(t_true - 1.8) < 0.01


def test_true_condition_requires_sustained_period():
    # one-sample spike in an otherwise flat trace: not a sustained period
    ts = [i * 0.001 for i in range(2000)]
    ns = [1.0] * len(ts)
    ns[1000] = 1.001
    assert true_condition_times(ts, ns) is None


def test_evaluate_deterministic_with_seeded_noise():
    fn = Step(300 * PCM, t0=1.0)
    times, powers, _, _ = simulate(fn, 5.0, 1e-3)
    r1 = evaluate(times, powers, sample_dt=0.01,
                  noise=gaussian_noise(42, 0.002))
    r2 = evaluate(times, powers, sample_dt=0.01,
                  noise=gaussian_noise(42, 0.002))
    assert r1 == r2


def test_negative_step_never_trips():
    fn = Step(-600 * PCM, t0=1.0)
    times, powers, _, _ = simulate(fn, 20.0, 1e-3)
    assert true_condition_times(times, powers) is None
    tripped, _, _ = evaluate(times, powers, sample_dt=0.01)
    assert not tripped
