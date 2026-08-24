from kineticslab.protection import evaluate
from kineticslab.scenarios import (make_scenarios, label_scenario,
                                   gaussian_noise)


def test_scenarios_deterministic():
    a = make_scenarios(30, seed=7)
    b = make_scenarios(30, seed=7)
    assert [s["desc"] for s in a] == [s["desc"] for s in b]
    assert [s["noise_seed"] for s in a] == [s["noise_seed"] for s in b]


def test_scenarios_count():
    assert len(make_scenarios(120, seed=1)) == 120


def test_scenarios_cover_step_and_ramp_shapes():
    descs = " ".join(s["desc"] for s in make_scenarios(60, seed=3))
    assert "step" in descs and "ramp" in descs and "zigzag" in descs


def test_gaussian_noise_seeded():
    f1 = gaussian_noise(99, 0.002)
    f2 = gaussian_noise(99, 0.002)
    assert [f1(i) for i in range(10)] == [f2(i) for i in range(10)]


def test_label_and_trip_small_matrix_no_errors():
    # small end to end matrix: labels from ground truth, detector evaluated
    scenarios = [label_scenario(s, t_end=12.0, h=2e-3)
                 for s in make_scenarios(12, seed=999)]
    assert any(s["should_trip"] for s in scenarios)
    assert any(not s["should_trip"] for s in scenarios)
    missed = false = 0
    for sc in scenarios:
        tripped, _, _ = evaluate(sc["times"], sc["powers"], sample_dt=0.01)
        if sc["should_trip"] and not tripped:
            missed += 1
        if not sc["should_trip"] and tripped:
            false += 1
    assert missed == 0
    assert false == 0
