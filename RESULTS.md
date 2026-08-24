# Results notebook

My running notes on what I measured and how to reproduce it. Machine:
Apple silicon Mac, single thread, CPython 3.9.6. Everything below was
produced by the scripts in `scripts/` and is stored raw in `results/`.
Numbers will wiggle slightly on other machines for the timing rows; the
accuracy rows should reproduce bit-for-bit or close to it.

Setup:

```
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install numpy scipy pytest pytest-cov
```

## Validation A: analytical checks

Reproduce: `.venv/bin/python scripts/run_validation.py` (writes
`results/validation.json`, takes a couple of minutes because the negative
step periods need very long runs).

Zero reactivity steady state, 100 s at h = 1 ms: max |n - 1| = 1.30e-12.
Machine epsilon accumulation, nothing else; the equilibrium state is an
exact stationary point of the discrete scheme up to roundoff.

Asymptotic period vs the inhour equation (simulated period fitted from the
log-slope of the trace tail):

| step (pcm) | inhour period (s) | rel error of simulated period |
|-----------:|------------------:|------------------------------:|
| +50 | 136.969 | 7.79e-07 |
| +100 | 54.995 | 2.29e-09 |
| +200 | 17.389 | 1.12e-09 |
| -100 | -129.950 | 3.06e-09 |
| -300 | -89.617 | 1.42e-09 |

The negative steps needed 2500 s of simulated time before the fit is clean:
the asymptotic root sits just above -lambda_1 and the next mode decays
slowly, so the tail stays contaminated for a long while. With only 900 s I
was getting 3e-4 level errors and initially mistook that for solver error.

Prompt jump, +100 pcm step: closed form beta/(beta-rho) = 1.181818.
Simulated power at five prompt time constants (t = 0.0909 s, h = 1e-5)
is 1.185252, a 0.29 percent gap. The slow-mode amplitude sum of the exact
7-exponential solution gives 1.178611, a 0.27 percent gap on the other
side. Both gaps are the prompt jump approximation itself, not the solver:
against the exact step solution the simulator agrees to 1.4e-7 (+100 pcm),
2.4e-7 (-100), 1.7e-7 (+300), 2.5e-6 (-500) max relative error over a
0.01 s to 20 s checkpoint grid at h = 1e-4.

Step size study (+100 pcm, error at t = 10 s vs exact solution):

| h (s) | rel error |
|------:|----------:|
| 0.1 | 4.66e-07 |
| 0.03 | 4.19e-08 |
| 0.01 | 4.66e-09 |
| 0.003 | 4.19e-10 |
| 0.001 | 4.78e-11 |
| 0.0003 | 5.03e-12 |
| 0.0001 | 1.37e-13 |

Observed convergence order between consecutive rows: 2.00, 2.00, 2.00,
1.98, 1.87, then 3.28 on the last pair where the error is hitting the
double precision floor. Clean second order, as trapezoidal should be.

## Validation B: independent scipy oracle

Reproduce: `.venv/bin/python scripts/run_oracle.py` (writes
`results/oracle.json`). My solver at h = 1e-4 vs solve_ivp LSODA and BDF at
rtol 1e-11 / atol 1e-13, compared at 40 grid-aligned checkpoints per
scenario so there is no interpolation error in the comparison.

| scenario | LSODA max rel dev | BDF max rel dev |
|----------|------------------:|----------------:|
| step +100 pcm | 1.99e-11 | 2.82e-11 |
| step +300 pcm | 5.86e-10 | 6.62e-10 |
| step -500 pcm | 6.06e-11 | 8.50e-11 |
| step -100 pcm | 3.60e-11 | 5.53e-11 |
| ramp +10 pcm/s, 20 s | 1.80e-10 | 1.26e-10 |
| ramp -20 pcm/s, 10 s | 4.35e-11 | 6.52e-11 |
| zigzag piecewise | 2.38e-11 | 6.38e-11 |

Overall max relative deviation: 6.62e-10.

## Trip logic matrix

Reproduce: `.venv/bin/python scripts/run_trips.py` (writes
`results/trips.json`). 120 seeded scenarios (seed 12345), simulated at
h = 1 ms for 30 s, labeled from the fine-grid ground truth: 61 should-trip,
59 should-not. Detector samples at 100 Hz. Settings: overpower setpoint
1.18 with reset 1.10, period setpoint 10 s with reset 12 s, 0.1 s
differentiation window, omega clamp 5 1/s, filter tau 1.0 s, 0.2 s (20
sample) confirmation.

Clean signal: 0 false trips, 0 missed trips. Response time from first true
condition violation to latch: min 0.000 s, median 0.011 s, mean 0.167 s,
max 1.253 s. The slow tail is fast ramps caught by the period channel,
which pays about a second of filter lag; large steps trip on overpower
within one or two samples.

Noisy signal (0.2 percent multiplicative gaussian, seeded): still 0 false,
0 missed. Median response 0.011 s, max 1.253 s, but 20 of the 61 should-trip
cases fired up to 0.83 s *before* the nominal crossing time (recorded as
negative response times, mean drops to 0.072 s) because noise pushes a
power signal hovering just under the setpoint over the line early. That is
the conservative direction for a protection channel, but it is worth being
honest that with noise the trip time distribution is two-sided.

For the record, my first channel design (two-sample inverse period, 0.1 s
filter, 3 sample confirmation) produced 5 false trips out of 59 on the
clean pass: the prompt jump of a 10 to 17 pcm benign step leaks through a
fast filter as a fake short period. The windowed estimate plus the 1 s
filter and 0.2 s confirmation is what fixed it, at the price of the period
channel lag noted above.

## Throughput

Reproduce: `.venv/bin/python scripts/run_bench.py` (writes
`results/bench.json`). Best of 3, single thread, trajectory recording off.

| h (s) | steps | wall (s) | steps/s | sim seconds per wall second |
|------:|------:|---------:|--------:|----------------------------:|
| 1e-3 | 60000 | 1.435 | 41823 | 41.8 |
| 1e-4 | 200001 | 4.752 | 42091 | 4.2 |

Each step is two 7x7 matrix builds and one dense solve in pure Python, so
about 42k steps/s felt reasonable; I did not micro-optimize.

## Tests

`.venv/bin/python -m pytest tests/ -q --cov=kineticslab`: 63 passed,
line coverage 95 percent (the misses are error branches and defensive
paths). The suite reruns cut-down versions of both validations, the trip
matrix on a fresh seed, determinism checks, and the stiff coarse-step
stability case.
