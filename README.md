# KineticsLab

A small reactor point kinetics playground I built to really understand stiff
ODEs in a setting where the stiffness has physical meaning. It solves the
standard 6-group point kinetics equations with my own implicit integrator
(no numpy in the solver), checks itself against closed-form analytical
solutions and an independent scipy oracle, and layers a toy protection
system (overpower and short-period trips) on top, evaluated over a matrix
of 120 seeded labeled transients.

Everything here is textbook physics. The delayed neutron data is the
classic Keepin 6-group set for thermal fission of U-235, as tabulated in
Duderstadt and Hamilton, "Nuclear Reactor Analysis" (Wiley, 1976), Table
6-2, and the generation time is a generic thermal-lattice textbook value
(1e-4 s). There is no proprietary data and no real plant model anywhere in
this repo; it is an educational model.

## Why this problem is fun

Point kinetics is a 7-dimensional linear ODE system whose eigenvalues span
five orders of magnitude: the prompt neutron time constant is around 15 ms
while the slowest precursor group has a 56 s mean life. Explicit
integrators either crawl or explode on it. At the same time the model is
simple enough that you can write down exact answers: the inhour equation
gives the asymptotic period, and a Laplace transform gives the full step
response as a sum of 7 exponentials. That combination (genuinely stiff, yet
exactly solvable) makes it a perfect test bed for building and honestly
validating your own solver.

## What is inside

- `kineticslab/solver.py`: Crank-Nicolson (trapezoidal) integrator. The
  system is linear in the state, so each step is a single 7x7 linear solve
  with partial pivoting (`linalg.py`), no Newton iteration. A-stable, second
  order, pure Python.
- `kineticslab/inhour.py`: all 7 roots of the inhour equation by bracketed
  bisection between the poles, the asymptotic period, the prompt jump
  closed form, and the exact 7-exponential step response via residues.
- `kineticslab/reactivity.py`: steps, ramps, piecewise reactivity inputs.
- `kineticslab/protection.py`: latching overpower trip (setpoint 1.18 of
  nominal, hysteresis reset at 1.10) and a short-period trip channel: a
  0.1 s windowed inverse-period estimate, clamped, low-pass filtered
  (tau = 1 s), with a 0.2 s confirmation time and hysteresis. The filtering
  is what lets the channel ride through the prompt jump of a small benign
  reactivity step without spurious trips; getting this to a clean 0 false /
  0 missed matrix took actual tuning, see RESULTS.md.
- `kineticslab/scenarios.py`: seeded transient generator. Labels come from
  the simulated ground truth trace (did power actually exceed the setpoint,
  was a short period actually sustained), not from the construction recipe,
  which keeps borderline cases honest.
- `scripts/run_validation.py`, `run_oracle.py`, `run_trips.py`,
  `run_bench.py`: everything measured lands in `results/*.json`.

numpy and scipy appear only in the oracle script and the oracle tests. The
simulator itself is dependency free on purpose.

## Headline numbers (Apple silicon, single thread, Python 3.9)

- Zero-reactivity steady state stays flat to 1.3e-12 over 100 s.
- Simulated asymptotic period matches the inhour equation to 7.8e-7
  relative or better across +50, +100, +200, -100, -300 pcm steps.
- Prompt jump for a +100 pcm step matches the closed form beta/(beta-rho)
  to 0.29 percent (that gap is the prompt jump approximation itself; the
  simulator matches the exact 7-exponential solution to about 1e-7).
- Max relative deviation vs scipy LSODA and BDF (rtol 1e-11) across a
  7-scenario suite: 6.6e-10.
- Trip matrix, 120 seeded labeled transients, clean and 0.2 percent noisy
  passes: 0 false trips, 0 missed trips. Median trip response 11 ms.
- Throughput: about 42,000 steps/s; at a 1 ms step that is roughly 42x
  faster than real time.

Reproduce commands and the full tables are in RESULTS.md.

## Running it

```
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install numpy scipy pytest pytest-cov
.venv/bin/python scripts/run_validation.py
.venv/bin/python scripts/run_oracle.py
.venv/bin/python scripts/run_trips.py
.venv/bin/python scripts/run_bench.py
.venv/bin/python -m pytest tests/ -q
```

## Limitations

- This is point kinetics: one lumped neutron population, no spatial flux
  shape, no thermal hydraulics, no temperature or void feedback. Reactivity
  is a pure external input, there is no control rod model.
- The solver is fixed step. That is fine here (Crank-Nicolson is A-stable
  and the runs are short) but an adaptive controller would be the natural
  next step for long transients with rare fast events.
- Prompt supercritical cases (rho >= beta) are outside the validated range;
  the inhour machinery refuses them and I never push the solver there.
- The protection layer is a toy signal-processing exercise, not a safety
  design. Setpoints, filter constants and confirmation times were tuned on
  the same scenario families they are evaluated on; a fair external test
  would use held-out scenario classes.
- The trip evaluation samples power at 100 Hz with idealized (or mildly
  noisy multiplicative gaussian) measurements. Real instrumentation has
  dynamics of its own that are not modeled.
- Performance numbers are single-thread pure Python on one machine; treat
  them as relative, not absolute.
