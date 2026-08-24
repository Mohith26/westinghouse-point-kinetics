"""Reactor protection style trip logic on a sampled power signal.

Two trip channels, both latching:

* Overpower trip: fires when relative power exceeds `overpower_setpoint`.
  The comparator has hysteresis (`overpower_reset` below the setpoint) so a
  noisy signal hovering at the setpoint does not chatter the comparator.

* Short-period trip (startup rate): fires when the measured inverse period
  omega, estimated over a sliding `window_s` differentiation window, then
  clamped and low-pass filtered, exceeds 1/`period_setpoint` for
  `confirm_samples` consecutive samples. Hysteresis on the inverse period
  comparator as well. The window, clamp, filter and confirmation time are
  what let the channel ride through the prompt jump of a small benign
  reactivity step (a near instantaneous power rise that is not a sustained
  short period) without spurious trips.

Once a channel latches it stays tripped. Response time is measured from the
first sample where the *true* (unfiltered, exact) condition is violated to
the sample where the trip latches.

This is a toy educational model of trip logic, not any real protection
system design.
"""

import math


class TripSystem:
    def __init__(self,
                 overpower_setpoint=1.18,
                 overpower_reset=1.10,
                 period_setpoint=10.0,
                 period_reset=12.0,
                 window_s=0.10,
                 omega_clamp=5.0,
                 filter_tau=1.00,
                 confirm_samples=20):
        self.overpower_setpoint = overpower_setpoint
        self.overpower_reset = overpower_reset
        self.period_setpoint = period_setpoint
        self.period_reset = period_reset
        self.window_s = window_s
        self.omega_clamp = omega_clamp
        self.filter_tau = filter_tau
        self.confirm_samples = confirm_samples
        self.reset()

    def reset(self):
        self.tripped = False
        self.trip_time = None
        self.trip_cause = None
        self._buf = []  # (t, ln n) samples inside the window
        self._prev_t = None
        self._omega_f = 0.0
        self._op_state = False
        self._pd_state = False
        self._pd_count = 0

    def update(self, t, n):
        """Feed one (time, power) sample. Returns True once tripped."""
        if self.tripped:
            return True

        # Overpower comparator with hysteresis, then latch.
        if self._op_state:
            if n < self.overpower_reset:
                self._op_state = False
        else:
            if n > self.overpower_setpoint:
                self._op_state = True
        if self._op_state:
            self._latch(t, "overpower")
            return True

        # Windowed inverse period estimate, clamped, then low-pass filtered.
        if n > 0.0:
            ln_n = math.log(n)
            self._buf.append((t, ln_n))
            while len(self._buf) > 2 and t - self._buf[0][0] > self.window_s:
                self._buf.pop(0)
            if self._prev_t is not None and len(self._buf) >= 2:
                t0, l0 = self._buf[0]
                span = t - t0
                if span > 0.0:
                    omega_raw = (ln_n - l0) / span
                    c = self.omega_clamp
                    omega_raw = max(-c, min(c, omega_raw))
                    dt = t - self._prev_t
                    alpha = dt / (self.filter_tau + dt)
                    self._omega_f += alpha * (omega_raw - self._omega_f)
        else:
            self._buf = []
        self._prev_t = t

        omega_trip = 1.0 / self.period_setpoint
        omega_reset = 1.0 / self.period_reset
        if self._pd_state:
            if self._omega_f < omega_reset:
                self._pd_state = False
                self._pd_count = 0
        else:
            if self._omega_f > omega_trip:
                self._pd_state = True
        if self._pd_state:
            self._pd_count += 1
            if self._pd_count >= self.confirm_samples:
                self._latch(t, "short_period")
                return True
        return False

    def _latch(self, t, cause):
        self.tripped = True
        self.trip_time = t
        self.trip_cause = cause


def true_condition_times(times, powers,
                         overpower_setpoint=1.18,
                         period_setpoint=10.0,
                         period_confirm=0.2):
    """Ground-truth first violation time on a fine trace, or None.

    Overpower: first time power exceeds the setpoint. Period: first time the
    exact two-point inverse period stays above 1/period_setpoint for at
    least `period_confirm` seconds continuously. Ignoring the instantaneous
    prompt jump spike is deliberate: the design basis here defines a short
    period condition as a *sustained* fast period, matching the trip
    channel's confirmation time, not a single fine-grid integration step.
    """
    op_t = None
    for t, n in zip(times, powers):
        if n > overpower_setpoint:
            op_t = t
            break
    pd_t = None
    run_start = None
    thr = 1.0 / period_setpoint
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0.0 or powers[i] <= 0.0 or powers[i - 1] <= 0.0:
            run_start = None
            continue
        omega = math.log(powers[i] / powers[i - 1]) / dt
        if omega > thr:
            if run_start is None:
                run_start = times[i - 1]
            if times[i] - run_start >= period_confirm:
                pd_t = run_start
                break
        else:
            run_start = None
    cands = [x for x in (op_t, pd_t) if x is not None]
    return min(cands) if cands else None


def evaluate(times, powers, sample_dt=0.01, noise=None, **trip_kwargs):
    """Run the trip system over a trace sampled every `sample_dt`.

    `noise` is an optional callable(index) returning a multiplicative factor
    applied to the sampled power (used for seeded measurement noise).
    Returns (tripped, trip_time, trip_cause).
    """
    sys = TripSystem(**trip_kwargs)
    next_t = times[0]
    k = 0
    for i, (t, n) in enumerate(zip(times, powers)):
        if t + 1e-12 < next_t:
            continue
        v = n
        if noise is not None:
            v *= noise(k)
        k += 1
        if sys.update(t, v):
            break
        next_t = t + sample_dt
    return sys.tripped, sys.trip_time, sys.trip_cause
