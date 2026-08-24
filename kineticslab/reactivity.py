"""Reactivity input models: steps, ramps, and piecewise schedules.

Reactivity is treated purely as an external input rho(t) in absolute units
(1 pcm = 1e-5). There is no control rod or feedback model here, that is out
of scope by design.
"""


class Step:
    """rho(t) = rho0 for t < t0, rho1 for t >= t0."""

    def __init__(self, rho1, t0=0.0, rho0=0.0):
        self.rho0 = rho0
        self.rho1 = rho1
        self.t0 = t0

    def __call__(self, t):
        return self.rho1 if t >= self.t0 else self.rho0


class Ramp:
    """Linear ramp from rho0 starting at t0 with the given rate, optionally
    clamped after t1."""

    def __init__(self, rate, t0=0.0, rho0=0.0, t1=None):
        self.rate = rate
        self.t0 = t0
        self.rho0 = rho0
        self.t1 = t1

    def __call__(self, t):
        if t < self.t0:
            return self.rho0
        t_eff = t if self.t1 is None else min(t, self.t1)
        return self.rho0 + self.rate * (t_eff - self.t0)


class Piecewise:
    """Piecewise linear reactivity through (t, rho) knots, held flat outside."""

    def __init__(self, knots):
        self.knots = sorted(knots)
        if len(self.knots) < 1:
            raise ValueError("need at least one knot")

    def __call__(self, t):
        k = self.knots
        if t <= k[0][0]:
            return k[0][1]
        if t >= k[-1][0]:
            return k[-1][1]
        for i in range(len(k) - 1):
            t0, r0 = k[i]
            t1, r1 = k[i + 1]
            if t0 <= t <= t1:
                if t1 == t0:
                    return r1
                w = (t - t0) / (t1 - t0)
                return r0 + w * (r1 - r0)
        return k[-1][1]


class Constant:
    def __init__(self, rho):
        self.rho = rho

    def __call__(self, t):
        return self.rho
