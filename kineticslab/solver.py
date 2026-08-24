"""Own implicit solver for the 6-group point kinetics equations.

State y = [n, c1..c6] with

    dn/dt   = ((rho(t) - beta) / Lambda) n + sum_i lambda_i c_i
    dci/dt  = (beta_i / Lambda) n - lambda_i c_i

This is a stiff linear time-varying ODE system (the prompt time constant is
Lambda / (beta - rho), around 15 ms here, while the slowest precursor group
has a 56 s mean life). I integrate it with the trapezoidal rule
(Crank-Nicolson), which is A-stable and second order:

    (I - h/2 A(t+h)) y_{n+1} = (I + h/2 A(t)) y_n

Because the system is linear in y for a given reactivity input, each step is
one 7x7 linear solve, no Newton iteration needed. Pure Python by design.
"""

from . import constants as K
from .linalg import solve, matvec


def kinetics_matrix(rho,
                    beta_i=K.GROUP_FRACTIONS,
                    lam=K.DECAY_CONSTANTS,
                    gen_time=K.LAMBDA_GEN):
    """Build the 7x7 point kinetics matrix A(rho)."""
    beta = sum(beta_i)
    n = 1 + len(beta_i)
    a = [[0.0] * n for _ in range(n)]
    a[0][0] = (rho - beta) / gen_time
    for i in range(len(beta_i)):
        a[0][1 + i] = lam[i]
        a[1 + i][0] = beta_i[i] / gen_time
        a[1 + i][1 + i] = -lam[i]
    return a


def equilibrium_state(n0=1.0,
                      beta_i=K.GROUP_FRACTIONS,
                      lam=K.DECAY_CONSTANTS,
                      gen_time=K.LAMBDA_GEN):
    """Critical steady state at power n0: c_i = beta_i n0 / (Lambda lambda_i)."""
    y = [n0]
    for bi, li in zip(beta_i, lam):
        y.append(bi * n0 / (gen_time * li))
    return y


def step_cn(y, t, h, rho_fn, gen_time=K.LAMBDA_GEN,
            beta_i=K.GROUP_FRACTIONS, lam=K.DECAY_CONSTANTS):
    """One Crank-Nicolson step from t to t+h."""
    a0 = kinetics_matrix(rho_fn(t), beta_i, lam, gen_time)
    a1 = kinetics_matrix(rho_fn(t + h), beta_i, lam, gen_time)
    n = len(y)
    # rhs = (I + h/2 A0) y
    rhs = matvec(a0, y)
    rhs = [y[i] + 0.5 * h * rhs[i] for i in range(n)]
    # lhs matrix = I - h/2 A1
    m = [[(1.0 if i == j else 0.0) - 0.5 * h * a1[i][j] for j in range(n)]
         for i in range(n)]
    return solve(m, rhs)


def simulate(rho_fn, t_end, h, y0=None, t0=0.0, record_every=1,
             gen_time=K.LAMBDA_GEN, beta_i=K.GROUP_FRACTIONS,
             lam=K.DECAY_CONSTANTS):
    """Fixed-step Crank-Nicolson integration.

    Returns (times, powers, y_final, n_steps). `record_every` thins the
    stored trajectory; the final state is always exact for t_end (the last
    step is shortened if t_end is not a multiple of h).
    """
    if y0 is None:
        y0 = equilibrium_state(1.0, beta_i, lam, gen_time)
    y = y0[:]
    t = t0
    times = [t]
    powers = [y[0]]
    steps = 0
    eps = 1e-12 * max(1.0, abs(t_end))
    while t < t_end - eps:
        hh = min(h, t_end - t)
        y = step_cn(y, t, hh, rho_fn, gen_time, beta_i, lam)
        t += hh
        steps += 1
        if steps % record_every == 0 or t >= t_end - eps:
            times.append(t)
            powers.append(y[0])
    return times, powers, y, steps


def asymptotic_period(times, powers, tail_frac=0.2):
    """Estimate the asymptotic period from the tail of a power trace.

    Least-squares slope of ln(n) over the last `tail_frac` of the trace:
    T = 1 / omega. Returns None when power is not strictly positive.
    """
    import math
    k0 = int(len(times) * (1.0 - tail_frac))
    ts = times[k0:]
    ns = powers[k0:]
    if any(p <= 0.0 for p in ns):
        return None
    xs = ts
    ys = [math.log(p) for p in ns]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0.0:
        return None
    omega = num / den
    if omega == 0.0:
        return float("inf")
    return 1.0 / omega
