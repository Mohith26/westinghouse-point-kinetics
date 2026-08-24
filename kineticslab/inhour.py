"""Inhour equation and the exact solution for a reactivity step.

For a constant reactivity rho the inhour equation

    rho = Lambda * omega + sum_i beta_i * omega / (omega + lambda_i)

has 7 real roots (one per exponential mode). The largest root gives the
asymptotic (stable) period T = 1 / omega_0.

For a step from an equilibrium state the full solution is a closed-form sum
of 7 exponentials obtained by Laplace transform:

    n(t) = sum_j P(omega_j) / D'(omega_j) * exp(omega_j t)

with D(s) = s - (rho - beta)/Lambda - sum_i lambda_i beta_i / (Lambda (s + lambda_i))
and  P(s) = 1 + sum_i beta_i / (Lambda (s + lambda_i)).

This is standard textbook material, see Duderstadt and Hamilton, "Nuclear
Reactor Analysis" (1976), Ch. 6.
"""

import math

from . import constants as K


def inhour_rho(omega, beta_i=K.GROUP_FRACTIONS, lam=K.DECAY_CONSTANTS,
               gen_time=K.LAMBDA_GEN):
    """Right hand side of the inhour equation as a function of omega."""
    return gen_time * omega + sum(
        bi * omega / (omega + li) for bi, li in zip(beta_i, lam))


def _bisect(f, lo, hi, iters=200):
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("no sign change in bracket (%g, %g)" % (lo, hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return mid
        if flo * fm < 0.0:
            hi = mid
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def inhour_roots(rho, beta_i=K.GROUP_FRACTIONS, lam=K.DECAY_CONSTANTS,
                 gen_time=K.LAMBDA_GEN):
    """All 7 real roots of the inhour equation, sorted descending.

    Roots are separated by the poles at omega = -lambda_i. rho must satisfy
    rho < beta (below prompt critical) and rho != 0.
    """
    beta = sum(beta_i)
    if rho >= beta:
        raise ValueError("rho >= beta: prompt supercritical not supported here")
    f = lambda w: inhour_rho(w, beta_i, lam, gen_time) - rho
    poles = sorted(lam)  # ascending: 0.0124 .. 3.01
    eps_rel = 1e-13
    roots = []

    # Root 0: in (0, inf) for rho > 0, in (-lambda_min, 0) for rho < 0.
    if rho > 0.0:
        hi = 1.0
        while f(hi) < 0.0:
            hi *= 2.0
        roots.append(_bisect(f, 1e-300, hi))
    elif rho < 0.0:
        lo = -poles[0] * (1.0 - eps_rel)
        roots.append(_bisect(f, lo, -1e-300))
    else:
        raise ValueError("rho == 0 is degenerate (root at omega = 0)")

    # Interior roots: one in each (-lambda_{k+1}, -lambda_k), k = 0..4.
    for k in range(len(poles) - 1):
        lo = -poles[k + 1] * (1.0 - eps_rel)
        hi = -poles[k] * (1.0 + eps_rel)
        roots.append(_bisect(f, lo, hi))
        if rho < 0.0 and k == 0:
            pass  # the k=0 interval still holds exactly one root for rho<0

    # Prompt root: in (-inf, -lambda_max). Near -(beta - rho)/Lambda.
    guess = -(beta - rho) / gen_time
    lo = min(guess * 4.0, -poles[-1] * 10.0)
    hi = -poles[-1] * (1.0 + 1e-9)
    roots.append(_bisect(f, lo, hi))

    roots = sorted(set(roots), reverse=True)
    if len(roots) != 1 + len(lam):
        raise RuntimeError("expected %d inhour roots, found %d"
                           % (1 + len(lam), len(roots)))
    return roots


def asymptotic_period_from_inhour(rho, **kw):
    """Stable reactor period T = 1/omega_0 from the largest inhour root."""
    return 1.0 / inhour_roots(rho, **kw)[0]


def prompt_jump_ratio(rho, beta=K.BETA_TOTAL):
    """Closed-form prompt jump approximation n(0+)/n(0-) = beta/(beta - rho)."""
    return beta / (beta - rho)


def step_solution_modes(rho, beta_i=K.GROUP_FRACTIONS, lam=K.DECAY_CONSTANTS,
                        gen_time=K.LAMBDA_GEN):
    """(omega_j, A_j) pairs of the exact step response from equilibrium n0=1."""
    beta = sum(beta_i)
    roots = inhour_roots(rho, beta_i, lam, gen_time)
    modes = []
    for w in roots:
        p = 1.0 + sum(bi / (gen_time * (w + li))
                      for bi, li in zip(beta_i, lam))
        dp = 1.0 + sum(li * bi / (gen_time * (w + li) ** 2)
                       for bi, li in zip(beta_i, lam))
        modes.append((w, p / dp))
    return modes


def step_solution(rho, t, **kw):
    """Exact n(t) for a reactivity step at t=0 from equilibrium n0=1."""
    modes = step_solution_modes(rho, **kw)
    return sum(a * math.exp(w * t) for w, a in modes)
