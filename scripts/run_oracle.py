"""Validation B: independent scipy stiff-ODE oracle. Writes results/oracle.json.

numpy/scipy are used ONLY here (and in the mirroring tests), never in the
solver itself.
"""

import json
import os
import sys
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.integrate import solve_ivp

from kineticslab import constants as K
from kineticslab.reactivity import Step, Ramp, Piecewise
from kineticslab.solver import simulate, equilibrium_state

PCM = K.PCM
BETA = np.array(K.GROUP_FRACTIONS)
LAM = np.array(K.DECAY_CONSTANTS)
GEN = K.LAMBDA_GEN


def rhs(t, y, rho_fn):
    n = y[0]
    c = y[1:]
    rho = rho_fn(t)
    dn = (rho - BETA.sum()) / GEN * n + float(LAM @ c)
    dc = BETA / GEN * n - LAM * c
    return np.concatenate(([dn], dc))


SCENARIOS = [
    ("step +100 pcm", Step(100 * PCM, t0=0.0), 20.0),
    ("step +300 pcm", Step(300 * PCM, t0=0.0), 10.0),
    ("step -500 pcm", Step(-500 * PCM, t0=0.0), 30.0),
    ("step -100 pcm", Step(-100 * PCM, t0=0.0), 30.0),
    ("ramp +10 pcm/s for 20 s", Ramp(10 * PCM, t0=0.0, t1=20.0), 30.0),
    ("ramp -20 pcm/s for 10 s", Ramp(-20 * PCM, t0=0.0, t1=10.0), 30.0),
    ("zigzag piecewise", Piecewise([(0, 0), (2, 80 * PCM), (5, -120 * PCM),
                                    (8, 40 * PCM), (12, 0)]), 20.0),
]

H = 1e-4
out = {"machine": platform.platform() + " / " + platform.python_version(),
       "solver_h_s": H, "oracle_rtol": 1e-11, "oracle_atol": 1e-13,
       "rows": []}

y0 = np.array(equilibrium_state(1.0))
overall = 0.0
for name, fn, t_end in SCENARIOS:
    # Run my solver once on the full window, then compare at grid-aligned
    # checkpoint times so no interpolation error sneaks in.
    times, powers, _, _ = simulate(fn, t_end, H)
    stride = max(1, len(times) // 40)
    idx = list(range(stride, len(times), stride))
    t_eval = np.array([times[i] for i in idx])
    mine = np.array([powers[i] for i in idx])
    row = {"scenario": name, "t_end_s": t_end, "checkpoints": len(idx)}
    for method in ("LSODA", "BDF"):
        sol = solve_ivp(rhs, (0.0, t_end), y0, method=method, args=(fn,),
                        rtol=1e-11, atol=1e-13, t_eval=t_eval,
                        max_step=0.5, dense_output=False)
        assert sol.success
        worst = float(np.max(np.abs(mine - sol.y[0]) / np.abs(sol.y[0])))
        row["max_rel_dev_" + method] = worst
        overall = max(overall, worst)
    out["rows"].append(row)
    print(row)

out["overall_max_rel_dev"] = overall
path = os.path.join(os.path.dirname(__file__), "..", "results", "oracle.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("overall max rel dev:", overall)
