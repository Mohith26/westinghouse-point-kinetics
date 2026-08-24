"""Validation A: analytical checks. Writes results/validation.json."""

import json
import math
import os
import sys
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kineticslab import constants as K
from kineticslab import inhour
from kineticslab.reactivity import Step, Constant
from kineticslab.solver import simulate, asymptotic_period

PCM = K.PCM
out = {"machine": platform.platform() + " / " + platform.python_version(),
       "note": "single thread, pure Python solver"}

# 1. Zero reactivity steady state over 100 s.
ts, ns, _, _ = simulate(Constant(0.0), 100.0, 1e-3, record_every=100)
out["steady_state"] = {
    "t_end_s": 100.0, "h_s": 1e-3,
    "max_abs_dev_from_1": max(abs(p - 1.0) for p in ns),
}

# 2. Asymptotic period vs inhour equation for several step sizes.
period_rows = []
for pcm in (50, 100, 200, -100, -300):
    rho = pcm * PCM
    T_in = inhour.asymptotic_period_from_inhour(rho)
    # long enough that transient modes have died; negative steps need much
    # longer because the asymptotic root sits close to the next mode
    t_end = 600.0 if pcm > 0 else 2500.0
    h_here = 2e-3 if pcm > 0 else 5e-3
    ts, ns, _, _ = simulate(Step(rho, t0=0.0), t_end, h_here, record_every=50)
    T_sim = asymptotic_period(ts, ns, tail_frac=0.2)
    period_rows.append({
        "rho_pcm": pcm, "period_inhour_s": T_in, "period_sim_s": T_sim,
        "rel_error": abs(T_sim - T_in) / abs(T_in),
    })
out["asymptotic_period"] = period_rows

# 3. Prompt jump for +100 pcm.
rho = 100 * PCM
pj_closed = inhour.prompt_jump_ratio(rho)
modes = inhour.step_solution_modes(rho)
prompt_root = min(modes, key=lambda m: m[0])
pj_modes = sum(a for w, a in modes if w != prompt_root[0])
tau_p = K.LAMBDA_GEN / (K.BETA_TOTAL - rho)
t_star = 5.0 * tau_p
ts, ns, _, _ = simulate(Step(rho, t0=0.0), t_star, 1e-5)
pj_sim = ns[-1]
out["prompt_jump"] = {
    "rho_pcm": 100,
    "closed_form": pj_closed,
    "exact_slow_mode_sum_at_t0": pj_modes,
    "rel_error_modes_vs_closed": abs(pj_modes - pj_closed) / pj_closed,
    "sim_at_5_prompt_tau": pj_sim,
    "t_star_s": t_star,
    "rel_error_sim_vs_closed": abs(pj_sim - pj_closed) / pj_closed,
}

# 4. Sim vs exact analytic step solution, max rel error over a time grid.
an_rows = []
for pcm in (100, -100, 300, -500):
    rho = pcm * PCM
    t_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    worst = 0.0
    for t in t_grid:
        na = inhour.step_solution(rho, t)
        _, ns2, y2, _ = simulate(Step(rho, t0=0.0), t, 1e-4)
        worst = max(worst, abs(y2[0] - na) / abs(na))
    an_rows.append({"rho_pcm": pcm, "h_s": 1e-4,
                    "max_rel_error_vs_exact": worst})
out["exact_step_solution"] = an_rows

# 5. Step-size study: error at t=10 s for +100 pcm vs exact, order estimate.
rho = 100 * PCM
exact = inhour.step_solution(rho, 10.0)
rows = []
for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
    _, _, y, steps = simulate(Step(rho, t0=0.0), 10.0, h)
    rows.append({"h_s": h, "steps": steps,
                 "rel_error_at_10s": abs(y[0] - exact) / exact})
orders = []
for a, b in zip(rows, rows[1:]):
    if b["rel_error_at_10s"] > 0:
        orders.append(math.log(a["rel_error_at_10s"] / b["rel_error_at_10s"]) /
                      math.log(a["h_s"] / b["h_s"]))
out["step_size_study"] = {"rho_pcm": 100, "t_eval_s": 10.0, "rows": rows,
                          "observed_orders": orders}

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"),
            exist_ok=True)
path = os.path.join(os.path.dirname(__file__), "..", "results",
                    "validation.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
