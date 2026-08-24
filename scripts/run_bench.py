"""Throughput benchmark for the pure Python solver. Writes results/bench.json."""

import json
import os
import sys
import time
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kineticslab import constants as K
from kineticslab.reactivity import Ramp
from kineticslab.solver import simulate

PCM = K.PCM

out = {"machine": platform.platform() + " / " + platform.python_version(),
       "note": "single thread, pure Python, no numpy in the hot loop",
       "rows": []}

fn = Ramp(5 * PCM, t0=1.0, t1=21.0)
for h in (1e-3, 1e-4):
    t_end = 60.0 if h == 1e-3 else 20.0
    # warmup
    simulate(fn, 1.0, h)
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        _, _, _, steps = simulate(fn, t_end, h, record_every=10 ** 9)
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
            best_steps = steps
    out["rows"].append({
        "h_s": h, "sim_seconds": t_end, "steps": best_steps,
        "wall_s_best_of_3": best,
        "steps_per_sec": best_steps / best,
        "sim_seconds_per_wall_second": t_end / best,
    })
    print(out["rows"][-1])

path = os.path.join(os.path.dirname(__file__), "..", "results", "bench.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
