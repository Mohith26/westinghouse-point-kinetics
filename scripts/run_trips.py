"""Trip-logic evaluation over seeded labeled transients.

Writes results/trips.json with false-trip and missed-trip rates plus
response-time statistics. Two passes: clean signal and 0.2 percent
multiplicative gaussian measurement noise (seeded), both at a 100 Hz
sampling rate.
"""

import json
import os
import sys
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kineticslab.scenarios import make_scenarios, label_scenario, gaussian_noise
from kineticslab.protection import evaluate

N = 120
SEED = 12345
SAMPLE_DT = 0.01
NOISE_SIGMA = 0.002

scenarios = [label_scenario(s) for s in make_scenarios(N, SEED)]

out = {"machine": platform.platform() + " / " + platform.python_version(),
       "n_scenarios": N, "seed": SEED, "sample_dt_s": SAMPLE_DT,
       "sim_h_s": 1e-3, "t_end_s": 30.0,
       "overpower_setpoint": 1.18, "overpower_reset": 1.10,
       "period_setpoint_s": 10.0, "period_reset_s": 12.0,
       "window_s": 0.10, "omega_clamp": 5.0,
       "filter_tau_s": 1.00, "confirm_samples": 20,
       "true_period_confirm_s": 0.2, "noise_sigma": NOISE_SIGMA}

for label, noise_sigma in (("clean", 0.0), ("noisy", NOISE_SIGMA)):
    false_trips = 0
    missed_trips = 0
    n_should = 0
    n_should_not = 0
    resp_times = []
    per = []
    for sc in scenarios:
        noise = (gaussian_noise(sc["noise_seed"], noise_sigma)
                 if noise_sigma > 0 else None)
        tripped, t_trip, cause = evaluate(sc["times"], sc["powers"],
                                          sample_dt=SAMPLE_DT, noise=noise)
        row = {"index": sc["index"], "desc": sc["desc"],
               "should_trip": sc["should_trip"],
               "true_time_s": sc["true_time"],
               "tripped": tripped, "trip_time_s": t_trip, "cause": cause}
        if sc["should_trip"]:
            n_should += 1
            if not tripped:
                missed_trips += 1
            else:
                rt = t_trip - sc["true_time"]
                row["response_time_s"] = rt
                resp_times.append(rt)
        else:
            n_should_not += 1
            if tripped:
                false_trips += 1
        per.append(row)
    summary = {
        "n_should_trip": n_should,
        "n_should_not_trip": n_should_not,
        "missed_trips": missed_trips,
        "false_trips": false_trips,
        "missed_trip_rate": missed_trips / n_should if n_should else None,
        "false_trip_rate": (false_trips / n_should_not
                            if n_should_not else None),
    }
    if resp_times:
        rs = sorted(resp_times)
        summary["response_time_s"] = {
            "min": rs[0], "median": rs[len(rs) // 2], "max": rs[-1],
            "mean": sum(rs) / len(rs),
        }
    out[label] = {"summary": summary, "scenarios": per}
    print(label, json.dumps(summary, indent=2))

path = os.path.join(os.path.dirname(__file__), "..", "results", "trips.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
