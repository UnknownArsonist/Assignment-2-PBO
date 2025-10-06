import math
import ioh
import pandas as pd
import numpy as np
from ACO import get_algorithm

# Get ACO function
ACO = get_algorithm("ACO")

# Functions from Exercise 2
FUNCTIONS = [1, 2, 3, 18, 4, 24, 25] #Using F4 instead since F23 is buggy at moment
N_FOR = {1:100, 2:100, 3:100, 18:100, 4:100, 24:400, 25:400}

BUDGET = 100_000
RUNS = 10

rows=[]
for fid in FUNCTIONS:
    n = N_FOR[fid]
    try:
        problem = ioh.get_problem(fid, n, problem_class=ioh.ProblemClass.PBO)
    except Exception as e:
        print(f"[warn] Skipping F{fid} at n={n}: {e}")
        rows.append({
            "Function": f"F{fid}",
            "n": n,
            "budget": BUDGET,
            "runs": RUNS,
            "mean": float('nan'),
            "std": float('nan'),
            "min": float('nan'),
            "max": float('nan'),
            "note": "skipped"
        })
        continue

    bests = []
    for r in range(RUNS):
        f_best, x_best = ACO(problem, budget=BUDGET, num_ants=20, rho=0.1, seed=1234 + r)
        bests.append(f_best)
        problem.reset()

    row = {
        "Function": f"F{fid}" + (" (sub for F23)" if fid == 4 else ""),
        "n": n,
        "budget": BUDGET,
        "runs": RUNS,
        "mean": float(np.mean(bests)),
        "std": float(np.std(bests, ddof=1)),
        "min": float(np.min(bests)),
        "max": float(np.max(bests)),
        "note": "" if fid != 4 else "F23 unavailable on this IOH build; used F4"
    }
    rows.append(row)
    print(f"{row['Function']}: mean={row['mean']:.3f} std={row['std']:.3f} (n={n})")
# Save results to CSV for later analysis in IOHanalyzer
df = pd.DataFrame(rows)
df.to_csv("ACO_results.csv", index=False)

print("All runs finished. Results saved to ACO_results.csv")
