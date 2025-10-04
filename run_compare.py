import ioh, numpy as np, pandas as pd
from ACO import get_algorithm as get_aco
from GA_Algorithms import get_algorithm as get_ga

ACO = get_aco("ACO")
MMAS = get_ga("MMAS")
MMASSTAR = get_ga("MMAS*")

FUNCTIONS = [1,2,3,18,4,24,25]#F4 is used to replace F23 since being buggy
DIMENSION = 100
BUDGET = 100_000
RUNS = 10

def eval_many(alg, problem, runs, budget):
    vals = []
    for r in range(runs):
        f_best, _ = alg(problem, budget=budget)
        vals.append(f_best)
        problem.reset()
    return np.array(vals)

rows = []
for fid in FUNCTIONS:
    prob = ioh.get_problem(fid, DIMENSION, problem_class=ioh.ProblemClass.PBO)

    aco_vals = eval_many(lambda f, budget=None: ACO(f, budget=budget, num_ants=20, rho=0.1), prob, RUNS, BUDGET)
    mmas_vals = eval_many(lambda f, budget=None: MMAS(f, runs=1, budget=BUDGET, evaporation_rate=0.1, pheromone_min=1e-3, pheromone_max=1.0, seed=42), prob, RUNS, BUDGET)
    mmasstar_vals = eval_many(lambda f, budget=None: MMASSTAR(f, runs=1, budget=BUDGET, evaporation_rate=0.1, pheromone_min=1e-3, pheromone_max=1.0, seed=99), prob, RUNS, BUDGET)

    for name, arr in [("ACO", aco_vals), ("MMAS", mmas_vals), ("MMAS*", mmasstar_vals)]:
        rows.append({
            "Function": f"F{fid}",
            "Alg": name,
            "n": DIMENSION,
            "budget": BUDGET,
            "runs": RUNS,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    print(f"F{fid} done.")

pd.DataFrame(rows).to_csv("compare_ACO_MMAS.csv", index=False)
print("Saved compare_ACO_MMAS.csv")
