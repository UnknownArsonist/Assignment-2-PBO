import ioh
import pandas as pd
from ACO import get_algorithm

# Get ACO function
ACO = get_algorithm("ACO")

# Functions from Exercise 2
functions = [1, 2, 3, 18, 23, 24, 25]
dimension = 100
budget = 100000  # as required by Exercise 2

# Collect results
records = []

for fid in functions:
    # Load problem from IOHexperimenter
    problem = ioh.get_problem(fid, dimension=dimension, instance=1)

    # Run ACO (internally already does 10 runs)
    f_best, x_best = ACO(problem, budget=budget)

    # Save record
    records.append({
        "Function": f"F{fid}",
        "BestFitness": f_best,
        "BestSolution": "".join(map(str, x_best))
    })

    print(f"[Done] Function F{fid}, best fitness = {f_best}")

# Save results to CSV for later analysis in IOHanalyzer
df = pd.DataFrame(records)
df.to_csv("ACO_results.csv", index=False)

print("All runs finished. Results saved to ACO_results.csv")
