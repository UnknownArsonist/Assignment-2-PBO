from ioh import get_problem, ProblemClass, logger
from GA_Algorithms import get_algorithm
import sys

ALGORITHM = sys.argv[1]
PROBLEM_IDS = [1, 2, 3, 18, 23, 24, 25]
N = 100
BUDGET = 100000 
FOLDER = f"run-{ALGORITHM}"
ROOT = "DATA"

def main():
    print(f"Algorithm: {ALGORITHM}")
    print(f"Problems: {PROBLEM_IDS}")
    print(f"n: {N}")
    print(f"Output: {ROOT}/{FOLDER}")

    alg_fn = get_algorithm(ALGORITHM) # rand, RLS, EA, MMAS, or MMAS*

    alg_logger = logger.Analyzer(
        root=ROOT,
        folder_name=FOLDER,
        algorithm_name=ALGORITHM,
        algorithm_info=f"{ALGORITHM} algorithm",
    )

    # build problems
    problems = [get_problem(fid=pid, dimension=N, instance=1, problem_class=ProblemClass.PBO)
                for pid in PROBLEM_IDS]

    # run the chosen algorithm (each algorithm handles its own runs internally)
    for p in problems:
        p.attach_logger(alg_logger)
        n = p.meta_data.n_variables
        budget = BUDGET
        print(f"Running {ALGORITHM} on problem {p.meta_data.problem_id} (n={n}) | budget={budget}")
        alg_fn(p, budget=BUDGET)   # RLS/EA/MMAS/MMAS* already loop 10 runs inside and reset per run

    del alg_logger
    print("\nDone")

if __name__ == "__main__":
    main()
