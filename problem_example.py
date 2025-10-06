from ioh import get_problem, ProblemClass, logger
from GA_Algorithms import get_algorithm
import sys
import numpy as np

# Please replace this `random search` by your `genetic algorithm`.
def random_search(func, budget = None):
    # budget of each run: 50n^2
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y
    print(optimum)
    # 10 independent runs for each algorithm on each problem.
    for r in range(10):
        f_opt = sys.float_info.min
        x_opt = None
        for i in range(budget):
            x = np.random.randint(2, size = func.meta_data.n_variables)
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
        func.reset()
    return f_opt, x_opt


ALGORITHM = sys.argv[1]
PROBLEM_IDS = [1, 2, 3, 18, 23, 24, 25]
N = 100
BUDGET = 100000 
FOLDER = f"run-{ALGORITHM}"
ROOT = "TEST"

def main():
    print(f"Algorithm: {ALGORITHM}")
    print(f"Problems: {PROBLEM_IDS}")
    print(f"n: {N}")
    print(f"Output: {ROOT}/{FOLDER}")

    alg_fn = get_algorithm(ALGORITHM) # RLS, EA, MMAS, or MMAS*

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
