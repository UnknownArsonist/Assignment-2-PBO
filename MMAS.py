from ioh import get_problem, ProblemClass, logger
import sys
import numpy as np
import math

# ------------------- EXERCISE 4 -------------------

def construct_solution(pheromones, random_numgen):
    """Construct a bitstring guided by pheromone values (as shown in Fig 1 and Fig 2)"""
    n = pheromones.shape[0]
    solution = np.zeros(n, dtype=int)
    for i in range(n):
        p0, p1 = pheromones[i, 0], pheromones[i, 1]
        total = p0 + p1
        prob_one = 0.5 if total <= 0 else p1 / total
        solution[i] = 1 if random_numgen.random() < prob_one else 0
    return solution


def update_pheromones(pheromones, best_solution, evaporation_rate, pheromone_min, pheromone_max):
    """Updates pheromone values in MMAS (reinforces edges that represent the best-so-far solution)"""
    pheromones *= (1.0 - evaporation_rate)                                             # evaporate
    pheromones[np.arange(len(best_solution)), best_solution] += evaporation_rate       # deposit on chosen edges
    np.clip(pheromones, pheromone_min, pheromone_max, out=pheromones)                             # min and max bounds


def mmas(problem, runs, budget, evaporation_rate, pheromone_min, pheromone_max, seed, strict=False):
    """
    Run the Max–Min Ant System (MMAS) on a PBO problem

    strict=false: MMAS (accepts ties)
    strict=true: MMAS* (strict improvements only)

    Returns:
        (best_value, best_solution): best fitness value found and its bitstring
    """
    random_engine = np.random.default_rng(seed)
    num_bits = problem.meta_data.n_variables

    optimum_value = problem.optimum.y
    best_overall_value, best_overall_solution = -np.inf, None

    for _ in range(runs):
        # Initialisation (Fig 3, step 1 and 2)
        pheromones = np.full((num_bits, 2), 0.5, dtype=float)
        current_best_solution = construct_solution(pheromones, random_engine)
        current_best_value = problem(current_best_solution)

        # Initial update (Fig 3, step 3)
        update_pheromones(pheromones, current_best_solution, evaporation_rate, pheromone_min, pheromone_max)

        evaluations = 1
        # Main loop (Fig 3, step 4)
        while evaluations < budget and current_best_value < optimum_value:
            candidate_solution = construct_solution(pheromones, random_engine)
            candidate_value = problem(candidate_solution)
            evaluations += 1

            # MMAS vs MMAS*
            is_better = (candidate_value > current_best_value) if strict else \
                        (candidate_value >= current_best_value)

            if is_better:
                current_best_solution, current_best_value = candidate_solution, candidate_value

            update_pheromones(pheromones, current_best_solution, evaporation_rate, pheromone_min, pheromone_max)

        # Track best over all runs
        if current_best_value > best_overall_value:
            best_overall_value, best_overall_solution = current_best_value, current_best_solution

        problem.reset()  # fresh run

    return best_overall_value, best_overall_solution


def run_mmas(problem, **kwargs):
    return mmas(problem, strict=False, **kwargs)  # MMAS

def run_mmas_star(problem, **kwargs):
    return mmas(problem, strict=True, **kwargs)   # MMAS*


def run():
    n = 100
    reps = 10
    eval_budget = 100000                                             # budget from exercise 2
    evaporation_rate_values = [1.0, 1.0 / math.sqrt(n), 1.0 / n]     # evap rates 1, 1/sqrt(n), and 1/n (from exercise 4)
    fids = [1, 2, 3, 18, 23, 24, 25]                                 # PBO problems
    instance = 1

    base_logger = logger.Analyzer(
        root="ex4_data",
        folder_name="MMAS_vs_MMASstar",
        algorithm_name="MMAS-MMASstar",
    )

    for fid in fids:
        problem = get_problem(fid=fid, dimension=n, instance=instance, problem_class=ProblemClass.PBO)

        for evaporation_rate in evaporation_rate_values:
            problem.attach_logger(base_logger)

            # MMAS
            print(f"[fid={fid}] MMAS, evaporation_rate={evaporation_rate}")
            run_mmas(problem,
                     runs=reps,
                     budget=eval_budget,
                     evaporation_rate=evaporation_rate,
                     pheromone_min=1e-3,
                     pheromone_max=1.0,
                     seed=99)
            problem.reset()

            # MMAS*
            print(f"[fid={fid}] MMAS*, evaporation_rate={evaporation_rate}")
            problem.attach_logger(base_logger) 
            run_mmas_star(problem,
                          runs=reps,
                          budget=eval_budget,
                          evaporation_rate=evaporation_rate,
                          pheromone_min=1e-3,
                          pheromone_max=1.0,
                          seed=99)
            problem.reset()

    base_logger.close()
    print("Finished!")

if __name__ == "__main__":
    run()