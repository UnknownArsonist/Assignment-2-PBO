from ioh import get_problem, ProblemClass
from ioh import logger
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

# Declaration of problems to be tested.
algorithm_names = ["RLS", "EA"]
problem_ids = [1, 2, 3, 18, 23, 24, 25]

algorithms = {a : get_algorithm(a) for a in algorithm_names}
loggers = {a : logger.Analyzer(
    root = "data",
    folder_name = "run-" + a, 
    algorithm_name = a, 
    algorithm_info = a + " algorithm"
    ) for a in algorithm_names}
problems = [get_problem(fid=id, dimension=100, instance=1, problem_class = ProblemClass.PBO) for id in problem_ids]

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress this folder and upload it to IOHanalyzer

for alg in algorithm_names:
    for p in problems:
        p.attach_logger(loggers[alg])
        algorithms[alg](p, 100000)


# This statemenet is necessary in case data is not flushed yet.
for alg in algorithm_names:
    del loggers[alg]