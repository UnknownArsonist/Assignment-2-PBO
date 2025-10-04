import sys
import numpy as np
from MMAS import mmas as _mmas
import math

def get_algorithm(name):
    if (name == "EA"):
        return EA
    if name == "MMAS":
        return MMAS
    if name in ("MMAS*", "MMASSTAR", "MMAS_star"):
        return MMASSTAR
    return RLS

def RLS(func, budget = None):
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)
    
    optimum = func.optimum.y
    for r in range(10):
        f_opt = sys.float_info.min
        x_opt = None
        x = np.random.randint(2, size = func.meta_data.n_variables)
        for i in range(budget):
            f = func(x)
            if f >= f_opt:
                f_opt = f
                x_opt = x
                x = x_opt
            if f_opt >= optimum:
                break
            randi = np.random.randint(func.meta_data.n_variables)
            x[randi] = 0 if x[randi] == 1 else 1
        func.reset()
    return f_opt, x_opt

def EA(func, budget = None):
    size = func.meta_data.n_variables
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

    optimum = func.optimum.y
    for r in range(10):
        f_opt = sys.float_info.min
        x_opt = None
        x = np.random.randint(2, size = func.meta_data.n_variables)
        for i in range(budget):
            f = func(x)
            if f >= f_opt:
                f_opt = f
                x_opt = x
                x = x_opt
            if f_opt >= optimum:
                break
            x = np.array([(a if np.random.rand() > (1/size) else (1 if a == 0 else 0)) for a in x])
        func.reset()
    return f_opt, x_opt

def MMAS(func, budget=None, *, runs=10, evaporation_rate=None,
         pheromone_min=1e-3, pheromone_max=1.0, seed=99):
    
    n = func.meta_data.n_variables

    if budget is None:
        budget = int(n * n * 50)

    if evaporation_rate is None:
        evaporation_rate = 1.0 / math.sqrt(n)   # default

    return _mmas(

        func,
        runs=runs,
        budget=budget,
        evaporation_rate=evaporation_rate,
        pheromone_min=pheromone_min,
        pheromone_max=pheromone_max,
        seed=seed,
        strict=False
    )

def MMASSTAR(func, budget=None, *, runs=10, evaporation_rate=None,
             pheromone_min=1e-3, pheromone_max=1.0, seed=99):
    
    n = func.meta_data.n_variables

    if budget is None:
        budget = int(n * n * 50)

    if evaporation_rate is None:
        evaporation_rate = 1.0 / math.sqrt(n)

    return _mmas(
        func,
        runs=runs,
        budget=budget,
        evaporation_rate=evaporation_rate,
        pheromone_min=pheromone_min,
        pheromone_max=pheromone_max,
        seed=seed,
        strict=True
    )
