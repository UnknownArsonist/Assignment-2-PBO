import sys
import numpy as np

def get_algorithm(name):
    if (name == "EA"):
        return EA
    return RLS

def RLS(func, budget = None):
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)
    for r in range(10):
        f_opt = sys.float_info.min
        x_opt = None
        x = np.random.randint(2, size = func.meta_data.n_variables)
        for i in range(budget):
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
                x = x_opt
            randi = np.random.randint(func.meta_data.n_variables)
            x[randi] = 0 if x[randi] == 1 else 1
        func.reset()
    return f_opt, x_opt

def EA(func, budget = None):
    size = func.meta_data.n_variables
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)
    for r in range(10):
        f_opt = sys.float_info.min
        x_opt = None
        x = np.random.randint(2, size = func.meta_data.n_variables)
        for i in range(budget):
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
                x = x_opt
            x = np.array([(a if np.random.rand() > (1/size) else (1 if a == 0 else 0)) for a in x])
        func.reset()
    return f_opt, x_opt