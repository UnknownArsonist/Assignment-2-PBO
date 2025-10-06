import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120})

try:
    import ioh
except Exception as e:
    raise RuntimeError("Install IOHexperimenter with `pip install ioh`.") from e

GA_MODULE = None
GA_ALGOS = None
YOUR_GA = None

try:
    import GA_Algorithms as GA_ALGOS
except Exception:
    GA_ALGOS = None

try:
    import uniform_ga_mut as GA_MODULE
    if hasattr(GA_MODULE, "run_ga_once"):
        YOUR_GA = GA_MODULE.run_ga_once
    else:
        YOUR_GA = None
except Exception:
    GA_MODULE = None
    YOUR_GA = None

FUNCTIONS = [1, 2, 3, 18, 23, 24, 25]
DIM = 100
BUDGET = 100_000
RUNS = 10
RNG_SEED = 42

def valid_function(fid: int) -> int:
    try:
        _ = ioh.get_problem(fid, 1, DIM, ioh.ProblemClass.PBO)
        return fid
    except Exception:
        return 4 if fid == 23 else fid

FUNCTIONS = [valid_function(fid) for fid in FUNCTIONS]

def record_curve(problem, algorithm_callable, budget, **kwargs):
    problem.reset()
    best_curve = np.empty(budget, dtype=float)
    best = -np.inf
    evals = 0

    def evaluate(x):
        nonlocal evals, best
        y = problem(x)
        if y > best:
            best = y
        if evals < budget:
            best_curve[evals] = best
        evals += 1
        return y

    n = problem.meta_data.n_variables
    algorithm_callable(problem=problem, evaluate=evaluate, budget=budget, n=n, **kwargs)
    if evals < budget:
        best_curve[evals:] = best
    return best_curve

def rs_once(problem=None, evaluate=None, budget=None, n=None, **kwargs):
    for _ in range(budget):
        x = np.random.randint(0, 2, size=n).astype(int)
        evaluate(x)

def rls_once(problem=None, evaluate=None, budget=None, n=None, **kwargs):
    x = np.random.randint(0, 2, size=n).astype(int)
    fx = evaluate(x)
    for _ in range(budget - 1):
        i = np.random.randint(n)
        y = x.copy()
        y[i] ^= 1
        fy = evaluate(y)
        if fy >= fx:
            x, fx = y, fy

def ea11_once(problem=None, evaluate=None, budget=None, n=None, **kwargs):
    p = 1.0 / n
    x = np.random.randint(0, 2, size=n).astype(int)
    fx = evaluate(x)
    for _ in range(budget - 1):
        y = x.copy()
        flips = np.random.rand(n) < p
        y[flips] ^= 1
        fy = evaluate(y)
        if fy >= fx:
            x, fx = y, fy

def ga_fallback_once(problem=None, evaluate=None, budget=None, n=None, mu=20, lam=20, pc=0.8, pm=None, **kwargs):
    if pm is None:
        pm = 1.0 / n
    P = np.random.randint(0, 2, size=(mu, n)).astype(int)
    fitness = np.array([evaluate(ind.copy()) for ind in P])
    t = mu
    while t < budget:
        off = []
        for _ in range(lam):
            a, b = np.random.randint(0, len(P), size=2)
            p1, p2 = P[a], P[b]
            if np.random.rand() < pc:
                mask = np.random.randint(0, 2, size=n).astype(bool)
                child = np.where(mask, p1, p2).copy()
            else:
                child = p1.copy() if np.random.rand() < 0.5 else p2.copy()
            flips = np.random.rand(n) < pm
            child[flips] ^= 1
            off.append(child)
        off = np.array(off)
        off_fit = []
        for child in off:
            if t >= budget:
                break
            off_fit.append(evaluate(child.copy()))
            t += 1
        P_all = np.vstack([P, off])
        F_all = np.hstack([fitness, off_fit])
        keep = np.argsort(F_all)[-len(P):]
        P, fitness = P_all[keep], F_all[keep]
        if t >= budget:
            break

def get_random_search():
    return rs_once

def get_rls():
    return rls_once

def get_ea11():
    return ea11_once

def get_your_ga():
    if YOUR_GA:
        return lambda problem=None, evaluate=None, budget=None, n=None, **kwargs: YOUR_GA(
            problem=problem, evaluate=evaluate, budget=budget, n=n, **kwargs
        )
    return ga_fallback_once

DEFAULT_GA_KWARGS = dict(mu=30, lam=30, pc=0.9, pm=None)

def eval_many(algo, problem_id, dim, budget, runs, rng_seed, **kwargs):
    np.random.seed(rng_seed)
    curves = []
    for r in range(runs):
        np.random.seed(rng_seed + r + 1337)
        problem = ioh.get_problem(problem_id, 1, dim, ioh.ProblemClass.PBO)
        curve = record_curve(problem, algo, budget, **kwargs)
        curves.append(curve)
    return np.vstack(curves)

def main():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    algos = {
        "Random": get_random_search(),
        "RLS": get_rls(),
        "(1+1) EA": get_ea11(),
        "Your GA": get_your_ga(),
    }
    for fid in FUNCTIONS:
        results = {}
        for name, algo in algos.items():
            M = eval_many(algo, fid, DIM, BUDGET, RUNS, RNG_SEED, **DEFAULT_GA_KWARGS)
            results[name] = M
            mean, std = M.mean(axis=0), M.std(axis=0)
            out = np.vstack([mean, std]).T
            np.savetxt(
                f"results/F{fid}_{name.replace(' ','_')}.csv",
                out,
                delimiter=",",
                header="mean,std",
                comments="",
            )
        x = np.arange(BUDGET)
        plt.figure(figsize=(8, 5))
        for name, M in results.items():
            m, s = M.mean(axis=0), M.std(axis=0)
            plt.plot(x, m, label=name, linewidth=1.5)
            plt.fill_between(x, m - s, m + s, alpha=0.15)
        plt.title(f"Fixed-budget: F{fid}, n={DIM}, runs={RUNS}, budget={BUDGET}")
        plt.xlabel("Evaluations")
        plt.ylabel("Best fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/F{fid}_fixed_budget.png")
        plt.close()
    print("Done. Check ./plots and ./results")

if __name__ == "__main__":
    main()
