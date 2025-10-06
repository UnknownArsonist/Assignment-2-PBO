import numpy as np

def _construct_solutions(tau: np.ndarray, num_ants: int, rng: np.random.Generator):
    # tau[i] \in [0,1] is the prob of setting bit i = 1
    n = tau.shape[0]
    U = rng.random((num_ants, n))
    return (U < tau).astype(np.int32)

def _oneflip_local_search(x: np.ndarray, func, budget_left: int):
    # Greedy 1-flip hill climb (best-improving single flip)
    # Returns possibly improved x and number of evaluations used
    n = x.size
    used = 0
    fx = func(x); used += 1
    improved = True
    while improved and used < budget_left:
        improved = False
        best_gain = 0.0
        best_i = -1
        for i in range(n):
            if used >= budget_left:
                break
            x[i] ^= 1
            f_try = func(x); used += 1
            gain = f_try - fx
            if gain > best_gain:
                best_gain = gain
                best_i = i
                fx = f_try
            else:
                x[i] ^= 1  # revert
        if best_i >= 0:
            # we already kept the better flip; continue searching
            improved = True
    return x, used, fx


""" def ACO(func, budget=None, num_ants=20, rho=0.1,
        tau_init=0.5, tau_min=0.05, tau_max=0.95, p_explore=0.0):
    
    n=func.meta_data.n_variables
    if budget is None:
        budget = int(n * n * 50)

    try:
        optimum = func.optimum.y
    except:
        optimum = None

    global_best_f = -np.inf
    global_best_x = None    
    

    for r in range(10):
        tau=np.full(n, tau_init, dtype=float)
        run_best_f  = -np.inf
        run_best_x = None
        evals = 0

        while evals < budget:
            colony_X = np.zeros((num_ants, n), dtype=int)
            colony_F = np.full(num_ants, -np.inf, dtype=float)

            for a in range(num_ants):
                # Sample binary solution based on pheromone tau
                x = (np.random.rand(n) < tau).astype(int)

                # Optional extra random flips (exploration)
                if p_explore > 0:
                    flip_mask = np.random.rand(n) < p_explore
                    x[flip_mask] ^= 1

                f = func(x)   # evaluate
                colony_X[a] = x
                colony_F[a] = f
                evals += 1
                if evals >= budget:
                    break

            # Find iteration-best solution
            it_best_idx = int(np.argmax(colony_F))
            it_best_f = colony_F[it_best_idx]
            it_best_x = colony_X[it_best_idx]

            # Update run-best solution
            if it_best_f > run_best_f:
                run_best_f = it_best_f
                run_best_x = it_best_x.copy()

            # Early stop if optimum is found
            if optimum is not None and run_best_f >= optimum:
                break

            # Pheromone update: evaporation + reinforce best-so-far solution
            tau = (1.0 - rho) * tau + rho * run_best_x.astype(float)
            tau = np.clip(tau, tau_min, tau_max)   # enforce bounds

        try:
            func.reset()   # reset between runs
        except Exception:
            pass

        # Update global best
        if run_best_f > global_best_f:
            global_best_f = run_best_f
            global_best_x = run_best_x

    return global_best_f, global_best_x
"""

def ACO(
    func,
    budget: int | None = None,
    num_ants: int = 20,
    rho: float = 0.1,
    tau_init: float = 0.5,
    tau_min: float = 0.05,
    tau_max: float = 0.95,
    ls_steps: int = 0,
    seed: int | None = None,
):
    """
    Binary ACO for IOH PBO functions (maximization).

    Parameters
    ----------
    func : IOH problem; call with a {0,1}^n vector to get fitness.
    budget : total function evaluations allowed (default: 50*n^2).
    num_ants : number of ants (>=10 per spec).
    rho : evaporation rate (0<rho<=1).
    tau_init : initial prob of bit=1.
    tau_min, tau_max : probability bounds to prevent stagnation.
    ls_steps : if >0, apply 1-flip local search to the iteration best
               for at most `ls_steps` passes (each pass scans all bits).
    seed : RNG seed.
    """
    n = func.meta_data.n_variables
    if budget is None:
        budget = int(50 * n * n)

    try:
        target = func.optimum.y
    except Exception:
        target = np.inf  # fallback

    rng = np.random.default_rng(seed)

    # pheromone vector tau[i] = P(bit i = 1)
    tau = np.full(n, float(tau_init), dtype=float)

    evals = 0
    global_best_f = -np.inf
    global_best_x = None

    while evals < budget:
        # 1) Construct solutions
        X = _construct_solutions(tau, num_ants, rng)

        # 2) Evaluate & track best this iteration
        f_vals = []
        for a in range(num_ants):
            if evals >= budget:
                break
            f = func(X[a])
            evals += 1
            f_vals.append(f)
        if not f_vals:  # budget exhausted exactly at loop start
            break

        it_best_idx = int(np.argmax(f_vals))
        it_best_x = X[it_best_idx].copy()
        it_best_f = float(f_vals[it_best_idx])

        # Optional 1-flip local search to polish the iteration best
        if ls_steps > 0 and evals < budget:
            remaining = budget - evals
            for _ in range(ls_steps):
                if remaining <= 0:
                    break
                it_best_x, used, new_f = _oneflip_local_search(it_best_x, func, remaining)
                remaining -= used
                evals += used
                if new_f > it_best_f:
                    it_best_f = new_f
                else:
                    break  # no progress this pass

        # 3) Update global best
        if it_best_f > global_best_f:
            global_best_f = it_best_f
            global_best_x = it_best_x.copy()

        # Early stop if known optimum reached
        if np.isfinite(target) and global_best_f >= target:
            break

        # 4) Pheromone update (best-so-far, bounded)
        # Move probabilities toward the global-best bits
        tau = (1.0 - rho) * tau + rho * global_best_x
        tau = np.clip(tau, tau_min, tau_max)

    # Reset the IOH problem state between runs if the caller loops
    try:
        func.reset()
    except Exception:
        pass
    return global_best_f, global_best_x
    