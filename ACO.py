import numpy as np

def get_algorithm(name):
    if (name == "ACO"):
        return ACO
    raise ValueError("Algorithm not recognized")

def ACO(func, budget=None, num_ants=20, rho=0.1,
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
        best_f = -np.inf
        best_x = None
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
        
    