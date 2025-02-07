import numpy as np
import matplotlib.pyplot as plt

import GA
import PSO
import SGAPSO
import AGAPSO
import PGAPSO
import IGAPSO

from src.IGAPSO import IGAPSO


def run_optimizer(optimizer_class, function, runs=10, **kwargs):
    best_fitness_values = []
    convergence_history_list = []

    for _ in range(runs):
        optimizer = optimizer_class(**kwargs)
        result = optimizer.optimize(function)
        best_fitness_values.append(result.best_fitness)
        convergence_history_list.append(result.convergence_history)

    return best_fitness_values, np.array(convergence_history_list)

def plot_history_lines(history_list, model):
    fig, ax = plt.subplots(1, 1, figsize=(7, 3), layout="constrained")
    n_runs, n_iter = history_list.shape
    iterations = np.arange(0,n_iter,1)
    for i in range(n_runs):
        ax.plot(iterations, history_list[i])

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')
    ax.set_title(model)
    plt.show()

def run_optimizers(function,
                   problem_dim,
                   population_size,
                   max_iterations,
                   search_space_bounds,
                   runs):

    models = {'GA':GA.GA,
              'PSO':PSO.PSO,
              'SGAPSO':SGAPSO.SGAPSO,
              'AGAPSO':AGAPSO.APSOGA,
              'PGAPSO':PGAPSO.PGAPSO,
              'IGAPSO':IGAPSO.IGAPSO}

    results = {}

    for name, model in models.items():
        best_fitness_list, history_list = run_optimizer(model,
                                  function,
                                  runs,
                                  problem_dim=problem_dim,
                                  population_size=population_size,
                                  max_iterations=max_iterations,
                                  search_space_bounds=search_space_bounds)

        results[name] = {'best_fitness_list':best_fitness_list,
                         'history_list':history_list}
    return results

