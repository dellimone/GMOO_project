from typing import Tuple, Callable, Optional, List
import numpy as np
from numpy.typing import NDArray

import benchmarks

from util import OptimizationResult, Config, Population, Fitness
from ga import GA
from pso import PSO

class SGAPSO(GA, PSO):
    """
    Sequential Genetic Algorithm - Particle Swarm Optimization implementation.
    Runs GA phase followed by PSO phase for optimization.
    """

    def __init__(self, config: Config):
        """
        Initializes SGAPSO with given configuration.
        """
        self.config = config

    def optimize(self, fitness_function: Callable[[NDArray], float]) -> OptimizationResult:
        """
        Execute the sequential optimization process.
        """
        population = Population().init_population(self.config)
        fitness = Fitness().init_fitness(population)
        self._update_fitness_stat(fitness, fitness_function, population)
        convergence_history = [fitness.global_best_fitness]

        for iteration in range(self.config.ga_iterations):
            self._ga_update(fitness, population, iteration)
            self._update_fitness_stat(fitness, fitness_function, population)
            convergence_history.append(fitness.global_best_fitness)

        for iteration in range(self.config.pso_iterations):
            self._pso_update(fitness, population, iteration)
            self._update_fitness_stat(fitness, fitness_function, population)
            convergence_history.append(fitness.global_best_fitness)

        return OptimizationResult(
            best_solution=fitness.global_best,
            best_fitness=fitness.global_best_fitness,
            convergence_history=convergence_history
        )


def main():
    """Main function to run the SGAPSO optimization."""
    config = Config(problem_dim=5, population_size=100, max_iterations=1000, search_space_bounds=(-500, 500),
                    ga_iterations=500, pso_iterations=500)
    optimizer = SGAPSO(config)
    result = optimizer.optimize(benchmarks.sphere_function)
    print(f"Best Fitness: {result.best_fitness}")
    print(f"Best Solution: {result.best_solution}")


if __name__ == "__main__":
    main()