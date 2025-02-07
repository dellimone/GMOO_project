from typing import Tuple, Callable, Optional, List
import numpy as np
from numpy.typing import NDArray

import benchmarks

from util import OptimizationResult, Config, Population, Fitness


class PSO:
    """
    Particle Swarm Optimization (PSO) implementation for optimization problems.
    """

    def __init__(self, config: Config):
        """
        Initializes the PSO algorithm with a given configuration.

        Args:
            config (Config): Configuration object containing PSO parameters.
        """
        self.config = config
        self.population = Population().init_population(config)
        self.fitness = Fitness().init_fitness(self.population)
        self.convergence_history: List[float] = []

    def optimize(self, fitness_function: Callable[[NDArray], float]) -> OptimizationResult:
        """
        Runs PSO for the configured number of iterations, updating internal state.

        Args:
            fitness_function (Callable[[NDArray], float]): Function to evaluate fitness of solutions.

        Returns:
            OptimizationResult: The best solution found and its corresponding fitness.
        """
        self._update_fitness_stat(self.fitness, fitness_function, self.population)
        for iteration in range(self.config.max_iterations):
            self._pso_update(self.fitness, self.population, iteration)
            self._update_fitness_stat(self.fitness, fitness_function, self.population)
            self.convergence_history.append(self.fitness.global_best_fitness)

        return OptimizationResult(
            best_solution=self.fitness.global_best,
            best_fitness=self.fitness.global_best_fitness,
            convergence_history=self.convergence_history
        )

    def _evaluate_population(self, fitness_function: Callable[[NDArray], float], population: Population) -> NDArray:
        """
        Evaluates the fitness of the entire population.

        Args:
            fitness_function (Callable[[NDArray], float]): The function to compute fitness.
            population (Population): The population object containing individuals.

        Returns:
            NDArray: Array of computed fitness values.
        """
        return np.array([fitness_function(pos) for pos in population.positions])

    def _update_fitness_stat(self, fitness: Fitness, fitness_function: Callable[[NDArray], float],
                             population: Population) -> None:
        """
        Updates the fitness statistics of the population.

        Args:
            fitness (Fitness): The fitness object to update.
            fitness_function (Callable[[NDArray], float]): Function to evaluate fitness.
            population (Population): The population object.
        """
        fitness.fitness = self._evaluate_population(fitness_function, population)

        # Update personal bests
        better_mask = fitness.fitness < fitness.personal_best_fitness
        fitness.personal_best_fitness[better_mask] = fitness.fitness[better_mask]
        fitness.personal_best[better_mask] = population.positions[better_mask].copy()

        # Update global best
        current_best_idx = np.argmin(fitness.fitness)
        if fitness.fitness[current_best_idx] < fitness.global_best_fitness:
            fitness.global_best_fitness = fitness.fitness[current_best_idx]
            fitness.global_best = population.positions[current_best_idx].copy()

    def _pso_update(self, fitness: Fitness, population: Population, iteration: int) -> None:
        """
        Updates the population using velocity and position updates in PSO.

        Args:
            fitness (Fitness): The fitness object.
            population (Population): The population object.
            iteration (int): The current iteration number.
        """
        inertia = (self.config.inertia_weight_start -
                   (iteration / self.config.max_iterations) *
                   (self.config.inertia_weight_start - self.config.inertia_weight_end))

        cognitive_rand = np.random.random((population.size, population.dim))
        social_rand = np.random.random((population.size, population.dim))

        population.velocities = (
                inertia * population.velocities +
                self.config.cognitive_param * cognitive_rand * (fitness.personal_best - population.positions) +
                self.config.social_param * social_rand * (fitness.global_best - population.positions)
        )

        population.velocities = np.clip(
            population.velocities,
            population.velocity_bounds[0],
            population.velocity_bounds[1]
        )

        # Update positions
        new_positions = population.positions + population.velocities

        # Handle boundary rebounds
        lower_bound = self.config.search_space_bounds[0]
        upper_bound = self.config.search_space_bounds[1]

        # Check for lower bound violations
        lower_violations = new_positions < lower_bound
        if np.any(lower_violations):
            new_positions[lower_violations] = 2 * lower_bound - new_positions[lower_violations]
            population.velocities[lower_violations] *= -1

        # Check for upper bound violations
        upper_violations = new_positions > upper_bound
        if np.any(upper_violations):
            new_positions[upper_violations] = 2 * upper_bound - new_positions[upper_violations]
            population.velocities[upper_violations] *= -1

        population.positions = np.clip(new_positions, lower_bound, upper_bound)


def main():
    """
    Main function to run the PSO optimization.
    """
    config = Config(
        problem_dim=5,
        population_size=100,
        max_iterations=200,
        search_space_bounds=(-500, 500),
        inertia_weight_start=0.9,
        inertia_weight_end=0.4,
        cognitive_param=1.49445,
        social_param=1.49445,
        velocity_clip=0.2
    )

    optimizer = PSO(config)
    result = optimizer.optimize(benchmarks.schwefel_function)
    print(f"Best Fitness: {result.best_fitness}")
    print(f"Best Solution: {result.best_solution}")


if __name__ == "__main__":
    main()
