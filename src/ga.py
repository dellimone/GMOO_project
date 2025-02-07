from typing import Tuple, Callable, Optional, List
import numpy as np
from numpy.typing import NDArray

import benchmarks
from util import OptimizationResult, Config, Population, Fitness


class GA:
    """
    Genetic Algorithm (GA) implementation for optimization problems.
    """

    def __init__(self, config: Config):
        """
        Initializes the GA with a given configuration.

        Args:
            config (Config): Configuration object containing GA parameters.
        """
        self.config = config

    def optimize(self, fitness_function: Callable[[NDArray], float]) -> OptimizationResult:
        """
        Runs the GA optimization process.

        Args:
            fitness_function (Callable[[NDArray], float]): Function to evaluate fitness of solutions.

        Returns:
            OptimizationResult: The best solution found and its corresponding fitness.
        """
        population = Population().init_population(self.config)
        fitness = Fitness().init_fitness(population)

        # Initialize fitness statistics
        self._update_fitness_stat(fitness, fitness_function, population)
        convergence_history = [fitness.global_best_fitness]

        # Main GA loop
        for iteration in range(self.config.max_iterations):
            self._ga_update(fitness, population, iteration)
            self._update_fitness_stat(fitness, fitness_function, population)
            convergence_history.append(fitness.global_best_fitness)

        return OptimizationResult(
            best_solution=fitness.global_best,
            best_fitness=fitness.global_best_fitness,
            convergence_history=convergence_history
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

    def _tournament_selection(self, fitness: Fitness, population: Population) -> NDArray:
        """
        Performs tournament selection to choose a parent for reproduction.

        Args:
            fitness (Fitness): The fitness object.
            population (Population): The population object.

        Returns:
            NDArray: The selected parent individual.
        """
        indices = np.random.choice(population.size, size=self.config.tournament_size, replace=False)
        winner_idx = indices[np.argmin(fitness.fitness[indices])]
        return population.positions[winner_idx].copy()

    def _blxalpha_crossover(self, parent1: NDArray, parent2: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Performs BLX-alpha crossover to generate offspring.

        Args:
            parent1 (NDArray): First parent.
            parent2 (NDArray): Second parent.

        Returns:
            Tuple[NDArray, NDArray]: Two offspring solutions.
        """
        if np.random.random() < self.config.crossover_rate:
            alpha = 0.3
            gamma = np.random.uniform(-alpha, 1 + alpha, self.config.problem_dim)
            child1 = parent1 + gamma * (parent2 - parent1)
            child2 = parent2 + gamma * (parent1 - parent2)

            # Ensure offspring are within search bounds
            child1 = np.clip(child1, *self.config.search_space_bounds)
            child2 = np.clip(child2, *self.config.search_space_bounds)
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutation(self, individual: NDArray, iteration: int) -> NDArray:
        """
        Applies mutation to an individual based on iteration progress.

        Args:
            individual (NDArray): The individual solution.
            iteration (int): The current iteration number.

        Returns:
            NDArray: Mutated individual.
        """
        progress_ratio = iteration / self.config.max_iterations
        mutation_scale = (self.config.mutation_scale_start -
                          progress_ratio * (self.config.mutation_scale_start - self.config.mutation_scale_end))

        mutation_mask = np.random.random(self.config.problem_dim) < self.config.mutation_rate
        if mutation_mask.any():
            individual = individual.copy()
            space_range = self.config.search_space_bounds[1] - self.config.search_space_bounds[0]
            mutation = np.random.normal(0, mutation_scale * space_range, self.config.problem_dim)
            individual[mutation_mask] += mutation[mutation_mask]
            individual = np.clip(individual, *self.config.search_space_bounds)
        return individual

    def _ga_update(self, fitness: Fitness, population: Population, iteration: int) -> None:
        """
        Updates the population for the next generation using selection, crossover, and mutation.

        Args:
            fitness (Fitness): The fitness object.
            population (Population): The population object.
            iteration (int): The current iteration number.
        """
        # Preserve elite individuals
        elite_indices = np.argsort(fitness.fitness)[:self.config.elite_size]
        elite_individuals = population.positions[elite_indices].copy()

        # Generate new offspring
        for i in range(self.config.elite_size, population.size, 2):
            parent1 = self._tournament_selection(fitness, population)
            parent2 = self._tournament_selection(fitness, population)
            child1, child2 = self._blxalpha_crossover(parent1, parent2)

            population.positions[i] = self._mutation(child1, iteration)
            if i + 1 < population.size:
                population.positions[i + 1] = self._mutation(child2, iteration)

        # Retain elite individuals
        population.positions[:self.config.elite_size] = elite_individuals


def main():
    """
    Main function to run the GA optimization.
    """
    config = Config(
        problem_dim=5,
        population_size=20,
        max_iterations=100,
        search_space_bounds=(-500, 500),
    )

    optimizer = GA(config)
    result = optimizer.optimize(benchmarks.schwefel_function)
    print(f"Best Fitness: {result.best_fitness}")
    print(f"Best Solution: {result.best_solution}")


if __name__ == "__main__":
    main()
