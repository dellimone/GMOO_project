from typing import Tuple, Callable, Optional, List
import numpy as np
from numpy.typing import NDArray

import benchmarks

from ga import GA
from pso import PSO

from util import OptimizationResult, Config, Population, Fitness


class PGAPSO(GA, PSO):
    """
    Hybrid Genetic Algorithm - Particle Swarm Optimization implementation.
    Combines evolutionary operations with swarm intelligence for optimization.
    """

    def __init__(self, config: Config):
        """
        Initializes the PGAPSO optimizer with a given configuration.

        Args:
            config (Config): Configuration object containing optimization settings.
        """
        self.config = config

    def optimize(self, fitness_function: Callable[[NDArray], float]) -> OptimizationResult:
        """
        Execute the main optimization loop combining Genetic Algorithm and Particle Swarm Optimization.

        Args:
            fitness_function (Callable[[NDArray], float]): A function to evaluate solution fitness.
                                                          It should return a float value indicating the fitness (typically to minimize).

        Returns:
            OptimizationResult: Contains the best solution, best fitness, and convergence history.
        """
        # Initialize population and fitness.
        population = Population().init_population(self.config)
        fitness = Fitness().init_fitness(population)
        self._update_fitness_stat(fitness, fitness_function, population)
        convergence_history = [fitness.global_best_fitness]

        for iteration in range(self.config.max_iterations):
            # Split population and fitness into two halves.
            bot_half_population, top_half_population = population.split(fitness)
            bot_half_fitness, top_half_fitness = fitness.split(population)

            # Update the two halves using GA and PSO.
            self._ga_update(top_half_fitness, top_half_population, iteration)
            self._pso_update(bot_half_fitness, bot_half_population, iteration)

            # Join updated halves back into the population and fitness.
            population.join(bot_half_population, top_half_population)
            fitness.join(bot_half_fitness, top_half_fitness, population)

            # Re-evaluate fitness.
            self._update_fitness_stat(fitness, fitness_function, population)

            convergence_history.append(fitness.global_best_fitness)

        return OptimizationResult(
            best_solution=fitness.global_best,
            best_fitness=fitness.global_best_fitness,
            convergence_history=convergence_history
        )

    def _ga_update(self, fitness: Fitness, population: Population, iteration: int) -> None:
        """
        Perform Genetic Algorithm updates to the population, including elitism, tournament selection,
        crossover, and mutation for positions and velocities.

        Args:
            fitness (Fitness): Fitness object containing fitness values for the population.
            population (Population): Population object holding current positions and velocities of individuals.
            iteration (int): The current iteration of the optimization process, used to adjust mutation rates.

        Returns:
            None: This function modifies the population in-place.
        """
        # Preserve elites
        elite_indices = np.argsort(fitness.fitness)[:self.config.elite_size]
        elite_positions = population.positions[elite_indices].copy()
        elite_velocities = population.velocities[elite_indices].copy()

        # Main GA loop for the rest of the population
        for i in range(self.config.elite_size, population.size, 2):
            # Tournament selection for both positions and velocities
            parent1_idx = self._tournament_selection_index(fitness, population)
            parent2_idx = self._tournament_selection_index(fitness, population)

            parent1_pos = population.positions[parent1_idx]
            parent2_pos = population.positions[parent2_idx]
            parent1_vel = population.velocities[parent1_idx]
            parent2_vel = population.velocities[parent2_idx]

            # Apply crossover to both positions and velocities
            child1_pos, child2_pos = self._blxalpha_crossover(parent1_pos, parent2_pos)
            child1_vel, child2_vel = self._blxalpha_crossover(parent1_vel, parent2_vel)

            # Clip velocities to bounds
            child1_vel = np.clip(child1_vel, *population.velocity_bounds)
            child2_vel = np.clip(child2_vel, *population.velocity_bounds)

            # Apply mutation to positions and adjust velocities accordingly
            child1_pos = self._mutation(child1_pos, iteration)
            if i + 1 < population.size:
                child2_pos = self._mutation(child2_pos, iteration)

                # Update population with both children
                population.positions[i] = child1_pos
                population.positions[i + 1] = child2_pos
                population.velocities[i] = child1_vel
                population.velocities[i + 1] = child2_vel
            else:
                # If we only have space for one child, update just the first one
                population.positions[i] = child1_pos
                population.velocities[i] = child1_vel

        # Restore elites
        population.positions[:self.config.elite_size] = elite_positions
        population.velocities[:self.config.elite_size] = elite_velocities

    def _tournament_selection_index(self, fitness: Fitness, population: Population) -> int:
        """
        Selects an index from the population using tournament selection.

        Args:
            fitness (Fitness): Fitness object to evaluate the candidates.
            population (Population): Population object from which individuals are selected.

        Returns:
            int: The index of the selected individual.
        """
        tournament_indices = np.random.choice(
            population.size,
            size=self.config.tournament_size,
            replace=False
        )
        return tournament_indices[np.argmin(fitness.fitness[tournament_indices])]


def main() -> None:
    """Example usage of the PGAPSO optimizer to optimize the Schwefel function"""
    config = Config(
        problem_dim=5,
        population_size=100,
        max_iterations=200,
        search_space_bounds=(-500, 500),
        elite_size=3,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.1,
        mutation_scale_start=1.0,
        mutation_scale_end=0.1,
        inertia_weight_start=0.9,
        inertia_weight_end=0.4,
        cognitive_param=1.49445,
        social_param=1.49445,
        velocity_clip=0.2
    )

    optimizer = PGAPSO(config)
    result = optimizer.optimize(benchmarks.schwefel_function)
    print(f"Best Fitness: {result.best_fitness}")
    print(f"Best Solution: {result.best_solution}")


if __name__ == "__main__":
    main()
