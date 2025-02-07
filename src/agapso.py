from typing import Callable
import numpy as np
from numpy.typing import NDArray

import benchmarks
from ga import GA
from pso import PSO
from util import OptimizationResult, Config, Population, Fitness


class AGAPSO(GA, PSO):
    """
    Hybrid Particle Swarm Optimization - Genetic Algorithm implementation.
    Combines swarm intelligence with evolutionary operations for optimization.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the AGAPSO optimizer with the given configuration.

        Args:
            config (Config): Configuration object containing optimization parameters.
        """
        config.elite_size = config.population_size - config.offspring_size
        self.config = config

    def optimize(self, fitness_function: Callable[[NDArray], float]) -> OptimizationResult:
        """
        Execute the main optimization loop.

        Args:
            fitness_function (Callable[[NDArray], float]): Function to evaluate solution fitness (minimize).

        Returns:
            OptimizationResult: The best solution and convergence history.
        """
        population = Population().init_population(self.config)
        fitness = Fitness().init_fitness(population)
        self._update_fitness_stat(fitness, fitness_function, population)
        convergence_history = [fitness.global_best_fitness]

        for iteration in range(self.config.max_iterations):
            # PSO Phase
            self._pso_update(fitness, population, iteration)
            self._update_fitness_stat(fitness, fitness_function, population)

            # GA Phase
            # self._ga_update(fitness, population, iteration)
            # self._update_fitness_stat(fitness, fitness_function, population)

            convergence_history.append(fitness.global_best_fitness)

        return OptimizationResult(
            best_solution=fitness.global_best,
            best_fitness=fitness.global_best_fitness,
            convergence_history=convergence_history
        )

    def _ga_update(self, fitness: Fitness, population: Population, iteration: int) -> None:
        """
        Apply Genetic Algorithm operations with velocity handling.

        Args:
            fitness (Fitness): Fitness object containing fitness values.
            population (Population): Population object containing positions and velocities.
            iteration (int): Current iteration number.
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
        Perform tournament selection to choose an individual index.

        Args:
            fitness (Fitness): Fitness object containing fitness values.
            population (Population): Population object containing individuals.

        Returns:
            int: Index of the selected individual.
        """
        tournament_indices = np.random.choice(
            population.size,
            size=self.config.tournament_size,
            replace=False
        )
        return tournament_indices[np.argmin(fitness.fitness[tournament_indices])]


def main() -> None:
    """Example usage of the Hybrid PSO-GA optimizer."""
    config = Config(
        problem_dim=5,
        population_size=100,
        max_iterations=200,
        search_space_bounds=(-500, 500),
        # Sequential GAPSO
        ga_iterations=500,
        pso_iterations=500,
        # Alternating GAPSO
        offspring_size=50,
        # Island parameters
        island_size=20,
        migration_interval=50,
        migration_size=5,
        migration_topology="ring",
        # PSO parameters
        inertia_weight_start=0.9,
        inertia_weight_end=0.4,
        cognitive_param=1.49445,
        social_param=1.49445,
        velocity_clip=0.2,
        # GA parameters
        elite_size=3,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.1,
        mutation_scale_start=1.0,
        mutation_scale_end=0.1)

    optimizer = AGAPSO(config)
    result = optimizer.optimize(benchmarks.schwefel_function)
    print(f"Best Fitness: {result.best_fitness}")
    print(f"Best Solution: {result.best_solution}")


if __name__ == "__main__":
    main()