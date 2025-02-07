from typing import Tuple, Callable, List, Optional
import numpy as np
from numpy.typing import NDArray
import random

from pso import PSO
from ga import GA
from util import OptimizationResult, Config, PSOConfig, GAConfig

import benchmarks


class IGAPSO:
    """
    Island Genetic Algorithm - Particle Swarm Optimization (IGAPSO).
    Combines GA and PSO with migration between islands to optimize a fitness function.
    """

    def __init__(self, config: Config):
        """
        Initializes the IGAPSO optimizer.

        Args:
            config (Config): Configuration object containing optimization settings including
                             GA, PSO, migration parameters, and others.
        """
        self.config = config
        self.num_islands = config.population_size // config.island_size
        self.ga = self._init_ga()
        self.islands = self._init_islands()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []

    def _init_ga(self) -> GA:
        """
        Initializes the Genetic Algorithm (GA) helper for island migration.

        Returns:
            GA: The initialized GA instance for migration.
        """
        ga_config = GAConfig(
            problem_dim=self.config.problem_dim,
            population_size=self.config.island_size,
            max_iterations=self.config.max_iterations,
            search_space_bounds=self.config.search_space_bounds,
            elite_size=3,
            tournament_size=self.config.tournament_size,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            mutation_scale_start=self.config.mutation_scale_start,
            mutation_scale_end=self.config.mutation_scale_end
        )
        return GA(ga_config)

    def _init_islands(self) -> List[PSO]:
        """
        Initializes the PSO instances for each island.

        Returns:
            List[PSO]: List of PSO instances for each island.
        """
        pso_config = PSOConfig(
            problem_dim=self.config.problem_dim,
            population_size=self.config.island_size,
            max_iterations=self.config.migration_interval,
            search_space_bounds=self.config.search_space_bounds,
            inertia_weight_start=self.config.inertia_weight_start,
            inertia_weight_end=self.config.inertia_weight_end,
            cognitive_param=self.config.cognitive_param,
            social_param=self.config.social_param,
            velocity_clip=self.config.velocity_clip
        )
        return [PSO(pso_config) for _ in range(self.num_islands)]

    def _get_destination_island(self, source_id: int) -> int:
        """
        Determines the destination island for migration based on the specified topology.

        Args:
            source_id (int): The source island's index.

        Returns:
            int: The destination island's index.
        """
        if self.config.migration_topology == 'ring':
            return (source_id + 1) % self.num_islands
        elif self.config.migration_topology == 'fully_connected':
            possible = list(range(self.num_islands))
            possible.remove(source_id)
            return random.choice(possible)

    def _perform_migration(self, iteration: int) -> None:
        """
        Executes migration between islands using GA operators.

        Args:
            iteration (int): The current iteration of the optimization loop, used for adjusting migration parameters.

        Returns:
            None: This function modifies the islands in place.
        """
        migrations = []
        for source_id, source in enumerate(self.islands):
            dest_id = self._get_destination_island(source_id)
            dest = self.islands[dest_id]
            migrants = self._generate_migrants(source, dest, iteration)
            migrations.append((dest_id, migrants))

        # Apply migrations to destination islands
        for dest_id, migrants in migrations:
            dest = self.islands[dest_id]
            fitness = dest.fitness.fitness
            worst_indices = np.argsort(fitness)[-self.config.migration_size:]
            dest.population.positions[worst_indices] = migrants[:self.config.migration_size]

    def _generate_migrants(self, source: PSO, dest: PSO, iteration: int) -> NDArray:
        """
        Generates migrants using tournament selection, crossover, and mutation.

        Args:
            source (PSO): The source island where parents are selected.
            dest (PSO): The destination island where migrants are sent.
            iteration (int): The current iteration of the optimization process.

        Returns:
            NDArray: The generated migrants (children) to be transferred to the destination island.
        """
        migrants = []
        for _ in range(self.config.migration_size):
            # Select parents from source and destination
            parent1 = self.ga._tournament_selection(source.fitness, source.population)
            parent2 = self.ga._tournament_selection(dest.fitness, dest.population)
            # Crossover and mutate
            child1, child2 = self.ga._blxalpha_crossover(parent1, parent2)
            child1 = self.ga._mutation(child1, iteration)
            child2 = self.ga._mutation(child2, iteration)
            migrants.extend([child1, child2])
        return np.array(migrants)

    def optimize(self, fitness_function: Callable[[NDArray], float]) -> OptimizationResult:
        """
        Runs the optimization loop, applying PSO on each island with periodic migrations.

        Args:
            fitness_function (Callable[[NDArray], float]): A function to evaluate the fitness of a solution.

        Returns:
            OptimizationResult: The result of the optimization, containing best solution, best fitness, and convergence history.
        """
        total_steps = self.config.max_iterations // self.config.migration_interval

        for step in range(total_steps):
            # Run PSO on each island for migration_interval iterations
            for island in self.islands:
                island.optimize(fitness_function)
                # Update global best
                if island.fitness.global_best_fitness < self.best_fitness:
                    self.best_fitness = island.fitness.global_best_fitness
                    self.best_solution = island.fitness.global_best.copy()
            self.convergence_history.append(self.best_fitness)
            # Perform migration after each interval
            self._perform_migration(step * self.config.migration_interval)
        return OptimizationResult(self.best_solution,
                                  self.best_fitness,
                                  self.convergence_history)


def main() -> None:
    """Example usage of the Island-GA-PSO (IGAPSO) optimizer."""
    config = Config(
        problem_dim=5,
        population_size=100,
        max_iterations=500,
        search_space_bounds=(-500, 500),
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
        tournament_size=3,
        crossover_rate=0.9,
        mutation_rate=0.1,
        mutation_scale_start=0.5,
        mutation_scale_end=0.1,
    )

    # Initialize IGAPSO and run optimization
    optimizer = IGAPSO(config)
    result = optimizer.optimize(benchmarks.schwefel_function)  # Assume Benchmarks is imported

    # Print results
    print(f"Best Fitness: {result.best_fitness}")
    print(f"Best Solution: {result.best_solution}")


if __name__ == "__main__":
    main()
