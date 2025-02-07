from dataclasses import dataclass
from typing import Tuple, Callable, List, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    best_solution: NDArray
    best_fitness: float
    convergence_history: List[float]

@dataclass
class Config:
    problem_dim: int
    population_size: int
    max_iterations: int
    search_space_bounds: Tuple[float, float]
    # Sequential GAPSO
    ga_iterations: int = 1
    pso_iterations: int = 1
    # Alternating GAPSO
    offspring_size: int = 10
    # Island parameters
    island_size: int = 20
    migration_interval: int = 20
    migration_size: int = 5
    migration_topology: str = "ring"
    # PSO parameters
    inertia_weight_start: float = 0.9
    inertia_weight_end: float = 0.4
    cognitive_param: float = 1.49445
    social_param: float = 1.49445
    velocity_clip: float = 0.2
    # GA parameters
    elite_size: int = 3
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_scale_start: float = 1.0
    mutation_scale_end: float = 0.1

@dataclass
class PSOConfig:
    problem_dim: int
    population_size: int
    max_iterations: int
    search_space_bounds: tuple[float, float]
    inertia_weight_start: float = 0.9
    inertia_weight_end: float = 0.4
    cognitive_param: float = 1.49445
    social_param: float = 1.49445
    velocity_clip: float = 0.2

@dataclass
class GAConfig:
    problem_dim: int
    population_size: int
    max_iterations: int
    search_space_bounds: tuple[float, float]
    elite_size: int = 3
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_scale_start: float = 1.0
    mutation_scale_end: float = 0.1


class Fitness:
    def __init__(self):
        pass

    def init_fitness(self, population):
        self.fitness = np.full(population.size, float('inf'))
        self.global_best: Optional[NDArray] = None
        self.global_best_fitness = float('inf')
        self.personal_best = population.positions.copy()
        self.personal_best_fitness = np.full(population.size, float('inf'))
        return self

    def split(self, population):
        midpoint = population.size // 2

        sorted_indexes = np.argsort(self.fitness)
        top_half_indexes = sorted_indexes[:midpoint]
        bot_half_indexes = sorted_indexes[midpoint:]

        top_half = Fitness()
        top_half.fitness = self.fitness[top_half_indexes]
        top_half.global_best_fitness = np.min(top_half.fitness)
        top_half.global_best = population.positions[top_half_indexes][np.argmin(top_half.fitness)]
        top_half.personal_best = self.personal_best[top_half_indexes]
        top_half.personal_best_fitness = self.personal_best_fitness[top_half_indexes]

        bot_half = Fitness()
        bot_half.fitness = self.fitness[bot_half_indexes]
        bot_half.global_best_fitness = np.min(bot_half.fitness)
        bot_half.global_best = population.positions[bot_half_indexes][np.argmin(bot_half.fitness)]
        bot_half.personal_best = self.personal_best[bot_half_indexes]
        bot_half.personal_best_fitness = self.personal_best_fitness[bot_half_indexes]

        return bot_half, top_half

    def join(self, fit1, fit2, population):
        self.fitness = np.concatenate((fit1.fitness, fit2.fitness))
        self.personal_best = np.concatenate((fit1.personal_best, fit2.personal_best))
        self.personal_best_fitness = np.concatenate((fit1.personal_best_fitness, fit2.personal_best_fitness))
        best_idx = np.argmin(self.personal_best_fitness)
        self.global_best = self.personal_best[best_idx]
        self.global_best_fitness = self.personal_best_fitness[best_idx]


class Population:
    def __init__(self):
        pass

    def init_population(self, config: Config):
        self.size = config.population_size
        self.dim = config.problem_dim

        self.positions = np.random.uniform(
            config.search_space_bounds[0],
            config.search_space_bounds[1],
            (self.size, self.dim)
        )

        max_velocity = (config.search_space_bounds[1] - config.search_space_bounds[0]) * config.velocity_clip
        self.velocities = np.random.uniform(
            -max_velocity,
            max_velocity,
            (self.size, self.dim)
        )
        self.velocity_bounds = (-max_velocity, max_velocity)
        return self

    def split(self, fitness: Fitness):
        midpoint = self.size // 2

        sorted_indexes = np.argsort(fitness.fitness)
        top_half_indexes = sorted_indexes[:midpoint]
        bot_half_indexes = sorted_indexes[midpoint:]

        top_half = Population()
        top_half.size = len(top_half_indexes)
        top_half.dim = self.dim
        top_half.positions = self.positions[top_half_indexes]
        top_half.velocity_bounds = self.velocity_bounds
        top_half.velocities = self.velocities[top_half_indexes]

        bot_half = Population()
        bot_half.size = len(bot_half_indexes)
        bot_half.dim = self.dim
        bot_half.positions = self.positions[bot_half_indexes]
        bot_half.velocity_bounds = self.velocity_bounds
        bot_half.velocities = self.velocities[bot_half_indexes]

        return bot_half, top_half

    def join(self, pop1, pop2):
        self.positions = np.concatenate((pop1.positions, pop2.positions))
        self.velocities = np.concatenate((pop1.velocities, pop2.velocities))