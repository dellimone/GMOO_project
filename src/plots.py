import numpy as np
import matplotlib.pyplot as plt

from ga import GA
from pso import PSO
from sgapso import SGAPSO
from agapso import AGAPSO
from pgapso import PGAPSO
from igapso import IGAPSO


def run_optimizer(optimizer_class, function, config, runs=10):
    """Run a single optimizer multiple times and collect results."""
    best_fitness_values = []
    convergence_history_list = []

    # Create a new config for each run to avoid state sharing
    for _ in range(runs):
        # Create a fresh config instance for each run
        run_config = type(config)(**vars(config))
        optimizer = optimizer_class(run_config)
        result = optimizer.optimize(function)
        best_fitness_values.append(result.best_fitness)
        convergence_history_list.append(result.convergence_history)

    # Ensure all histories have the same length by padding shorter ones
    max_length = max(len(history) for history in convergence_history_list)
    padded_histories = []
    for history in convergence_history_list:
        if len(history) < max_length:
            # Pad with the last value
            padded_history = np.pad(history,
                                    (0, max_length - len(history)),
                                    'edge')
            padded_histories.append(padded_history)
        else:
            padded_histories.append(history)

    return best_fitness_values, np.array(padded_histories)


def run_optimizers(function, config, runs, seed=1):
    """Run all optimizers with the given configuration."""
    if seed is not None:
        np.random.seed(seed)

    optimizers = {
        'GA': GA,
        'PSO': PSO,
        'SGAPSO': SGAPSO,
        'AGAPSO': AGAPSO,
        'PGAPSO': PGAPSO,
        'IGAPSO': IGAPSO
    }

    results = {}
    for name, optimizer_class in optimizers.items():
        # Create optimizer-specific config adjustments if needed
        optimizer_config = adjust_config_for_optimizer(config, name)
        best_fitness_list, history_list = run_optimizer(
            optimizer_class, function, optimizer_config, runs)
        results[name] = {
            'best_fitness_list': best_fitness_list,
            'history_list': history_list
        }

    return results


def adjust_config_for_optimizer(config, optimizer_name):
    """Adjust configuration based on optimizer requirements."""
    # Create a new config instance to avoid modifying the original
    adjusted_config = type(config)(**vars(config))

    if optimizer_name == 'SGAPSO':
        # SGAPSO needs ga_iterations and pso_iterations
        if not hasattr(adjusted_config, 'ga_iterations'):
            adjusted_config.ga_iterations = adjusted_config.max_iterations // 2
        if not hasattr(adjusted_config, 'pso_iterations'):
            adjusted_config.pso_iterations = adjusted_config.max_iterations // 2

    elif optimizer_name == 'AGAPSO':
        # AGAPSO needs offspring_size
        if not hasattr(adjusted_config, 'offspring_size'):
            adjusted_config.offspring_size = adjusted_config.population_size // 2

    elif optimizer_name == 'IGAPSO':
        # IGAPSO needs island-specific parameters
        if not hasattr(adjusted_config, 'island_size'):
            adjusted_config.island_size = adjusted_config.population_size // 5
        if not hasattr(adjusted_config, 'migration_interval'):
            adjusted_config.migration_interval = adjusted_config.max_iterations // 10
        if not hasattr(adjusted_config, 'migration_size'):
            adjusted_config.migration_size = adjusted_config.island_size // 4
        if not hasattr(adjusted_config, 'migration_topology'):
            adjusted_config.migration_topology = 'ring'

    return adjusted_config


def plot_history_lines(history_list, model, ax):
    """Plot convergence histories for a single model."""
    n_runs, n_iter = history_list.shape
    iterations = np.arange(n_iter)

    # Plot individual runs
    for i in range(n_runs):
        ax.plot(iterations, history_list[i], alpha=0.3, color='blue')

    # Plot mean and standard deviation
    mean_history = np.mean(history_list, axis=0)
    std_history = np.std(history_list, axis=0)

    ax.plot(iterations, mean_history, color='red', linewidth=2,
            label='Mean')
    ax.fill_between(iterations,
                    mean_history - std_history,
                    mean_history + std_history,
                    color='red', alpha=0.2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness')
    ax.set_title(f'{model} Convergence')
    ax.grid(True)
    ax.legend()


def plot_all_histories(results):
    """Plot convergence histories for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Convergence Histories of Different Optimizers')

    for i, (model, data) in enumerate(results.items()):
        row = i // 3
        col = i % 3
        plot_history_lines(data['history_list'], model, axes[row, col])

    plt.tight_layout()
    plt.show()


def plot_3d_function(func, x_range, y_range, title, resolution=100):
    """Plot a 2D function in 3D."""
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)

    # Vectorize the function evaluation
    positions = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.array([func(pos) for pos in positions]).reshape(X.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(X, Y, Z, cmap='viridis',
                              edgecolor='none', alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    ax.set_title(title)

    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()