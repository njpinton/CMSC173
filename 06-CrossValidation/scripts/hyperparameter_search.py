#!/usr/bin/env python3
"""
Hyperparameter Search Visualization Scripts
CMSC 173 - Machine Learning

This script generates visualizations for hyperparameter search methods including:
- Grid Search visualization and performance
- Random Search comparison
- Bayesian Optimization (Optuna) traces
- Search space coverage comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
# import optuna  # Commented out - will simulate instead
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

def create_grid_search_visualization():
    """Create visualization of grid search parameter space."""
    # Define parameter grids
    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.001, 0.01, 0.1, 1]

    # Create meshgrid for visualization
    C_mesh, gamma_mesh = np.meshgrid(C_values, gamma_values)

    # Simulate performance scores (for visualization)
    np.random.seed(42)
    scores = np.array([
        [0.85, 0.88, 0.90, 0.87],
        [0.89, 0.92, 0.94, 0.91],
        [0.91, 0.93, 0.95, 0.92],
        [0.88, 0.90, 0.91, 0.89]
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Grid points
    ax1.scatter(C_mesh, gamma_mesh, s=100, c='red', alpha=0.7, edgecolors='darkred')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('C (Regularization)', fontweight='bold')
    ax1.set_ylabel('γ (Gamma)', fontweight='bold')
    ax1.set_title('Grid Search: Parameter Space Coverage', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add grid lines
    for c in C_values:
        ax1.axvline(x=c, color='gray', alpha=0.3, linestyle='--')
    for g in gamma_values:
        ax1.axhline(y=g, color='gray', alpha=0.3, linestyle='--')

    # Plot 2: Performance heatmap
    im = ax2.imshow(scores, cmap='RdYlBu_r', aspect='auto', origin='lower')
    ax2.set_xticks(range(len(C_values)))
    ax2.set_yticks(range(len(gamma_values)))
    ax2.set_xticklabels([f'{c}' for c in C_values])
    ax2.set_yticklabels([f'{g}' for g in gamma_values])
    ax2.set_xlabel('C (Regularization)', fontweight='bold')
    ax2.set_ylabel('γ (Gamma)', fontweight='bold')
    ax2.set_title('Grid Search: Performance Heatmap', fontweight='bold')

    # Add score annotations
    for i in range(len(gamma_values)):
        for j in range(len(C_values)):
            ax2.text(j, i, f'{scores[i, j]:.3f}', ha='center', va='center',
                    fontweight='bold', color='white')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Cross-Validation Accuracy', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/grid_search_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_random_search_visualization():
    """Create visualization comparing grid search vs random search."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Grid Search points
    C_grid = [0.1, 1, 10, 100]
    gamma_grid = [0.001, 0.01, 0.1, 1]
    C_mesh, gamma_mesh = np.meshgrid(C_grid, gamma_grid)

    ax1.scatter(C_mesh.ravel(), gamma_mesh.ravel(), s=100, c='red', alpha=0.7,
               edgecolors='darkred', label='Grid Points')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('C', fontweight='bold')
    ax1.set_ylabel('γ', fontweight='bold')
    ax1.set_title('Grid Search\n(16 points)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Random Search points
    np.random.seed(42)
    n_random = 16
    C_random = 10 ** np.random.uniform(-1, 2, n_random)  # 0.1 to 100
    gamma_random = 10 ** np.random.uniform(-3, 0, n_random)  # 0.001 to 1

    ax2.scatter(C_random, gamma_random, s=100, c='blue', alpha=0.7,
               edgecolors='darkblue', label='Random Points')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('C', fontweight='bold')
    ax2.set_ylabel('γ', fontweight='bold')
    ax2.set_title('Random Search\n(16 points)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Comparison of coverage
    # Create a finer grid to show coverage
    C_fine = np.logspace(-1, 2, 100)
    gamma_fine = np.logspace(-3, 0, 100)

    # Calculate coverage for each method
    grid_coverage = []
    for c in C_fine:
        for g in gamma_fine:
            min_dist_grid = min([abs(np.log10(c) - np.log10(cg)) + abs(np.log10(g) - np.log10(gg))
                               for cg in C_grid for gg in gamma_grid])
            grid_coverage.append(min_dist_grid)

    random_coverage = []
    for c in C_fine:
        for g in gamma_fine:
            min_dist_random = min([abs(np.log10(c) - np.log10(cr)) + abs(np.log10(g) - np.log10(gr))
                                 for cr, gr in zip(C_random, gamma_random)])
            random_coverage.append(min_dist_random)

    # Create histogram comparison
    ax3.hist(grid_coverage, bins=20, alpha=0.7, label='Grid Search', color='red', density=True)
    ax3.hist(random_coverage, bins=20, alpha=0.7, label='Random Search', color='blue', density=True)
    ax3.set_xlabel('Distance to Nearest Searched Point', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title('Parameter Space Coverage', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/random_vs_grid_search.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_optuna_optimization_trace():
    """Create simulated Bayesian optimization trace visualization."""
    # Create synthetic optimization data
    np.random.seed(42)

    def objective_function(C, gamma):
        """Synthetic objective function for demonstration."""
        # Synthetic performance function with noise
        optimal_C, optimal_gamma = 10, 0.1
        score = 0.95 - 0.1 * ((np.log10(C) - np.log10(optimal_C))**2 +
                             (np.log10(gamma) - np.log10(optimal_gamma))**2)
        score += np.random.normal(0, 0.02)  # Add noise
        return score

    # Simulate Bayesian optimization behavior
    n_trials = 50
    trial_numbers = list(range(n_trials))
    objective_values = []
    C_values = []
    gamma_values = []

    # Initial random exploration
    for i in range(5):
        C = 10 ** np.random.uniform(-1, 2)  # 0.1 to 100
        gamma = 10 ** np.random.uniform(-3, 0)  # 0.001 to 1
        score = objective_function(C, gamma)

        C_values.append(C)
        gamma_values.append(gamma)
        objective_values.append(score)

    # Bayesian optimization-like behavior (exploit good regions)
    for i in range(5, n_trials):
        # With probability, explore around best found parameters
        if np.random.random() < 0.7 and len(objective_values) > 0:
            # Find best parameters so far
            best_idx = np.argmax(objective_values)
            best_C, best_gamma = C_values[best_idx], gamma_values[best_idx]

            # Sample around best parameters with decreasing variance
            variance_factor = max(0.1, 1.0 - i/n_trials)  # Decrease over time
            C = best_C * np.exp(np.random.normal(0, variance_factor))
            gamma = best_gamma * np.exp(np.random.normal(0, variance_factor))

            # Clip to bounds
            C = np.clip(C, 0.1, 100)
            gamma = np.clip(gamma, 0.001, 1)
        else:
            # Random exploration
            C = 10 ** np.random.uniform(-1, 2)
            gamma = 10 ** np.random.uniform(-3, 0)

        score = objective_function(C, gamma)
        C_values.append(C)
        gamma_values.append(gamma)
        objective_values.append(score)

    best_values = [max(objective_values[:i+1]) for i in range(len(objective_values))]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Optimization history
    ax1.plot(trial_numbers, objective_values, 'o-', alpha=0.6, label='Trial Value', markersize=4)
    ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
    ax1.set_xlabel('Trial Number', fontweight='bold')
    ax1.set_ylabel('Objective Value', fontweight='bold')
    ax1.set_title('Optuna Optimization History', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter exploration over time
    scatter = ax2.scatter(C_values, gamma_values, c=trial_numbers, cmap='viridis',
                         s=50, alpha=0.7, edgecolors='black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('C', fontweight='bold')
    ax2.set_ylabel('γ', fontweight='bold')
    ax2.set_title('Parameter Exploration Over Time', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Trial Number', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: C parameter over time
    ax3.semilogx(trial_numbers, C_values, 'o-', markersize=4, alpha=0.7)
    ax3.set_xlabel('Trial Number', fontweight='bold')
    ax3.set_ylabel('C Value', fontweight='bold')
    ax3.set_title('C Parameter Evolution', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Gamma parameter over time
    ax4.semilogy(trial_numbers, gamma_values, 'o-', markersize=4, alpha=0.7, color='orange')
    ax4.set_xlabel('Trial Number', fontweight='bold')
    ax4.set_ylabel('γ Value', fontweight='bold')
    ax4.set_title('γ Parameter Evolution', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/optuna_optimization_trace.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_search_efficiency_comparison():
    """Create comparison of search method efficiency."""
    np.random.seed(42)
    n_trials = 50

    # Simulate optimization performance for each method
    trials = np.arange(1, n_trials + 1)

    # Grid search: systematic but may waste evaluations
    grid_performance = []
    current_best = 0.7
    for i in trials:
        if i % 4 == 0:  # Only improve every 4 trials (systematic nature)
            current_best += np.random.normal(0.01, 0.005)
        grid_performance.append(min(current_best, 0.95))

    # Random search: more exploration, steady improvement
    random_performance = []
    current_best = 0.7
    for i in trials:
        improvement = np.random.exponential(0.008) if np.random.random() > 0.7 else 0
        current_best = max(current_best, current_best + improvement)
        random_performance.append(min(current_best, 0.95))

    # Bayesian optimization: smart, focused search
    bayesian_performance = []
    current_best = 0.7
    for i in trials:
        # Bayesian optimization improves faster initially, then plateaus
        improvement_prob = max(0.9 - i/60, 0.1)  # Decreasing exploration
        improvement = np.random.exponential(0.015) if np.random.random() < improvement_prob else 0
        current_best = max(current_best, current_best + improvement)
        bayesian_performance.append(min(current_best, 0.96))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Performance over trials
    ax1.plot(trials, grid_performance, 'r-', linewidth=2, label='Grid Search', marker='s', markersize=3)
    ax1.plot(trials, random_performance, 'b-', linewidth=2, label='Random Search', marker='o', markersize=3)
    ax1.plot(trials, bayesian_performance, 'g-', linewidth=2, label='Bayesian Optimization', marker='^', markersize=3)

    ax1.set_xlabel('Number of Trials', fontweight='bold')
    ax1.set_ylabel('Best Validation Score', fontweight='bold')
    ax1.set_title('Search Method Efficiency Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.65, 1.0)

    # Plot 2: Final performance distribution
    methods = ['Grid\nSearch', 'Random\nSearch', 'Bayesian\nOptimization']
    final_scores = [grid_performance[-1], random_performance[-1], bayesian_performance[-1]]
    colors = ['red', 'blue', 'green']

    bars = ax2.bar(methods, final_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Best Validation Score', fontweight='bold')
    ax2.set_title('Final Performance Comparison', fontweight='bold')
    ax2.set_ylim(0.85, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, final_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('../figures/search_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hyperparameter_surface():
    """Create 3D surface plot of hyperparameter performance."""
    # Create parameter meshgrid
    C_vals = np.logspace(-1, 2, 20)
    gamma_vals = np.logspace(-3, 0, 20)
    C_mesh, gamma_mesh = np.meshgrid(C_vals, gamma_vals)

    # Synthetic performance function
    def performance_function(C, gamma):
        """Synthetic performance function with multiple peaks."""
        log_C = np.log10(C)
        log_gamma = np.log10(gamma)

        # Main peak around C=10, gamma=0.1
        peak1 = 0.95 * np.exp(-((log_C - 1)**2 + (log_gamma + 1)**2))

        # Secondary peak around C=1, gamma=0.01
        peak2 = 0.85 * np.exp(-((log_C - 0)**2 + (log_gamma + 2)**2))

        # Base performance
        base = 0.6

        return base + peak1 + peak2

    Z = performance_function(C_mesh, gamma_mesh)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(np.log10(C_mesh), np.log10(gamma_mesh), Z,
                          cmap='RdYlBu_r', alpha=0.9, edgecolor='none')

    # Add contour lines
    contours = ax.contour(np.log10(C_mesh), np.log10(gamma_mesh), Z,
                         levels=10, colors='black', alpha=0.4, linewidths=0.5)

    ax.set_xlabel('log₁₀(C)', fontweight='bold')
    ax.set_ylabel('log₁₀(γ)', fontweight='bold')
    ax.set_zlabel('Validation Score', fontweight='bold')
    ax.set_title('Hyperparameter Performance Landscape', fontweight='bold', pad=20)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
    cbar.set_label('Validation Score', fontweight='bold')

    # Set better viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig('../figures/hyperparameter_surface.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating hyperparameter search visualizations...")

    create_grid_search_visualization()
    print("✓ Grid search visualization created")

    create_random_search_visualization()
    print("✓ Random vs grid search comparison created")

    create_optuna_optimization_trace()
    print("✓ Optuna optimization trace created")

    create_search_efficiency_comparison()
    print("✓ Search efficiency comparison created")

    create_hyperparameter_surface()
    print("✓ Hyperparameter surface plot created")

    print("All hyperparameter search visualizations completed!")