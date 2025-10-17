"""
Core methods for Model Selection and Evaluation visualizations.

This script generates fundamental illustrations for:
- Bias-Variance Tradeoff
- Training vs Validation Error curves
- Model Complexity visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Enhanced styling
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Professional color palette
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#D32F2F',
    'warning': '#F57C00',
    'info': '#0288D1',
    'train': '#1976D2',
    'val': '#E53935',
    'test': '#43A047',
}

def generate_synthetic_data(n_samples=100, noise=0.5, random_state=42):
    """Generate synthetic data for demonstrations."""
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples)
    y_true = np.sin(X) + 0.5 * X
    y = y_true + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y, y_true


def plot_bias_variance_tradeoff():
    """Visualize the bias-variance tradeoff with enhanced styling."""
    fig, ax = plt.subplots(figsize=(12, 7))

    complexity = np.linspace(1, 10, 150)

    # Theoretical curves with smoother interpolation
    bias_squared = 10 / complexity**1.5
    variance = 0.1 * complexity**1.8
    irreducible_error = 0.3
    total_error = bias_squared + variance + irreducible_error

    # Plot with gradient fills
    ax.plot(complexity, bias_squared, color=COLOR_PALETTE['primary'],
            linewidth=3, label='Bias²', zorder=3)
    ax.fill_between(complexity, 0, bias_squared, alpha=0.15,
                     color=COLOR_PALETTE['primary'], zorder=1)

    ax.plot(complexity, variance, color=COLOR_PALETTE['danger'],
            linewidth=3, label='Variance', zorder=3)
    ax.fill_between(complexity, 0, variance, alpha=0.15,
                     color=COLOR_PALETTE['danger'], zorder=1)

    ax.plot(complexity, total_error, color=COLOR_PALETTE['success'],
            linewidth=4, label='Total Error', zorder=4)

    ax.axhline(irreducible_error, color='gray', linestyle='--',
               linewidth=2, label='Irreducible Error', alpha=0.7, zorder=2)

    # Mark optimal point with annotation
    optimal_idx = np.argmin(total_error)
    optimal_x = complexity[optimal_idx]
    optimal_y = total_error[optimal_idx]
    ax.plot(optimal_x, optimal_y, 'o', color=COLOR_PALETTE['success'],
            markersize=15, markeredgecolor='white', markeredgewidth=2,
            label='Optimal Complexity', zorder=5)

    ax.annotate('Optimal\nComplexity', xy=(optimal_x, optimal_y),
                xytext=(optimal_x + 1.5, optimal_y + 0.5),
                fontsize=11, fontweight='bold', color=COLOR_PALETTE['success'],
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_PALETTE['success']))

    # Annotations for regions
    ax.text(2, 8, 'High Bias\nUnderfitting', fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                     edgecolor=COLOR_PALETTE['primary'], linewidth=2, alpha=0.7))
    ax.text(8, 8, 'High Variance\nOverfitting', fontsize=12, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral',
                     edgecolor=COLOR_PALETTE['danger'], linewidth=2, alpha=0.7))

    ax.set_xlabel('Model Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error', fontsize=14, fontweight='bold')
    ax.set_title('Bias-Variance Tradeoff', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper center', frameon=True, shadow=True)
    ax.set_ylim([0, 10])
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('../figures/bias_variance_tradeoff.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: bias_variance_tradeoff.png")


def plot_underfitting_overfitting():
    """Illustrate underfitting and overfitting with enhanced visuals."""
    X, y, y_true = generate_synthetic_data(n_samples=30, noise=1.5)
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_test_true = np.sin(X_test.ravel()) + 0.5 * X_test.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    degrees = [1, 4, 15]
    titles = ['Underfitting (High Bias)', 'Good Fit (Balanced)', 'Overfitting (High Variance)']
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['success'], COLOR_PALETTE['danger']]

    for ax, degree, title, color in zip(axes, degrees, titles, colors):
        # Fit polynomial
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_test_poly = poly.transform(X_test)

        model = Ridge(alpha=0.001)
        model.fit(X_poly, y)
        y_pred = model.predict(X_test_poly)

        # Plot with enhanced styling
        ax.scatter(X, y, c='navy', s=80, alpha=0.7, edgecolors='white',
                  linewidth=1.5, label='Training Data', zorder=3)
        ax.plot(X_test, y_pred, color=color, linewidth=3.5,
               label=f'Model (degree {degree})', zorder=2)
        ax.plot(X_test, y_test_true, '--', color='darkgreen', linewidth=2.5,
               label='True Function', alpha=0.8, zorder=1)

        # Compute and display MSE
        train_mse = mean_squared_error(y, model.predict(X_poly))
        ax.text(0.05, 0.95, f'Train MSE: {train_mse:.2f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('y', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', color=color, pad=15)
        ax.legend(fontsize=9, frameon=True, shadow=True, loc='lower right')
        ax.set_ylim([-4, 10])
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('../figures/underfitting_overfitting.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: underfitting_overfitting.png")


def plot_learning_curves():
    """Generate learning curves with enhanced visualization."""
    X, y, _ = generate_synthetic_data(n_samples=200, noise=1.0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    train_sizes = np.linspace(10, len(X_train), 20, dtype=int)
    degrees = [1, 4, 15]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ['Underfitting (High Bias)', 'Well-fitted Model', 'Overfitting (High Variance)']
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['success'], COLOR_PALETTE['danger']]

    for ax, degree, title, color in zip(axes, degrees, titles, colors):
        train_errors = []
        val_errors = []

        for size in train_sizes:
            X_subset = X_train[:size]
            y_subset = y_train[:size]

            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X_subset)
            X_val_poly = poly.transform(X_val)

            model = Ridge(alpha=0.001)
            model.fit(X_poly, y_subset)

            train_pred = model.predict(X_poly)
            val_pred = model.predict(X_val_poly)

            train_errors.append(mean_squared_error(y_subset, train_pred))
            val_errors.append(mean_squared_error(y_val, val_pred))

        # Plot with filled regions for confidence
        ax.plot(train_sizes, train_errors, 'o-', color=COLOR_PALETTE['train'],
               linewidth=3, label='Training Error', markersize=7, markeredgecolor='white',
               markeredgewidth=1.5)
        ax.plot(train_sizes, val_errors, 's-', color=COLOR_PALETTE['val'],
               linewidth=3, label='Validation Error', markersize=7, markeredgecolor='white',
               markeredgewidth=1.5)

        # Shade the gap between curves
        ax.fill_between(train_sizes, train_errors, val_errors, alpha=0.2, color='gray')

        ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', color=color, pad=15)
        ax.legend(fontsize=10, frameon=True, shadow=True)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('../figures/learning_curves.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: learning_curves.png")


def plot_model_complexity_curve():
    """Visualize model complexity with enhanced design."""
    X, y, _ = generate_synthetic_data(n_samples=100, noise=1.0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = range(1, 16)
    train_errors = []
    val_errors = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        model = Ridge(alpha=0.001)
        model.fit(X_train_poly, y_train)

        train_pred = model.predict(X_train_poly)
        val_pred = model.predict(X_val_poly)

        train_errors.append(mean_squared_error(y_train, train_pred))
        val_errors.append(mean_squared_error(y_val, val_pred))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot with markers and gradients
    ax.plot(degrees, train_errors, 'o-', color=COLOR_PALETTE['train'],
           linewidth=3.5, label='Training Error', markersize=9,
           markeredgecolor='white', markeredgewidth=2)
    ax.fill_between(degrees, 0, train_errors, alpha=0.15, color=COLOR_PALETTE['train'])

    ax.plot(degrees, val_errors, 's-', color=COLOR_PALETTE['val'],
           linewidth=3.5, label='Validation Error', markersize=9,
           markeredgecolor='white', markeredgewidth=2)
    ax.fill_between(degrees, 0, val_errors, alpha=0.15, color=COLOR_PALETTE['val'])

    # Mark optimal complexity
    optimal_degree = degrees[np.argmin(val_errors)]
    optimal_error = min(val_errors)
    ax.axvline(optimal_degree, color=COLOR_PALETTE['success'], linestyle='--',
              linewidth=3, alpha=0.7)
    ax.plot(optimal_degree, optimal_error, 'o', color=COLOR_PALETTE['success'],
           markersize=16, markeredgecolor='white', markeredgewidth=2.5, zorder=5)

    ax.annotate(f'Optimal\nDegree = {optimal_degree}',
               xy=(optimal_degree, optimal_error),
               xytext=(optimal_degree + 2, optimal_error + 0.5),
               fontsize=12, fontweight='bold', color=COLOR_PALETTE['success'],
               arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_PALETTE['success']))

    # Annotate regions
    ax.text(2, max(val_errors) * 0.85, 'Underfitting\n(High Bias)', fontsize=12,
           ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue',
                    edgecolor=COLOR_PALETTE['primary'], linewidth=2, alpha=0.8))
    ax.text(12, max(val_errors) * 0.85, 'Overfitting\n(High Variance)', fontsize=12,
           ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral',
                    edgecolor=COLOR_PALETTE['danger'], linewidth=2, alpha=0.8))

    ax.set_xlabel('Polynomial Degree (Model Complexity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='bold')
    ax.set_title('Model Complexity vs Error', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    ax.set_xticks(degrees)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('../figures/model_complexity_curve.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: model_complexity_curve.png")


def plot_train_test_split():
    """Visualize train-test split with modern design."""
    fig, ax = plt.subplots(figsize=(14, 4))

    # Create dataset representation
    n_samples = 100
    train_ratio = 0.7
    n_train = int(n_samples * train_ratio)

    # Draw rectangles with gradients
    train_rect = plt.Rectangle((0, 0), n_train, 1,
                               facecolor=COLOR_PALETTE['train'],
                               edgecolor='navy', linewidth=3, alpha=0.8)
    test_rect = plt.Rectangle((n_train, 0), n_samples - n_train, 1,
                              facecolor=COLOR_PALETTE['test'],
                              edgecolor='darkgreen', linewidth=3, alpha=0.8)

    ax.add_patch(train_rect)
    ax.add_patch(test_rect)

    # Add sample indicators
    for i in range(0, n_train, 5):
        ax.plot([i, i], [0, 1], 'w-', linewidth=0.5, alpha=0.3)
    for i in range(n_train, n_samples, 5):
        ax.plot([i, i], [0, 1], 'w-', linewidth=0.5, alpha=0.3)

    # Add labels with better styling
    ax.text(n_train/2, 0.5, f'Training Set\n{train_ratio*100:.0f}% of data\n({n_train} samples)',
           ha='center', va='center', fontsize=14, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='navy', alpha=0.7))
    ax.text(n_train + (n_samples-n_train)/2, 0.5,
           f'Test Set\n{(1-train_ratio)*100:.0f}% of data\n({n_samples-n_train} samples)',
           ha='center', va='center', fontsize=14, fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen', alpha=0.7))

    # Add arrow indicating split point
    ax.annotate('', xy=(n_train, 1.3), xytext=(n_train, 1.15),
               arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(n_train, 1.35, 'Split Point', ha='center', fontsize=11, fontweight='bold')

    ax.set_xlim(-2, n_samples+2)
    ax.set_ylim(-0.2, 1.6)
    ax.set_xlabel('Dataset Samples', fontsize=14, fontweight='bold')
    ax.set_title('Train-Test Split Strategy', fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('../figures/train_test_split.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: train_test_split.png")


if __name__ == "__main__":
    print("Generating enhanced core method visualizations...")
    plot_bias_variance_tradeoff()
    plot_underfitting_overfitting()
    plot_learning_curves()
    plot_model_complexity_curve()
    plot_train_test_split()
    print("\nAll core method figures generated successfully!")
