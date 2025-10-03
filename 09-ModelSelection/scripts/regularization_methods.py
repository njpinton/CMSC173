"""
Regularization methods visualizations for Model Selection.

This script generates illustrations for:
- Ridge (L2) Regularization
- Lasso (L1) Regularization
- Elastic Net
- Regularization path
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    'ridge': '#2E86AB',
    'lasso': '#D32F2F',
    'enet': '#06A77D',
    'train': '#1976D2',
    'test': '#E53935',
}


def generate_data(n_samples=100, n_features=20, noise=0.5, random_state=42):
    """Generate synthetic data for regularization demonstrations."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # True coefficients (sparse)
    true_coef = np.zeros(n_features)
    true_coef[:5] = [3, -2, 1.5, -1, 0.5]
    y = X @ true_coef + np.random.normal(0, noise, n_samples)
    return X, y, true_coef


def plot_regularization_effect():
    """Visualize the effect of regularization on overfitting."""
    # Generate simple polynomial data
    np.random.seed(42)
    X = np.linspace(0, 10, 30).reshape(-1, 1)
    y_true = np.sin(X.ravel()) + 0.5 * X.ravel()
    y = y_true + np.random.normal(0, 1.5, len(X))

    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_test_true = np.sin(X_test.ravel()) + 0.5 * X_test.ravel()

    degree = 15
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    X_test_poly = poly.transform(X_test)

    alphas = [0, 0.01, 1.0, 100.0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()

    for idx, alpha in enumerate(alphas):
        model = Ridge(alpha=alpha)
        model.fit(X_poly, y)
        y_pred = model.predict(X_test_poly)

        # Plot with enhanced styling
        axes[idx].scatter(X, y, c='navy', s=70, alpha=0.7, edgecolors='white',
                         linewidth=1.5, label='Training Data', zorder=3)

        # Color code the prediction based on alpha
        if alpha == 0:
            color = '#D32F2F'  # Red for no regularization
        elif alpha < 1:
            color = '#F57C00'  # Orange for light regularization
        elif alpha < 10:
            color = '#06A77D'  # Green for good regularization
        else:
            color = '#2E86AB'  # Blue for strong regularization

        axes[idx].plot(X_test, y_pred, color=color, linewidth=3.5,
                      label=f'Ridge (α={alpha})', zorder=2)
        axes[idx].plot(X_test, y_test_true, '--', color='darkgreen', linewidth=2.5,
                      label='True Function', alpha=0.8, zorder=1)

        mse = mean_squared_error(y_test_true, y_pred)
        axes[idx].text(0.05, 0.95, f'MSE = {mse:.2f}',
                      transform=axes[idx].transAxes, fontsize=11,
                      verticalalignment='top', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        axes[idx].set_title(f'α = {alpha}', fontsize=14, fontweight='bold',
                           color=color, pad=15)
        axes[idx].set_xlabel('X', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('y', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10, frameon=True, shadow=True)
        axes[idx].set_ylim([-4, 10])
        axes[idx].spines['bottom'].set_linewidth(1.5)
        axes[idx].spines['left'].set_linewidth(1.5)

    plt.suptitle('Effect of Ridge Regularization Parameter', fontsize=16,
                fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('../figures/regularization_effect.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_effect.png")


def plot_regularization_paths():
    """Plot coefficient paths with enhanced styling."""
    X, y, true_coef = generate_data(n_samples=100, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    alphas = np.logspace(-3, 3, 50)

    # Ridge path
    ridge_coefs = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_coefs.append(ridge.coef_)
    ridge_coefs = np.array(ridge_coefs)

    # Lasso path
    lasso_coefs = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        lasso_coefs.append(lasso.coef_)
    lasso_coefs = np.array(lasso_coefs)

    # Plot with enhanced design
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Use a better color palette
    colors = plt.cm.tab10(np.linspace(0, 1, ridge_coefs.shape[1]))

    # Ridge path
    for i in range(ridge_coefs.shape[1]):
        axes[0].plot(alphas, ridge_coefs[:, i], linewidth=2.5,
                    color=colors[i], label=f'β{i+1}', alpha=0.8)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Regularization Parameter (α)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Coefficient Value', fontsize=13, fontweight='bold')
    axes[0].set_title('Ridge (L2) Regularization Path', fontsize=14, fontweight='bold',
                     color=COLOR_PALETTE['ridge'], pad=15)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[0].spines['bottom'].set_linewidth(1.5)
    axes[0].spines['left'].set_linewidth(1.5)
    axes[0].legend(ncol=2, fontsize=9, frameon=True, shadow=True)

    # Lasso path
    for i in range(lasso_coefs.shape[1]):
        axes[1].plot(alphas, lasso_coefs[:, i], linewidth=2.5,
                    color=colors[i], label=f'β{i+1}', alpha=0.8)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Regularization Parameter (α)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Coefficient Value', fontsize=13, fontweight='bold')
    axes[1].set_title('Lasso (L1) Regularization Path', fontsize=14, fontweight='bold',
                     color=COLOR_PALETTE['lasso'], pad=15)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[1].spines['bottom'].set_linewidth(1.5)
    axes[1].spines['left'].set_linewidth(1.5)
    axes[1].legend(ncol=2, fontsize=9, frameon=True, shadow=True)

    # Add annotations
    axes[0].text(0.5, 0.95, 'Coefficients shrink\nbut never reach zero',
                transform=axes[0].transAxes, fontsize=10, ha='center',
                va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[1].text(0.5, 0.95, 'Coefficients can\nbecome exactly zero',
                transform=axes[1].transAxes, fontsize=10, ha='center',
                va='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    plt.savefig('../figures/regularization_paths.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_paths.png")


def plot_l1_vs_l2_geometry():
    """Visualize L1 vs L2 constraint regions with enhanced design."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Generate contours for a simple quadratic function
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)

    # Loss function contours (elliptical)
    center1, center2 = 1.5, 1.0
    Loss = (W1 - center1)**2 + 0.5 * (W2 - center2)**2

    # L2 constraint
    t_l2 = 1.0
    theta = np.linspace(0, 2*np.pi, 100)
    circle_w1 = t_l2 * np.cos(theta)
    circle_w2 = t_l2 * np.sin(theta)

    # Enhanced L2 plot
    contours0 = axes[0].contour(W1, W2, Loss, levels=15, cmap='viridis', alpha=0.6, linewidths=1.5)
    axes[0].clabel(contours0, inline=True, fontsize=8)
    axes[0].plot(circle_w1, circle_w2, color=COLOR_PALETTE['ridge'],
                linewidth=4, label='L2 Constraint: ||w||₂ ≤ t', zorder=3)
    axes[0].fill(circle_w1, circle_w2, color=COLOR_PALETTE['ridge'],
                alpha=0.15, zorder=1)
    axes[0].plot(center1, center2, '*', color='darkred', markersize=20,
                label='Unconstrained Optimum', markeredgecolor='white', markeredgewidth=1.5)
    axes[0].plot([0], [0], 'o', color='white', markersize=12,
                markeredgecolor='black', markeredgewidth=2, label='Origin', zorder=4)

    # Find and mark the constrained optimum
    constrained_angle = np.arctan2(center2, center1)
    constrained_w1 = t_l2 * np.cos(constrained_angle)
    constrained_w2 = t_l2 * np.sin(constrained_angle)
    axes[0].plot(constrained_w1, constrained_w2, 'o', color='lime', markersize=12,
                markeredgecolor='darkgreen', markeredgewidth=2,
                label='Constrained Optimum', zorder=5)

    axes[0].set_xlabel('w₁', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('w₂', fontsize=14, fontweight='bold')
    axes[0].set_title('L2 (Ridge) Regularization', fontsize=15, fontweight='bold',
                     color=COLOR_PALETTE['ridge'], pad=15)
    axes[0].legend(fontsize=10, frameon=True, shadow=True)
    axes[0].axis('equal')
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].spines['bottom'].set_linewidth(1.5)
    axes[0].spines['left'].set_linewidth(1.5)

    # L1 constraint
    t_l1 = 1.5
    diamond_w1 = np.array([t_l1, 0, -t_l1, 0, t_l1])
    diamond_w2 = np.array([0, t_l1, 0, -t_l1, 0])

    # Enhanced L1 plot
    contours1 = axes[1].contour(W1, W2, Loss, levels=15, cmap='viridis', alpha=0.6, linewidths=1.5)
    axes[1].clabel(contours1, inline=True, fontsize=8)
    axes[1].plot(diamond_w1, diamond_w2, color=COLOR_PALETTE['lasso'],
                linewidth=4, label='L1 Constraint: ||w||₁ ≤ t', zorder=3)
    axes[1].fill(diamond_w1, diamond_w2, color=COLOR_PALETTE['lasso'],
                alpha=0.15, zorder=1)
    axes[1].plot(center1, center2, '*', color='darkred', markersize=20,
                label='Unconstrained Optimum', markeredgecolor='white', markeredgewidth=1.5)
    axes[1].plot([0], [0], 'o', color='white', markersize=12,
                markeredgecolor='black', markeredgewidth=2, label='Origin', zorder=4)

    # Mark sparse solution at corner
    axes[1].plot(t_l1, 0, 'o', color='lime', markersize=12,
                markeredgecolor='darkgreen', markeredgewidth=2,
                label='Constrained Optimum (Sparse!)', zorder=5)

    axes[1].set_xlabel('w₁', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('w₂', fontsize=14, fontweight='bold')
    axes[1].set_title('L1 (Lasso) Regularization', fontsize=15, fontweight='bold',
                     color=COLOR_PALETTE['lasso'], pad=15)
    axes[1].legend(fontsize=10, frameon=True, shadow=True)
    axes[1].axis('equal')
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].spines['bottom'].set_linewidth(1.5)
    axes[1].spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('../figures/l1_vs_l2_geometry.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: l1_vs_l2_geometry.png")


def plot_regularization_comparison():
    """Compare Ridge, Lasso, and Elastic Net with enhanced styling."""
    X, y, true_coef = generate_data(n_samples=100, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    alphas = np.logspace(-3, 2, 30)

    ridge_train_errors = []
    ridge_test_errors = []
    lasso_train_errors = []
    lasso_test_errors = []
    enet_train_errors = []
    enet_test_errors = []

    for alpha in alphas:
        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_train_errors.append(mean_squared_error(y_train, ridge.predict(X_train)))
        ridge_test_errors.append(mean_squared_error(y_test, ridge.predict(X_test)))

        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        lasso_train_errors.append(mean_squared_error(y_train, lasso.predict(X_train)))
        lasso_test_errors.append(mean_squared_error(y_test, lasso.predict(X_test)))

        # Elastic Net
        enet = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
        enet.fit(X_train, y_train)
        enet_train_errors.append(mean_squared_error(y_train, enet.predict(X_train)))
        enet_test_errors.append(mean_squared_error(y_test, enet.predict(X_test)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Training errors
    axes[0].plot(alphas, ridge_train_errors, 'o-', color=COLOR_PALETTE['ridge'],
                linewidth=3, label='Ridge', markersize=5, markeredgecolor='white',
                markeredgewidth=1)
    axes[0].plot(alphas, lasso_train_errors, 's-', color=COLOR_PALETTE['lasso'],
                linewidth=3, label='Lasso', markersize=5, markeredgecolor='white',
                markeredgewidth=1)
    axes[0].plot(alphas, enet_train_errors, '^-', color=COLOR_PALETTE['enet'],
                linewidth=3, label='Elastic Net', markersize=5, markeredgecolor='white',
                markeredgewidth=1)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Regularization Parameter (α)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Training MSE', fontsize=13, fontweight='bold')
    axes[0].set_title('Training Error Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(fontsize=12, frameon=True, shadow=True)
    axes[0].spines['bottom'].set_linewidth(1.5)
    axes[0].spines['left'].set_linewidth(1.5)

    # Test errors
    axes[1].plot(alphas, ridge_test_errors, 'o-', color=COLOR_PALETTE['ridge'],
                linewidth=3, label='Ridge', markersize=5, markeredgecolor='white',
                markeredgewidth=1)
    axes[1].plot(alphas, lasso_test_errors, 's-', color=COLOR_PALETTE['lasso'],
                linewidth=3, label='Lasso', markersize=5, markeredgecolor='white',
                markeredgewidth=1)
    axes[1].plot(alphas, enet_test_errors, '^-', color=COLOR_PALETTE['enet'],
                linewidth=3, label='Elastic Net', markersize=5, markeredgecolor='white',
                markeredgewidth=1)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Regularization Parameter (α)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Test MSE', fontsize=13, fontweight='bold')
    axes[1].set_title('Test Error Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(fontsize=12, frameon=True, shadow=True)
    axes[1].spines['bottom'].set_linewidth(1.5)
    axes[1].spines['left'].set_linewidth(1.5)

    # Mark optimal points
    for errors, ax in zip([ridge_test_errors, lasso_test_errors, enet_test_errors],
                          [axes[1], axes[1], axes[1]]):
        optimal_idx = np.argmin(errors)
        ax.plot(alphas[optimal_idx], errors[optimal_idx], '*', markersize=15,
               color='gold', markeredgecolor='darkgoldenrod', markeredgewidth=2, zorder=5)

    plt.tight_layout()
    plt.savefig('../figures/regularization_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_comparison.png")


def plot_sparsity_comparison():
    """Compare sparsity with enhanced bar chart."""
    X, y, true_coef = generate_data(n_samples=100, n_features=15)

    alpha = 0.1
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha, max_iter=10000)
    enet = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)

    ridge.fit(X, y)
    lasso.fit(X, y)
    enet.fit(X, y)

    fig, ax = plt.subplots(figsize=(14, 7))

    x_pos = np.arange(len(true_coef))
    width = 0.2

    # Enhanced bar styling
    bars1 = ax.bar(x_pos - width*1.5, true_coef, width, label='True Coefficients',
                   color='black', alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x_pos - width/2, ridge.coef_, width, label='Ridge',
                   color=COLOR_PALETTE['ridge'], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x_pos + width/2, lasso.coef_, width, label='Lasso',
                   color=COLOR_PALETTE['lasso'], alpha=0.8, edgecolor='white', linewidth=1.5)
    bars4 = ax.bar(x_pos + width*1.5, enet.coef_, width, label='Elastic Net',
                   color=COLOR_PALETTE['enet'], alpha=0.8, edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Feature Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coefficient Value', fontsize=14, fontweight='bold')
    ax.set_title('Coefficient Sparsity Comparison (α = 0.1)', fontsize=16,
                fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # Add text showing non-zero counts
    non_zero_text = (f"Non-zero coefficients:\n"
                    f"Ridge: {np.sum(np.abs(ridge.coef_) > 0.01)}/15\n"
                    f"Lasso: {np.sum(np.abs(lasso.coef_) > 0.01)}/15\n"
                    f"Elastic Net: {np.sum(np.abs(enet.coef_) > 0.01)}/15")
    ax.text(0.98, 0.97, non_zero_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('../figures/sparsity_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: sparsity_comparison.png")


if __name__ == "__main__":
    print("Generating enhanced regularization method visualizations...")
    plot_regularization_effect()
    plot_regularization_paths()
    plot_l1_vs_l2_geometry()
    plot_regularization_comparison()
    plot_sparsity_comparison()
    print("\nAll regularization figures generated successfully!")
