#!/usr/bin/env python3
"""
Generate kernel regression visualization figures for Kernel Methods presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def generate_regression_data():
    """Generate synthetic regression data with noise."""
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 0.5 * X.ravel() + np.sin(2 * X.ravel()) + np.random.normal(0, 0.3, X.shape[0])
    return X, y

def plot_svr_demonstration():
    """Demonstrate Support Vector Regression with different kernels."""
    X, y = generate_regression_data()

    # Split data for training
    X_train = X[::2]  # Every other point
    y_train = y[::2]
    X_test = X
    y_test = y

    kernels = ['linear', 'poly', 'rbf']
    kernel_names = ['Linear', 'Polynomial (degree=3)', 'RBF (γ=0.1)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        if kernel == 'poly':
            svr = SVR(kernel=kernel, degree=3, C=100, epsilon=0.1)
        elif kernel == 'rbf':
            svr = SVR(kernel=kernel, gamma=0.1, C=100, epsilon=0.1)
        else:
            svr = SVR(kernel=kernel, C=100, epsilon=0.1)

        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)

        # Plot training data
        axes[i].scatter(X_train, y_train, color='red', s=60, alpha=0.8,
                       label='Training Data', zorder=3)

        # Plot true function
        axes[i].plot(X_test, y_test, color='blue', linewidth=2, alpha=0.7,
                    label='True Function', zorder=1)

        # Plot prediction
        axes[i].plot(X_test, y_pred, color='green', linewidth=3,
                    label=f'SVR Prediction', zorder=2)

        # Add epsilon tube
        epsilon = 0.1
        axes[i].fill_between(X_test.ravel(), y_pred - epsilon, y_pred + epsilon,
                            alpha=0.2, color='green', label=f'ε-tube (ε={epsilon})')

        # Highlight support vectors
        support_vectors_mask = svr.support_
        if len(support_vectors_mask) > 0:
            axes[i].scatter(X_train[support_vectors_mask], y_train[support_vectors_mask],
                           s=120, facecolors='none', edgecolors='black',
                           linewidth=2, label='Support Vectors', zorder=4)

        axes[i].set_xlabel('X', fontsize=12)
        axes[i].set_ylabel('y', fontsize=12)
        axes[i].set_title(f'SVR with {name} Kernel', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/svr_demonstration.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_epsilon_parameter_effect():
    """Show the effect of epsilon parameter in SVR."""
    X, y = generate_regression_data()
    X_train = X[::3]  # Sparse training data
    y_train = y[::3]

    epsilons = [0.01, 0.1, 0.5]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, epsilon in enumerate(epsilons):
        svr = SVR(kernel='rbf', gamma=0.1, C=100, epsilon=epsilon)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X)

        # Plot data and predictions
        axes[i].scatter(X_train, y_train, color='red', s=60, alpha=0.8,
                       label='Training Data', zorder=3)
        axes[i].plot(X, y, color='blue', linewidth=2, alpha=0.7,
                    label='True Function', zorder=1)
        axes[i].plot(X, y_pred, color='green', linewidth=3,
                    label='SVR Prediction', zorder=2)

        # Add epsilon tube
        axes[i].fill_between(X.ravel(), y_pred - epsilon, y_pred + epsilon,
                            alpha=0.3, color='green', label=f'ε-tube (ε={epsilon})')

        # Highlight support vectors
        support_vectors_mask = svr.support_
        if len(support_vectors_mask) > 0:
            axes[i].scatter(X_train[support_vectors_mask], y_train[support_vectors_mask],
                           s=120, facecolors='none', edgecolors='black',
                           linewidth=2, label='Support Vectors', zorder=4)

        axes[i].set_xlabel('X', fontsize=12)
        axes[i].set_ylabel('y', fontsize=12)
        axes[i].set_title(f'SVR with ε = {epsilon}', fontsize=14, fontweight='bold')
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)

        # Add text showing number of support vectors
        n_sv = len(support_vectors_mask) if len(support_vectors_mask) > 0 else len(svr.support_)
        axes[i].text(0.05, 0.95, f'Support Vectors: {n_sv}',
                    transform=axes[i].transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')

    plt.tight_layout()
    plt.savefig('../figures/epsilon_parameter_effect.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_kernel_ridge_vs_svr():
    """Compare Kernel Ridge Regression with SVR."""
    X, y = generate_regression_data()
    X_train = X[::2]
    y_train = y[::2]

    # Kernel Ridge Regression
    kr = KernelRidge(kernel='rbf', gamma=0.1, alpha=1.0)
    kr.fit(X_train, y_train)
    y_pred_kr = kr.predict(X)

    # Support Vector Regression
    svr = SVR(kernel='rbf', gamma=0.1, C=100, epsilon=0.1)
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Kernel Ridge Regression
    axes[0].scatter(X_train, y_train, color='red', s=60, alpha=0.8,
                   label='Training Data', zorder=3)
    axes[0].plot(X, y, color='blue', linewidth=2, alpha=0.7,
                label='True Function', zorder=1)
    axes[0].plot(X, y_pred_kr, color='purple', linewidth=3,
                label='Kernel Ridge Prediction', zorder=2)

    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Kernel Ridge Regression', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Support Vector Regression
    axes[1].scatter(X_train, y_train, color='red', s=60, alpha=0.8,
                   label='Training Data', zorder=3)
    axes[1].plot(X, y, color='blue', linewidth=2, alpha=0.7,
                label='True Function', zorder=1)
    axes[1].plot(X, y_pred_svr, color='green', linewidth=3,
                label='SVR Prediction', zorder=2)

    # Add epsilon tube for SVR
    epsilon = 0.1
    axes[1].fill_between(X.ravel(), y_pred_svr - epsilon, y_pred_svr + epsilon,
                        alpha=0.2, color='green', label=f'ε-tube (ε={epsilon})')

    # Highlight support vectors
    support_vectors_mask = svr.support_
    if len(support_vectors_mask) > 0:
        axes[1].scatter(X_train[support_vectors_mask], y_train[support_vectors_mask],
                       s=120, facecolors='none', edgecolors='black',
                       linewidth=2, label='Support Vectors', zorder=4)

    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('Support Vector Regression', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/kernel_ridge_vs_svr.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_regularization_comparison():
    """Compare different regularization strengths."""
    X, y = generate_regression_data()
    X_train = X[::2]
    y_train = y[::2]

    # Different regularization parameters
    alphas = [0.01, 1.0, 100.0]  # For Kernel Ridge
    Cs = [1000, 10, 0.1]  # For SVR (inverse of regularization)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Kernel Ridge Regression with different alphas
    for i, alpha in enumerate(alphas):
        kr = KernelRidge(kernel='rbf', gamma=0.1, alpha=alpha)
        kr.fit(X_train, y_train)
        y_pred = kr.predict(X)

        axes[0, i].scatter(X_train, y_train, color='red', s=60, alpha=0.8,
                          label='Training Data', zorder=3)
        axes[0, i].plot(X, y, color='blue', linewidth=2, alpha=0.7,
                       label='True Function', zorder=1)
        axes[0, i].plot(X, y_pred, color='purple', linewidth=3,
                       label='KR Prediction', zorder=2)

        axes[0, i].set_xlabel('X', fontsize=12)
        axes[0, i].set_ylabel('y', fontsize=12)
        axes[0, i].set_title(f'Kernel Ridge: α = {alpha}', fontsize=14, fontweight='bold')
        axes[0, i].legend(fontsize=10)
        axes[0, i].grid(True, alpha=0.3)

    # SVR with different C values
    for i, C in enumerate(Cs):
        svr = SVR(kernel='rbf', gamma=0.1, C=C, epsilon=0.1)
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X)

        axes[1, i].scatter(X_train, y_train, color='red', s=60, alpha=0.8,
                          label='Training Data', zorder=3)
        axes[1, i].plot(X, y, color='blue', linewidth=2, alpha=0.7,
                       label='True Function', zorder=1)
        axes[1, i].plot(X, y_pred, color='green', linewidth=3,
                       label='SVR Prediction', zorder=2)

        # Add epsilon tube
        epsilon = 0.1
        axes[1, i].fill_between(X.ravel(), y_pred - epsilon, y_pred + epsilon,
                               alpha=0.2, color='green', label=f'ε-tube')

        axes[1, i].set_xlabel('X', fontsize=12)
        axes[1, i].set_ylabel('y', fontsize=12)
        axes[1, i].set_title(f'SVR: C = {C}', fontsize=14, fontweight='bold')
        axes[1, i].legend(fontsize=10)
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/regularization_comparison.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("Generating regression kernel figures...")
    plot_svr_demonstration()
    plot_epsilon_parameter_effect()
    plot_kernel_ridge_vs_svr()
    plot_regularization_comparison()
    print("Regression kernel figures saved to ../figures/")