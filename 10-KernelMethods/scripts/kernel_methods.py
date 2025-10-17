#!/usr/bin/env python3
"""
Generate kernel methods visualization figures for Kernel Methods presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles, make_moons
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_kernel_trick_transformation():
    """Visualize the kernel trick with 2D to 3D transformation."""
    # Generate non-linearly separable data
    X, y = make_circles(n_samples=200, factor=0.3, noise=0.1, random_state=42)

    fig = plt.figure(figsize=(15, 5))

    # Original 2D space
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=60, alpha=0.8)
    ax1.set_xlabel('X₁', fontsize=12)
    ax1.set_ylabel('X₂', fontsize=12)
    ax1.set_title('Original Space (2D)\nNon-linearly Separable', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Transform to 3D using polynomial features
    X_3d = np.column_stack([X[:, 0], X[:, 1], X[:, 0]**2 + X[:, 1]**2])

    # 3D transformed space
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='RdYlBu', s=60, alpha=0.8)
    ax2.set_xlabel('X₁', fontsize=10)
    ax2.set_ylabel('X₂', fontsize=10)
    ax2.set_zlabel('X₁² + X₂²', fontsize=10)
    ax2.set_title('Transformed Space (3D)\nLinearly Separable', fontsize=12, fontweight='bold')

    # Add separating plane in 3D
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 10), np.linspace(-1.5, 1.5, 10))
    zz = np.ones_like(xx) * 0.5  # Separating plane at z=0.5
    ax2.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

    # Back to 2D with RBF kernel
    ax3 = fig.add_subplot(133)
    clf = svm.SVC(kernel='rbf', gamma=1.0, C=1.0)
    clf.fit(X, y)

    # Create mesh for decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax3.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)
    ax3.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['lightblue', 'lightcoral'], alpha=0.3)

    scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=60, alpha=0.8)
    ax3.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=120, linewidth=2, facecolors='none', edgecolors='black')

    ax3.set_xlabel('X₁', fontsize=12)
    ax3.set_ylabel('X₂', fontsize=12)
    ax3.set_title('RBF Kernel Result (2D)\nNon-linear Decision Boundary', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/kernel_trick_transformation.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_different_kernels():
    """Compare different kernel functions on the same dataset."""
    # Generate non-linearly separable data
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_names = ['Linear', 'Polynomial (degree=3)', 'RBF (γ=1)', 'Sigmoid']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        if kernel == 'poly':
            clf = svm.SVC(kernel=kernel, degree=3, C=1.0)
        elif kernel == 'rbf':
            clf = svm.SVC(kernel=kernel, gamma=1.0, C=1.0)
        else:
            clf = svm.SVC(kernel=kernel, C=1.0)

        clf.fit(X, y)

        # Create mesh
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        axes[i].contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        axes[i].contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
                        colors=['lightblue', 'lightcoral'], alpha=0.4)

        # Plot data points
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=60, alpha=0.8)

        # Plot support vectors
        axes[i].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                       s=120, linewidth=2, facecolors='none', edgecolors='black')

        axes[i].set_xlabel('Feature 1', fontsize=11)
        axes[i].set_ylabel('Feature 2', fontsize=11)
        axes[i].set_title(f'{name} Kernel', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/different_kernels.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_rbf_kernel_parameters():
    """Show effect of RBF kernel parameters."""
    X, y = make_circles(n_samples=150, factor=0.3, noise=0.1, random_state=42)

    gammas = [0.1, 1.0, 10.0]
    Cs = [0.1, 1.0, 10.0]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i, gamma in enumerate(gammas):
        for j, C in enumerate(Cs):
            clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
            clf.fit(X, y)

            # Create mesh
            h = 0.01
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))

            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Plot
            axes[i, j].contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
            axes[i, j].contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
                               colors=['lightblue', 'lightcoral'], alpha=0.4)

            scatter = axes[i, j].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50, alpha=0.8)
            axes[i, j].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                              s=100, linewidth=2, facecolors='none', edgecolors='black')

            axes[i, j].set_title(f'γ={gamma}, C={C}', fontsize=12, fontweight='bold')
            axes[i, j].grid(True, alpha=0.3)

            if i == 2:
                axes[i, j].set_xlabel('Feature 1', fontsize=11)
            if j == 0:
                axes[i, j].set_ylabel('Feature 2', fontsize=11)

    plt.tight_layout()
    plt.savefig('../figures/rbf_kernel_parameters.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_kernel_functions():
    """Visualize common kernel functions mathematically."""
    x = np.linspace(-3, 3, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Linear kernel: K(x, x') = x · x'
    x_vals = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, x_vals)
    Z_linear = X * Y
    im1 = axes[0, 0].contourf(X, Y, Z_linear, levels=20, cmap='viridis')
    axes[0, 0].set_title('Linear Kernel: K(x,x\') = x·x\'', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('x\'', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 0])

    # Polynomial kernel: K(x, x') = (γx·x' + r)^d
    Z_poly = (0.5 * X * Y + 1)**3
    im2 = axes[0, 1].contourf(X, Y, Z_poly, levels=20, cmap='viridis')
    axes[0, 1].set_title('Polynomial Kernel: K(x,x\') = (γx·x\'+r)³', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel('x\'', fontsize=11)
    plt.colorbar(im2, ax=axes[0, 1])

    # RBF kernel: K(x, x') = exp(-γ||x-x'||²)
    Z_rbf = np.exp(-1.0 * (X - Y)**2)
    im3 = axes[1, 0].contourf(X, Y, Z_rbf, levels=20, cmap='viridis')
    axes[1, 0].set_title('RBF Kernel: K(x,x\') = exp(-γ||x-x\'||²)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x', fontsize=11)
    axes[1, 0].set_ylabel('x\'', fontsize=11)
    plt.colorbar(im3, ax=axes[1, 0])

    # Sigmoid kernel: K(x, x') = tanh(γx·x' + r)
    Z_sigmoid = np.tanh(0.5 * X * Y + 0.1)
    im4 = axes[1, 1].contourf(X, Y, Z_sigmoid, levels=20, cmap='viridis')
    axes[1, 1].set_title('Sigmoid Kernel: K(x,x\') = tanh(γx·x\'+r)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_ylabel('x\'', fontsize=11)
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('../figures/kernel_functions.png', dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("Generating kernel methods figures...")
    plot_kernel_trick_transformation()
    plot_different_kernels()
    plot_rbf_kernel_parameters()
    plot_kernel_functions()
    print("Kernel methods figures saved to ../figures/")