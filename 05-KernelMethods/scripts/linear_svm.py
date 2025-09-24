#!/usr/bin/env python3
"""
Generate linear SVM visualization figures for Kernel Methods presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def generate_linear_svm_data():
    """Generate linearly separable data for SVM demonstration."""
    # Create linearly separable data
    X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                      random_state=42, cluster_std=1.2)
    return X, y

def plot_linear_svm_margins():
    """Plot linear SVM with decision boundary and margins."""
    X, y = generate_linear_svm_data()

    # Create SVM classifier
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    # Create a mesh for plotting decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Plot
    plt.figure(figsize=(10, 8))

    # Decision function
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=100, alpha=0.8)

    # Highlight support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=200, linewidth=2, facecolors='none', edgecolors='black')

    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.title('Linear SVM: Decision Boundary and Margins', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add legend
    plt.colorbar(scatter, label='Class')

    # Add annotations
    plt.text(0.02, 0.98, 'Support Vectors', transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('../figures/linear_svm_margins.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_svm_optimization():
    """Plot SVM optimization objective visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: Margin maximization concept
    X, y = generate_linear_svm_data()
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    # Plot data and decision boundary
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=100, alpha=0.8)
    ax1.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=200, linewidth=2, facecolors='none', edgecolors='black')

    # Add margin lines
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(X[:, 0].min(), X[:, 0].max())
    yy = a * xx - (clf.intercept_[0]) / w[1]
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin

    ax1.plot(xx, yy, 'k-', linewidth=3, label='Decision Boundary')
    ax1.plot(xx, yy_down, 'k--', linewidth=2, alpha=0.7, label='Margin')
    ax1.plot(xx, yy_up, 'k--', linewidth=2, alpha=0.7)
    ax1.fill_between(xx, yy_down, yy_up, alpha=0.2, color='gray')

    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('Maximum Margin Classifier', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Objective function visualization
    C_values = np.logspace(-2, 2, 100)
    margin_term = 1 / C_values
    loss_term = C_values * 0.1  # Simplified representation

    ax2.loglog(C_values, margin_term, 'b-', linewidth=3, label='Margin Term (1/C)')
    ax2.loglog(C_values, loss_term, 'r-', linewidth=3, label='Loss Term (C×L)')
    ax2.loglog(C_values, margin_term + loss_term, 'g-', linewidth=3,
              label='Total Objective')

    ax2.set_xlabel('Regularization Parameter C', fontsize=12)
    ax2.set_ylabel('Objective Value', fontsize=12)
    ax2.set_title('SVM Objective Function Components', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/svm_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_hard_vs_soft_margin():
    """Compare hard margin vs soft margin SVM."""
    # Generate data with some overlap
    np.random.seed(42)
    X1 = np.random.randn(50, 2) + [2, 2]
    X2 = np.random.randn(50, 2) + [-2, -2]

    # Add some overlapping points
    X1_noise = np.random.randn(10, 2) + [-0.5, -0.5]
    X2_noise = np.random.randn(10, 2) + [0.5, 0.5]

    X = np.vstack([X1, X2, X1_noise, X2_noise])
    y = np.hstack([np.ones(50), -np.ones(50), np.ones(10), -np.ones(10)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Hard margin (high C)
    clf_hard = svm.SVC(kernel='linear', C=1000)
    clf_hard.fit(X, y)

    # Soft margin (low C)
    clf_soft = svm.SVC(kernel='linear', C=0.1)
    clf_soft.fit(X, y)

    for ax, clf, title, C_val in [(ax1, clf_hard, 'Hard Margin (C=1000)', 1000),
                                  (ax2, clf_soft, 'Soft Margin (C=0.1)', 0.1)]:
        # Plot data
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=80, alpha=0.8)

        # Plot support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                  s=150, linewidth=2, facecolors='none', edgecolors='black')

        # Plot decision boundary
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                  linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/hard_vs_soft_margin.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating linear SVM figures...")
    plot_linear_svm_margins()
    plot_svm_optimization()
    plot_hard_vs_soft_margin()
    print("Linear SVM figures saved to ../figures/")