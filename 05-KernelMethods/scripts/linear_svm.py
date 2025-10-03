#!/usr/bin/env python3
"""
Kernel Methods - Linear SVM Figure Generation
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates visualizations for linear Support Vector Machines.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

# PROFESSIONAL STYLING - ALWAYS USE THIS
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

# PROFESSIONAL COLOR PALETTE
COLOR_PALETTE = {
    'primary': '#2E86AB',      # Blue for primary concepts
    'secondary': '#A23B72',    # Purple for secondary
    'accent': '#F18F01',       # Orange for highlights
    'success': '#06A77D',      # Green for optimal/correct
    'danger': '#D32F2F',       # Red for errors/overfitting
    'warning': '#F57C00',      # Orange for warnings
    'info': '#0288D1',         # Light blue for info
    'train': '#1976D2',        # Blue for training data
    'val': '#E53935',          # Red for validation
    'test': '#43A047',         # Green for test
}

def create_output_dir():
    """Create output directory for figures"""
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

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
               linestyles=['--', '-', '--'], linewidths=[2.5, 3.5, 2.5])

    # Plot data points with professional styling
    plt.scatter(X[y==0, 0], X[y==0, 1], s=100, alpha=0.8,
               c=COLOR_PALETTE['primary'], edgecolors='white',
               linewidth=1.5, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], s=100, alpha=0.8,
               c=COLOR_PALETTE['accent'], edgecolors='white',
               linewidth=1.5, label='Class 1')

    # Highlight support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=250, linewidth=3, facecolors='none', edgecolors='black',
               label='Support Vectors')

    plt.xlabel('Feature 1', fontsize=12, fontweight='bold')
    plt.ylabel('Feature 2', fontsize=12, fontweight='bold')
    plt.title('Linear SVM: Decision Boundary and Margins',
             fontsize=16, fontweight='bold', pad=20)

    # Professional legend
    plt.legend(frameon=True, shadow=True, fancybox=True,
              framealpha=0.95, loc='best')

    # Add annotation
    plt.annotate('Maximum\nMargin',
                xy=(0, 0), xytext=(2, -1),
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=COLOR_PALETTE['warning'],
                         edgecolor=COLOR_PALETTE['danger'],
                         linewidth=2, alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=2,
                              color=COLOR_PALETTE['danger']))

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/linear_svm_margins.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_svm_optimization():
    """Plot SVM optimization objective visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: Margin maximization concept
    X, y = generate_linear_svm_data()
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    # Plot data and decision boundary
    ax1.scatter(X[y==0, 0], X[y==0, 1], s=100, alpha=0.8,
               c=COLOR_PALETTE['primary'], edgecolors='white',
               linewidth=1.5, label='Class 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], s=100, alpha=0.8,
               c=COLOR_PALETTE['accent'], edgecolors='white',
               linewidth=1.5, label='Class 1')
    ax1.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=250, linewidth=3, facecolors='none', edgecolors='black',
               label='Support Vectors')

    # Add margin lines
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(X[:, 0].min(), X[:, 0].max())
    yy = a * xx - (clf.intercept_[0]) / w[1]
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin

    ax1.plot(xx, yy, color='black', linewidth=3.5, label='Decision Boundary')
    ax1.plot(xx, yy_down, 'k--', linewidth=2.5, alpha=0.7, label='Margin')
    ax1.plot(xx, yy_up, 'k--', linewidth=2.5, alpha=0.7)
    ax1.fill_between(xx, yy_down, yy_up, alpha=0.15, color=COLOR_PALETTE['info'])

    ax1.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax1.set_title('Maximum Margin Classifier', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)

    # Right plot: Objective function visualization
    C_values = np.logspace(-2, 2, 100)
    margin_term = 1 / C_values
    loss_term = C_values * 0.1  # Simplified representation

    ax2.loglog(C_values, margin_term, color=COLOR_PALETTE['primary'],
              linewidth=3, label='Margin Term (1/C)', marker='o',
              markersize=4, markevery=10, markeredgecolor='white',
              markeredgewidth=1)
    ax2.loglog(C_values, loss_term, color=COLOR_PALETTE['danger'],
              linewidth=3, label='Loss Term (C×L)', marker='s',
              markersize=4, markevery=10, markeredgecolor='white',
              markeredgewidth=1)
    ax2.loglog(C_values, margin_term + loss_term,
              color=COLOR_PALETTE['success'], linewidth=3,
              label='Total Objective', marker='^',
              markersize=4, markevery=10, markeredgecolor='white',
              markeredgewidth=1)

    ax2.set_xlabel('Regularization Parameter C', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
    ax2.set_title('SVM Objective Function Components', fontsize=14,
                 fontweight='bold', pad=15)
    ax2.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/svm_optimization.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
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
        # Plot data with professional colors
        mask_pos = y == 1
        mask_neg = y == -1

        ax.scatter(X[mask_pos, 0], X[mask_pos, 1], s=80, alpha=0.8,
                  c=COLOR_PALETTE['primary'], edgecolors='white',
                  linewidth=1.5, label='Class +1')
        ax.scatter(X[mask_neg, 0], X[mask_neg, 1], s=80, alpha=0.8,
                  c=COLOR_PALETTE['accent'], edgecolors='white',
                  linewidth=1.5, label='Class -1')

        # Plot support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                  s=200, linewidth=3, facecolors='none', edgecolors='black',
                  label='Support Vectors')

        # Plot decision boundary
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                  linestyles=['--', '-', '--'], linewidths=[2.5, 3.5, 2.5])

        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/hard_vs_soft_margin.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def main():
    """Generate all linear SVM figures"""
    print("=" * 60)
    print("Generating Linear SVM Figures")
    print("CMSC 173 - Machine Learning")
    print("=" * 60)

    plot_linear_svm_margins()
    print("✓ Generated linear_svm_margins.png")

    plot_svm_optimization()
    print("✓ Generated svm_optimization.png")

    plot_hard_vs_soft_margin()
    print("✓ Generated hard_vs_soft_margin.png")

    print("=" * 60)
    print("✅ Linear SVM figures generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()