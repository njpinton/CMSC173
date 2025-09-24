#!/usr/bin/env python3
"""
Generate replacement figures for TikZ diagrams in Kernel Methods presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.datasets import make_classification, make_blobs
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_supervised_learning_overview():
    """Generate supervised learning overview figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 4)

    # Main box
    main_box = patches.Rectangle((0, 0), 12, 3, linewidth=2,
                                edgecolor='black', facecolor='lightblue', alpha=0.3)
    ax.add_patch(main_box)

    # Title and text
    ax.text(6, 2.5, 'Supervised Learning', fontsize=16, fontweight='bold',
            ha='center', va='center')
    ax.text(6, 2.0, 'Learn a mapping or function: $y = f(x)$', fontsize=12,
            ha='center', va='center')
    ax.text(6, 1.5, 'from inputs (x) to outputs (y),', fontsize=12,
            ha='center', va='center')
    ax.text(6, 1.0, 'given a labelled set of input-output examples', fontsize=12,
            ha='center', va='center', style='italic')

    # Regression box
    reg_box = patches.Rectangle((0.5, 0.3), 5, 0.4, linewidth=2,
                               edgecolor='green', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(reg_box)
    ax.text(3, 0.5, 'Regression', fontsize=14, fontweight='bold',
            ha='center', va='center')

    # Classification box
    class_box = patches.Rectangle((6.5, 0.3), 5, 0.4, linewidth=2,
                                 edgecolor='red', facecolor='lightcoral', alpha=0.5)
    ax.add_patch(class_box)
    ax.text(9, 0.5, 'Classification', fontsize=14, fontweight='bold',
            ha='center', va='center')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/supervised_learning_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_linear_vs_nonlinear_data():
    """Generate linear vs non-linear data comparison with properly separated data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear data - properly separated clusters
    np.random.seed(42)
    # Class 1 (Blue): bottom-right cluster
    X1_class1 = np.random.randn(15, 2) * 0.4 + [4.5, 1.5]
    # Class 2 (Red): top-left cluster
    X1_class2 = np.random.randn(15, 2) * 0.4 + [1.5, 4.5]

    ax1.scatter(X1_class1[:, 0], X1_class1[:, 1], c='blue', s=60, alpha=0.8,
               label='Class 1 (Blue)', edgecolors='darkblue', linewidth=0.5)
    ax1.scatter(X1_class2[:, 0], X1_class2[:, 1], c='red', s=60, alpha=0.8,
               label='Class 2 (Red)', edgecolors='darkred', linewidth=0.5)

    # Add linear decision boundary that clearly separates the classes
    x_line = np.linspace(0, 6, 100)
    y_line = x_line - 0.5  # Original decision boundary (keeping original line)
    ax1.plot(x_line, y_line, 'k-', linewidth=3, label='Linear Decision Boundary')

    # Add margin lines to show separation
    ax1.plot(x_line, y_line + 0.5, 'k--', linewidth=1, alpha=0.5)
    ax1.plot(x_line, y_line - 0.5, 'k--', linewidth=1, alpha=0.5)

    ax1.set_title('Linearly Separable Data', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 6)

    # Non-linear data - concentric circles pattern
    np.random.seed(42)
    # Inner circle (Class 2 - Red)
    n_inner = 20
    theta_inner = np.random.uniform(0, 2*np.pi, n_inner)
    r_inner = 0.8 + 0.3 * np.random.randn(n_inner)
    r_inner = np.abs(r_inner)  # Ensure positive radius

    X2_inner_x = r_inner * np.cos(theta_inner) + 3
    X2_inner_y = r_inner * np.sin(theta_inner) + 3

    # Outer ring (Class 1 - Blue)
    n_outer = 25
    theta_outer = np.random.uniform(0, 2*np.pi, n_outer)
    r_outer = 2.0 + 0.4 * np.random.randn(n_outer)
    r_outer = np.abs(r_outer)  # Ensure positive radius

    X2_outer_x = r_outer * np.cos(theta_outer) + 3
    X2_outer_y = r_outer * np.sin(theta_outer) + 3

    ax2.scatter(X2_outer_x, X2_outer_y, c='blue', s=60, alpha=0.8,
               label='Class 1 (Blue)', edgecolors='darkblue', linewidth=0.5)
    ax2.scatter(X2_inner_x, X2_inner_y, c='red', s=60, alpha=0.8,
               label='Class 2 (Red)', edgecolors='darkred', linewidth=0.5)

    # Add circular decision boundaries
    circle1 = plt.Circle((3, 3), 1.4, fill=False, color='black', linewidth=3,
                        label='Non-linear Decision Boundary')
    ax2.add_patch(circle1)

    # Add margin circles
    circle2 = plt.Circle((3, 3), 1.2, fill=False, color='black', linewidth=1,
                        linestyle='--', alpha=0.5)
    circle3 = plt.Circle((3, 3), 1.6, fill=False, color='black', linewidth=1,
                        linestyle='--', alpha=0.5)
    ax2.add_patch(circle2)
    ax2.add_patch(circle3)

    ax2.set_title('Non-linearly Separable Data', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 6)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('../figures/linear_vs_nonlinear_data.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_perceptron_vs_svm():
    """Generate perceptron vs SVM comparison with properly separated data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Create clearly separated data with same placement as linear_vs_nonlinear
    np.random.seed(42)
    # Class 1 (Blue): bottom-right cluster with some points touching boundary
    X_class1 = np.random.randn(10, 2) * 0.35 + [3.8, 1.2]
    # Class 2 (Red): top-left cluster with some points touching boundary
    X_class2 = np.random.randn(10, 2) * 0.35 + [1.2, 3.8]

    # Perceptron - shows any separating line
    ax1.scatter(X_class1[:, 0], X_class1[:, 1], c='blue', s=70, alpha=0.8,
               label='Class -1 (Blue)', edgecolors='darkblue', linewidth=0.5)
    ax1.scatter(X_class2[:, 0], X_class2[:, 1], c='red', s=70, alpha=0.8,
               label='Class +1 (Red)', edgecolors='darkred', linewidth=0.5)

    # Show multiple possible separating lines for perceptron
    x_line = np.linspace(0, 5, 100)

    # Three different valid separating lines (original boundaries)
    y_line1 = x_line - 0.8
    y_line2 = x_line - 0.3
    y_line3 = x_line + 0.2

    ax1.plot(x_line, y_line1, 'g-', linewidth=2, alpha=0.7, label='Possible Line 1')
    ax1.plot(x_line, y_line2, 'orange', linewidth=2, alpha=0.7, label='Possible Line 2')
    ax1.plot(x_line, y_line3, 'm-', linewidth=2, alpha=0.7, label='Possible Line 3')

    ax1.text(2.5, 0.5, 'Multiple Valid\nSolutions', fontsize=10, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax1.set_title('Perceptron Algorithm', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)

    # SVM with maximum margin
    ax2.scatter(X_class1[:, 0], X_class1[:, 1], c='blue', s=70, alpha=0.8,
               label='Class -1 (Blue)', edgecolors='darkblue', linewidth=0.5)
    ax2.scatter(X_class2[:, 0], X_class2[:, 1], c='red', s=70, alpha=0.8,
               label='Class +1 (Red)', edgecolors='darkred', linewidth=0.5)

    # Maximum margin decision boundary (optimal line between clusters)
    y_boundary = x_line - 0.3  # Original optimal separation line
    y_margin1 = x_line - 0.8   # Lower margin
    y_margin2 = x_line + 0.2   # Upper margin

    ax2.plot(x_line, y_boundary, 'k-', linewidth=3, label='Optimal Decision Boundary')
    ax2.plot(x_line, y_margin1, 'k--', linewidth=1, alpha=0.7, label='Margin Boundary')
    ax2.plot(x_line, y_margin2, 'k--', linewidth=1, alpha=0.7)

    # Highlight support vectors (closest points to decision boundary)
    # Find points closest to the decision boundary
    distances_class1 = []
    distances_class2 = []

    for i, point in enumerate(X_class1):
        # Distance from point to line ax + by + c = 0 is |ax + by + c|/sqrt(a^2 + b^2)
        # Line: y = x - 0.3 => x - y - 0.3 = 0
        dist = abs(point[0] - point[1] - 0.3) / np.sqrt(2)
        distances_class1.append((dist, i))

    for i, point in enumerate(X_class2):
        dist = abs(point[0] - point[1] - 0.3) / np.sqrt(2)
        distances_class2.append((dist, i))

    # Get closest points
    distances_class1.sort()
    distances_class2.sort()

    sv_idx1 = distances_class1[0][1]  # Closest from class 1
    sv_idx2 = distances_class2[0][1]  # Closest from class 2

    # Highlight support vectors
    ax2.scatter([X_class1[sv_idx1, 0]], [X_class1[sv_idx1, 1]],
               s=150, facecolors='none', edgecolors='black', linewidth=3,
               label='Support Vectors')
    ax2.scatter([X_class2[sv_idx2, 0]], [X_class2[sv_idx2, 1]],
               s=150, facecolors='none', edgecolors='black', linewidth=3)

    # Add margin width annotation
    ax2.annotate('', xy=(1.8, 1.5), xytext=(2.3, 2.0),
                arrowprops=dict(arrowstyle='<->', lw=2, color='purple'))
    ax2.text(1.5, 1.8, 'Max\nMargin', fontsize=10, color='purple', ha='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))

    ax2.set_title('Support Vector Machine', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('../figures/perceptron_vs_svm.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_xor_problem():
    """Generate XOR problem visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR labels

    # Plot points
    colors = ['blue', 'red']
    labels = ['Class 0', 'Class 1']

    for i in range(2):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=200, alpha=0.8,
                  label=labels[i], edgecolors='black', linewidth=2)

    # Add point labels
    for i, (x, y_val) in enumerate(X):
        ax.annotate(f'{y[i]}', (x, y_val), xytext=(5, 5), textcoords='offset points',
                   fontsize=14, fontweight='bold')

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('Feature 1', fontsize=14)
    ax.set_ylabel('Feature 2', fontsize=14)
    ax.set_title('XOR Problem\n(Not Linearly Separable)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add some attempted linear boundaries to show impossibility
    x_line = np.linspace(-0.3, 1.3, 100)
    ax.plot(x_line, 0.5 * np.ones_like(x_line), 'k--', alpha=0.5, linewidth=1, label='Failed Linear Boundaries')
    ax.plot(0.5 * np.ones_like(x_line), x_line, 'k--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig('../figures/xor_problem.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_svm_geometry():
    """Generate SVM geometric interpretation with clear data separation."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Generate sample data with clusters closer together but not joining
    np.random.seed(42)
    # Class -1 (Blue): bottom-right region, moved closer to center
    X_class1 = np.array([[3.2, 1.8], [3.6, 2.0], [3.8, 2.2], [3.4, 2.4], [4.0, 2.0]])
    # Class +1 (Red): top-left region, moved closer to center
    X_class2 = np.array([[1.8, 3.2], [2.0, 3.6], [2.2, 3.8], [2.4, 3.4], [2.0, 4.0]])

    ax.scatter(X_class1[:, 0], X_class1[:, 1], c='blue', s=80, alpha=0.8,
               label='Class -1 (y = -1)', edgecolors='darkblue', linewidth=0.5)
    ax.scatter(X_class2[:, 0], X_class2[:, 1], c='red', s=80, alpha=0.8,
               label='Class +1 (y = +1)', edgecolors='darkred', linewidth=0.5)

    # Decision boundary and margins
    x = np.linspace(0, 5, 100)

    # Optimal decision boundary: w^T x + b = 0
    y_boundary = x + 0.3  # Original decision boundary
    mask = (y_boundary >= 0) & (y_boundary <= 5)  # Only plot within bounds
    ax.plot(x[mask], y_boundary[mask], 'k-', linewidth=3,
           label=r'Decision Boundary: $w^T x + b = 0$')

    # Margin boundaries
    y_margin_neg = x - 0.4  # w^T x + b = -1
    y_margin_pos = x + 1.0  # w^T x + b = +1

    mask_neg = (y_margin_neg >= 0) & (y_margin_neg <= 5)
    mask_pos = (y_margin_pos >= 0) & (y_margin_pos <= 5)

    ax.plot(x[mask_neg], y_margin_neg[mask_neg], 'k--', linewidth=2, alpha=0.7,
           label=r'Margin: $w^T x + b = -1$')
    ax.plot(x[mask_pos], y_margin_pos[mask_pos], 'k--', linewidth=2, alpha=0.7,
           label=r'Margin: $w^T x + b = +1$')

    # Identify and highlight support vectors (points closest to decision boundary)
    # For this example, manually select the support vectors
    support_vectors = np.array([[1.8, 1.2], [3.2, 3.5]])  # Original support vectors

    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=200,
               facecolors='none', edgecolors='black', linewidth=3,
               label='Support Vectors', zorder=10)

    # Add margin width annotation
    # Draw perpendicular line showing margin width
    mid_x, mid_y = 2.5, 2.8
    margin_vec_x = 0.35  # Perpendicular to decision boundary
    margin_vec_y = -0.35

    ax.annotate('', xy=(mid_x - margin_vec_x, mid_y - margin_vec_y),
                xytext=(mid_x + margin_vec_x, mid_y + margin_vec_y),
                arrowprops=dict(arrowstyle='<->', lw=2, color='green'))

    ax.text(mid_x + 0.5, mid_y, r'Margin = $\frac{2}{\|w\|}$', fontsize=12,
           color='green', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

    # Add normal vector arrow
    ax.arrow(2.0, 2.3, 0.3, 0.3, head_width=0.1, head_length=0.1,
            fc='purple', ec='purple', linewidth=2)
    ax.text(2.4, 2.7, r'$\vec{w}$', fontsize=14, color='purple', fontweight='bold')

    # Fill margin area
    x_fill = np.linspace(0, 5, 100)
    y_fill_low = x_fill - 0.4
    y_fill_high = x_fill + 1.0

    # Only fill where both boundaries are in bounds
    valid_mask = (y_fill_low >= 0) & (y_fill_high <= 5)
    ax.fill_between(x_fill[valid_mask], y_fill_low[valid_mask], y_fill_high[valid_mask],
                   alpha=0.1, color='gray', label='Margin Zone')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel(r'$x_1$ (Feature 1)', fontsize=14)
    ax.set_ylabel(r'$x_2$ (Feature 2)', fontsize=14)
    ax.set_title('SVM Geometric Interpretation:\nMaximum Margin Classification',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/svm_geometry.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating TikZ replacement figures...")
    plot_supervised_learning_overview()
    plot_linear_vs_nonlinear_data()
    plot_perceptron_vs_svm()
    plot_xor_problem()
    plot_svm_geometry()
    print("TikZ replacement figures saved to ../figures/")