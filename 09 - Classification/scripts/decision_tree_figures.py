#!/usr/bin/env python3
"""
Classification Methods - Decision Tree Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates visualizations for Decision Tree classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.patches import Rectangle, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# PROFESSIONAL STYLING
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

COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#D32F2F',
    'warning': '#F57C00',
    'info': '#0288D1',
    'class1': '#1976D2',
    'class2': '#E53935',
    'class3': '#43A047',
}

def create_output_dir():
    """Create output directory for figures"""
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_tree_structure():
    """Visualize decision tree structure with splits"""
    np.random.seed(42)

    # Generate simple dataset
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               flip_y=0.05, random_state=42)

    # Train a shallow tree
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: Tree structure
    ax = axes[0]
    plot_tree(dt, ax=ax, feature_names=['Feature 1', 'Feature 2'],
             class_names=['Class 0', 'Class 1'], filled=True,
             rounded=True, fontsize=10, proportion=True)

    ax.set_title('Decision Tree Structure (max_depth=3)',
                fontsize=13, fontweight='bold')

    # Right: Algorithm explanation
    ax = axes[1]
    ax.axis('off')

    algorithm_text = """
    Decision Tree Algorithm (CART):

    TRAINING PROCESS:

    1. Start with all data at root node

    2. For each node:
       a) If stopping criterion met:
          → Create leaf node
          → Assign majority class
          → STOP

       b) Otherwise, find best split:
          → Try all features
          → Try all possible thresholds
          → Choose split maximizing
            information gain (or Gini decrease)

    3. Recursively split:
       • Left child:  samples where x ≤ threshold
       • Right child: samples where x > threshold

    SPLITTING CRITERIA:

    Information Gain (Entropy):
    • Entropy: H(S) = -Σ pᵢ log₂(pᵢ)
    • Gain = H(parent) - Σ (nₖ/n) H(child_k)
    • Favors balanced, pure splits

    Gini Impurity:
    • Gini: G(S) = 1 - Σ pᵢ²
    • Decrease = G(parent) - Σ (nₖ/n) G(child_k)
    • Faster to compute
    • Similar results to entropy

    STOPPING CRITERIA:

    • max_depth reached
    • min_samples_split not met
    • min_samples_leaf constraint
    • No improvement in purity
    • All samples same class

    PREDICTION:

    1. Start at root
    2. Follow splits based on features
    3. Reach leaf node
    4. Return leaf's majority class

    Time Complexity:
    • Training: O(n × m × log n)
      n = samples, m = features
    • Prediction: O(log n) average
    """

    ax.text(0.05, 0.95, algorithm_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('Understanding Decision Trees',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/dt_tree_structure.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated dt_tree_structure.png")


def plot_splitting_criteria():
    """Visualize information gain and Gini impurity concepts"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: Impurity measures comparison
    ax = axes[0, 0]

    # Plot impurity measures as function of class probability
    p = np.linspace(0.001, 0.999, 1000)

    # Entropy
    entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)

    # Gini
    gini = 1 - p**2 - (1-p)**2

    # Misclassification error
    misclass = 1 - np.maximum(p, 1-p)

    ax.plot(p, entropy, linewidth=3, label='Entropy',
           color=COLOR_PALETTE['class1'])
    ax.plot(p, gini, linewidth=3, label='Gini Impurity',
           color=COLOR_PALETTE['class2'], linestyle='--')
    ax.plot(p, misclass, linewidth=3, label='Misclassification Error',
           color=COLOR_PALETTE['class3'], linestyle=':')

    ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.3)
    ax.set_xlabel('P(Class = 1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Impurity', fontsize=12, fontweight='bold')
    ax.set_title('Impurity Measures for Binary Classification',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.text(0.5, 0.85, 'Maximum Impurity\n(50-50 split)',
           ha='center', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))

    # Top-right: Example split evaluation
    ax = axes[0, 1]
    ax.axis('off')

    example_text = """
    Example: Evaluating a Split

    Parent Node: 100 samples
    • Class 0: 60 samples
    • Class 1: 40 samples

    BEFORE SPLIT:

    Gini(parent) = 1 - (60/100)² - (40/100)²
                 = 1 - 0.36 - 0.16
                 = 0.48

    Entropy(parent) = -0.6×log₂(0.6) - 0.4×log₂(0.4)
                    = 0.971 bits

    PROPOSED SPLIT: Feature X ≤ 5.3

    Left Child: 70 samples
    • Class 0: 55 samples
    • Class 1: 15 samples

    Gini(left) = 1 - (55/70)² - (15/70)²
               = 0.330

    Right Child: 30 samples
    • Class 0: 5 samples
    • Class 1: 25 samples

    Gini(right) = 1 - (5/30)² - (25/30)²
                = 0.278

    GINI DECREASE:

    Δ Gini = 0.48 - (70/100 × 0.330 + 30/30 × 0.278)
           = 0.48 - 0.314
           = 0.166  ← Information Gain!

    This split REDUCES impurity significantly!
    ✓ Good split!

    The algorithm tries ALL possible splits
    and chooses the one with maximum gain.
    """

    ax.text(0.05, 0.95, example_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['success'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    # Bottom-left: Visualize split on data
    ax = axes[1, 0]

    np.random.seed(42)
    # Create data with clear split
    n = 100
    X_split = np.random.randn(n, 2) * 2
    y_split = (X_split[:, 0] > 0).astype(int)

    # Add some noise
    noise_idx = np.random.choice(n, size=15, replace=False)
    y_split[noise_idx] = 1 - y_split[noise_idx]

    # Plot data
    ax.scatter(X_split[y_split==0, 0], X_split[y_split==0, 1],
              c=COLOR_PALETTE['class1'], s=80, alpha=0.6,
              edgecolors='white', linewidth=1.5, label='Class 0', marker='o')
    ax.scatter(X_split[y_split==1, 0], X_split[y_split==1, 1],
              c=COLOR_PALETTE['class2'], s=80, alpha=0.6,
              edgecolors='white', linewidth=1.5, label='Class 1', marker='s')

    # Best split line
    split_value = 0.0
    ax.axvline(split_value, color='black', linewidth=3, linestyle='--',
              label=f'Best Split: x₁ ≤ {split_value}')

    # Shade regions
    ax.axvspan(X_split[:, 0].min(), split_value, alpha=0.1,
              color=COLOR_PALETTE['class1'])
    ax.axvspan(split_value, X_split[:, 0].max(), alpha=0.1,
              color=COLOR_PALETTE['class2'])

    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title('Visualizing the Best Split',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Count samples in each region
    left_0 = np.sum((X_split[:, 0] <= split_value) & (y_split == 0))
    left_1 = np.sum((X_split[:, 0] <= split_value) & (y_split == 1))
    right_0 = np.sum((X_split[:, 0] > split_value) & (y_split == 0))
    right_1 = np.sum((X_split[:, 0] > split_value) & (y_split == 1))

    ax.text(X_split[:, 0].min() + 1, X_split[:, 1].max() - 1,
           f'Left:\nC0={left_0}\nC1={left_1}',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    alpha=0.8, edgecolor='black', linewidth=2))

    ax.text(X_split[:, 0].max() - 1, X_split[:, 1].max() - 1,
           f'Right:\nC0={right_0}\nC1={right_1}',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    alpha=0.8, edgecolor='black', linewidth=2))

    # Bottom-right: Feature importance concept
    ax = axes[1, 1]
    ax.axis('off')

    importance_text = """
    Feature Importance in Decision Trees:

    CALCULATION:

    For each feature f:

    Importance(f) = Σ (nₜ/n) × ΔImpurity(t)
                    t

    Where:
    • t = nodes that split on feature f
    • nₜ = samples reaching node t
    • n = total training samples
    • ΔImpurity = impurity decrease from split

    INTERPRETATION:

    • Normalized to sum to 1.0
    • Higher values → more important
    • Measures TOTAL reduction in impurity
    • Considers:
      - Number of samples affected
      - Quality of splits
      - Position in tree (earlier = more impact)

    PROPERTIES:

    ✓ Easy to interpret
    ✓ Fast to compute
    ✓ Built into sklearn
    ✓ Helps feature selection

    ⚠ Biased toward:
      - High-cardinality features
      - Features with many possible splits
      - Features used early in tree

    ⚠ Doesn't capture:
      - Feature interactions
      - Non-linear relationships
      - Correlation effects

    USAGE:

    • Feature selection
    • Model interpretation
    • Understanding predictions
    • Domain insights

    Best combined with other methods:
    • Permutation importance
    • SHAP values
    • Partial dependence plots
    """

    ax.text(0.05, 0.95, importance_text, transform=ax.transAxes,
           fontsize=8.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['warning'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('Decision Trees: Splitting Criteria and Information Gain',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/dt_splitting_criteria.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated dt_splitting_criteria.png")


def plot_decision_boundaries():
    """Visualize decision tree decision boundaries"""
    np.random.seed(42)

    # Generate synthetic data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               flip_y=0.1, random_state=42)

    # Different tree depths
    depths = [1, 3, 10]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, depth in enumerate(depths):
        ax = axes[idx]

        # Train decision tree
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X, y)

        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary (axis-aligned boxes)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu_r')
        ax.contour(xx, yy, Z, colors='black', linewidths=1.5, alpha=0.3)

        # Plot training points
        ax.scatter(X[y==0, 0], X[y==0, 1], c=COLOR_PALETTE['class1'],
                  s=60, alpha=0.7, edgecolors='white', linewidth=1.5,
                  label='Class 0', marker='o')
        ax.scatter(X[y==1, 0], X[y==1, 1], c=COLOR_PALETTE['class2'],
                  s=60, alpha=0.7, edgecolors='white', linewidth=1.5,
                  label='Class 1', marker='s')

        # Calculate training accuracy and number of leaves
        train_acc = dt.score(X, y)
        n_leaves = dt.get_n_leaves()

        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(f'max_depth = {depth}\nLeaves: {n_leaves} | Accuracy: {train_acc:.1%}',
                    fontsize=12, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # Add characterization
        if depth == 1:
            char = "Underfitting\n(Too Simple)"
            color = COLOR_PALETTE['warning']
        elif depth == 3:
            char = "Good Balance\n(Generalizes)"
            color = COLOR_PALETTE['success']
        else:
            char = "Overfitting\n(Too Complex)"
            color = COLOR_PALETTE['danger']

        ax.text(0.02, 0.02, char, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                        alpha=0.3, edgecolor='black', linewidth=2))

    plt.suptitle('Decision Tree Boundaries: Axis-Aligned Rectangular Regions',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/dt_decision_boundaries.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated dt_decision_boundaries.png")


def plot_overfitting_demo():
    """Demonstrate overfitting with shallow vs deep trees"""
    np.random.seed(42)

    # Generate data with noise
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               flip_y=0.15, random_state=42)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Train trees with different max_depths
    depths = range(1, 21)
    train_scores = []
    test_scores = []

    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        train_scores.append(dt.score(X_train, y_train))
        test_scores.append(dt.score(X_test, y_test))

    # Top-left: Shallow tree (max_depth=2)
    ax = axes[0, 0]
    dt_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt_shallow.fit(X_train, y_train)

    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z_shallow = dt_shallow.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_shallow = Z_shallow.reshape(xx.shape)

    ax.contourf(xx, yy, Z_shallow, alpha=0.3, cmap='RdYlBu_r')
    ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
              c=COLOR_PALETTE['class1'], s=60, alpha=0.7,
              edgecolors='white', linewidth=1.5, label='Class 0', marker='o')
    ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
              c=COLOR_PALETTE['class2'], s=60, alpha=0.7,
              edgecolors='white', linewidth=1.5, label='Class 1', marker='s')

    train_acc_shallow = dt_shallow.score(X_train, y_train)
    test_acc_shallow = dt_shallow.score(X_test, y_test)

    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title(f'Shallow Tree (depth=2)\nTrain: {train_acc_shallow:.1%} | Test: {test_acc_shallow:.1%}',
                fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Top-right: Deep tree (max_depth=20)
    ax = axes[0, 1]
    dt_deep = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt_deep.fit(X_train, y_train)

    Z_deep = dt_deep.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_deep = Z_deep.reshape(xx.shape)

    ax.contourf(xx, yy, Z_deep, alpha=0.3, cmap='RdYlBu_r')
    ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
              c=COLOR_PALETTE['class1'], s=60, alpha=0.7,
              edgecolors='white', linewidth=1.5, label='Class 0', marker='o')
    ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
              c=COLOR_PALETTE['class2'], s=60, alpha=0.7,
              edgecolors='white', linewidth=1.5, label='Class 1', marker='s')

    train_acc_deep = dt_deep.score(X_train, y_train)
    test_acc_deep = dt_deep.score(X_test, y_test)

    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title(f'Deep Tree (depth=20)\nTrain: {train_acc_deep:.1%} | Test: {test_acc_deep:.1%}',
                fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Training vs Test accuracy
    ax = axes[1, 0]

    ax.plot(depths, train_scores, 'o-', linewidth=3, markersize=6,
           color=COLOR_PALETTE['class1'], label='Training Accuracy', alpha=0.8)
    ax.plot(depths, test_scores, 's-', linewidth=3, markersize=6,
           color=COLOR_PALETTE['class2'], label='Test Accuracy', alpha=0.8)

    # Mark optimal depth
    optimal_depth = depths[np.argmax(test_scores)]
    ax.axvline(optimal_depth, color='black', linestyle='--', linewidth=2.5,
              label=f'Optimal depth={optimal_depth}')

    ax.set_xlabel('Maximum Tree Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting: Training vs Test Performance',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Shade overfitting region
    ax.axvspan(optimal_depth, 20, alpha=0.1, color=COLOR_PALETTE['danger'])
    ax.text(15, 0.88, 'Overfitting\nRegion', ha='center',
           fontsize=10, fontweight='bold', color=COLOR_PALETTE['danger'])

    # Bottom-right: Regularization strategies
    ax = axes[1, 1]
    ax.axis('off')

    regularization_text = """
    Preventing Overfitting in Decision Trees:

    PROBLEM:
    • Deep trees memorize training data
    • Perfect training accuracy (100%)
    • Poor test accuracy
    • High variance, low bias

    REGULARIZATION STRATEGIES:

    1. PRE-PRUNING (Early Stopping)
       Set constraints BEFORE growing tree:

       max_depth
       • Limit tree depth
       • Most common constraint
       • Try: 3-10 for start

       min_samples_split
       • Minimum samples to split node
       • Default: 2
       • Try: 10-50 for large datasets

       min_samples_leaf
       • Minimum samples in leaf
       • Prevents tiny leaves
       • Try: 5-20

       max_leaf_nodes
       • Limit total leaves
       • Controls complexity directly

       min_impurity_decrease
       • Minimum gain to split
       • Prunes weak splits

    2. POST-PRUNING
       Grow full tree, then prune:

       ccp_alpha (cost-complexity)
       • Penalizes tree complexity
       • Higher α → smaller tree
       • Find via cross-validation

    3. ENSEMBLE METHODS
       Combine multiple trees:

       • Random Forest
         - Average many trees
         - Reduces variance
         - Robust to overfitting

       • Gradient Boosting
         - Sequential trees
         - Corrects errors
         - Powerful but needs tuning

    BEST PRACTICES:

    ✓ Start simple (depth 3-5)
    ✓ Use cross-validation
    ✓ Monitor train/test gap
    ✓ Try ensemble methods
    ✓ Feature engineering matters
    """

    ax.text(0.05, 0.95, regularization_text, transform=ax.transAxes,
           fontsize=8.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['danger'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('Decision Trees: Overfitting and Regularization',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/dt_overfitting.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated dt_overfitting.png")


def plot_feature_importance():
    """Visualize feature importance from decision trees"""
    np.random.seed(42)

    # Generate data with varying feature importance
    n_samples = 500
    X = np.random.randn(n_samples, 5)

    # Create target with different feature contributions
    # Feature 0: strong signal
    # Feature 1: moderate signal
    # Feature 2: weak signal
    # Features 3-4: noise
    y = (2 * X[:, 0] + X[:, 1] + 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    feature_names = ['Strong Signal', 'Moderate Signal', 'Weak Signal',
                    'Noise 1', 'Noise 2']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: Feature importance
    ax = axes[0, 0]

    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)

    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]

    colors_importance = [COLOR_PALETTE['success'], COLOR_PALETTE['info'],
                        COLOR_PALETTE['warning'], COLOR_PALETTE['danger'],
                        COLOR_PALETTE['danger']]

    bars = ax.barh(range(len(importances)), importances[indices],
                   color=[colors_importance[i] for i in indices],
                   alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance (max_depth=8)\nTest Accuracy: {dt.score(X_test, y_test):.1%}',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Add value labels
    for i, (idx, bar) in enumerate(zip(indices, bars)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{importances[idx]:.3f}',
               ha='left', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Top-right: Decision tree visualization
    ax = axes[0, 1]

    plot_tree(dt, ax=ax, feature_names=feature_names,
             class_names=['Class 0', 'Class 1'], filled=True,
             rounded=True, fontsize=7, max_depth=3)

    ax.set_title('Tree Structure (showing depth ≤ 3)',
                fontsize=13, fontweight='bold')

    # Bottom-left: Feature importance vs depth
    ax = axes[1, 0]

    depths = [2, 5, 10, 15, 20]
    importance_evolution = []

    for depth in depths:
        dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt_temp.fit(X_train, y_train)
        importance_evolution.append(dt_temp.feature_importances_)

    importance_evolution = np.array(importance_evolution)

    for i, name in enumerate(feature_names):
        color = colors_importance[i]
        ax.plot(depths, importance_evolution[:, i], 'o-', linewidth=2.5,
               markersize=8, label=name, color=color, alpha=0.8)

    ax.set_xlabel('Maximum Tree Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Evolution with Tree Depth',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Comparison of tree complexities
    ax = axes[1, 1]

    depths_compare = [3, 5, 8, 12, 20]
    metrics = []

    for depth in depths_compare:
        dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt_temp.fit(X_train, y_train)
        metrics.append({
            'depth': depth,
            'leaves': dt_temp.get_n_leaves(),
            'train_acc': dt_temp.score(X_train, y_train),
            'test_acc': dt_temp.score(X_test, y_test)
        })

    x_pos = np.arange(len(depths_compare))
    width = 0.35

    train_accs = [m['train_acc'] for m in metrics]
    test_accs = [m['test_acc'] for m in metrics]

    ax.bar(x_pos - width/2, train_accs, width, label='Train Accuracy',
          color=COLOR_PALETTE['class1'], alpha=0.7, edgecolor='black', linewidth=2)
    ax.bar(x_pos + width/2, test_accs, width, label='Test Accuracy',
          color=COLOR_PALETTE['class2'], alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_xlabel('Maximum Tree Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Complexity vs Performance',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"d={d}\nL={metrics[i]['leaves']}"
                        for i, d in enumerate(depths_compare)])
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.05])

    # Add labels
    for i, x in enumerate(x_pos):
        ax.text(x - width/2, train_accs[i] + 0.01, f'{train_accs[i]:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(x + width/2, test_accs[i] + 0.01, f'{test_accs[i]:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Decision Tree: Feature Importance Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/dt_feature_importance.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated dt_feature_importance.png")


def plot_iris_example():
    """Real-world example: Decision Tree on Iris dataset"""
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Find optimal max_depth via cross-validation
    from sklearn.model_selection import cross_val_score
    depths = range(1, 16)
    cv_scores = []
    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(dt, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())

    optimal_depth = depths[np.argmax(cv_scores)]

    # Train final model
    dt = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
    dt.fit(X_train, y_train)

    # Predictions
    y_pred = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: Tree structure
    ax = axes[0, 0]

    plot_tree(dt, ax=ax, feature_names=feature_names,
             class_names=target_names, filled=True,
             rounded=True, fontsize=8)

    ax.set_title(f'Decision Tree Structure (max_depth={optimal_depth})',
                fontsize=13, fontweight='bold')

    # Top-right: max_depth selection
    ax = axes[0, 1]

    ax.plot(depths, cv_scores, 'o-', linewidth=3, markersize=8,
           color=COLOR_PALETTE['primary'], alpha=0.8)
    ax.axvline(optimal_depth, color=COLOR_PALETTE['success'], linestyle='--',
              linewidth=2.5, label=f'Optimal depth={optimal_depth}')
    ax.axhline(max(cv_scores), color=COLOR_PALETTE['danger'], linestyle=':',
              linewidth=2, alpha=0.5, label=f'Best CV Score: {max(cv_scores):.3f}')

    ax.set_xlabel('Maximum Tree Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Tuning: Finding Optimal Depth',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Confusion matrix
    ax = axes[1, 0]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            ax.text(j, i, f'{cm[i, j]}',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix\nTest Accuracy: {accuracy:.1%}',
                fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom-right: Feature importance
    ax = axes[1, 1]

    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]

    colors = [COLOR_PALETTE['class1'], COLOR_PALETTE['class2'],
             COLOR_PALETTE['class3'], COLOR_PALETTE['accent']]

    bars = ax.barh(range(len(importances)), importances[indices],
                   color=[colors[i] for i in range(len(importances))],
                   alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance in Iris Classification',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Add value labels and percentages
    total_importance = sum(importances)
    for idx, bar in zip(indices, bars):
        width = bar.get_width()
        percentage = (importances[idx] / total_importance) * 100
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {importances[idx]:.3f} ({percentage:.1f}%)',
               ha='left', va='center', fontsize=10, fontweight='bold')

    # Add performance summary
    report = classification_report(y_test, y_pred, target_names=target_names,
                                   output_dict=True, zero_division=0)

    summary = f"""
    Model Summary:
    • Optimal Depth: {optimal_depth}
    • Number of Leaves: {dt.get_n_leaves()}
    • Test Accuracy: {accuracy:.1%}
    • Macro F1-Score: {report['macro avg']['f1-score']:.3f}

    Key Insights:
    • {feature_names[indices[0]]} most important
    • Tree is interpretable and compact
    • High accuracy on test set
    • Minimal overfitting
    """

    ax.text(0.98, 0.02, summary, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           family='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_PALETTE['success'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('Decision Tree on Iris Dataset: Complete Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/dt_iris_example.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated dt_iris_example.png")


def main():
    """Generate all Decision Tree figures"""
    print("="*60)
    print("Generating Decision Tree Figures")
    print("="*60)

    plot_tree_structure()
    plot_splitting_criteria()
    plot_decision_boundaries()
    plot_overfitting_demo()
    plot_feature_importance()
    plot_iris_example()

    print("="*60)
    print("✅ Decision Tree figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
