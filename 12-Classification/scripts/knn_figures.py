#!/usr/bin/env python3
"""
Classification Methods - K-Nearest Neighbors Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates visualizations for K-Nearest Neighbors classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.patches import Circle, FancyBboxPatch
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


def plot_knn_concept():
    """Visualize KNN concept with k neighbors voting"""
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Generate simple data
    class1_points = np.random.randn(20, 2) + np.array([2, 2])
    class2_points = np.random.randn(20, 2) + np.array([5, 5])

    # Test point
    test_point = np.array([3.5, 3.8])

    # Left: k=3 example
    ax = axes[0]

    # Plot training data
    ax.scatter(class1_points[:, 0], class1_points[:, 1],
              c=COLOR_PALETTE['class1'], s=100, alpha=0.6,
              edgecolors='white', linewidth=2, label='Class 0 (Blue)', marker='o')
    ax.scatter(class2_points[:, 0], class2_points[:, 1],
              c=COLOR_PALETTE['class2'], s=100, alpha=0.6,
              edgecolors='white', linewidth=2, label='Class 1 (Red)', marker='s')

    # Plot test point
    ax.scatter(test_point[0], test_point[1], c='gold', s=300,
              edgecolors='black', linewidth=3, label='Test Point', marker='*', zorder=5)

    # Calculate distances to all points
    all_points = np.vstack([class1_points, class2_points])
    all_labels = np.array([0]*20 + [1]*20)
    distances = np.sqrt(np.sum((all_points - test_point)**2, axis=1))

    # Find k=3 nearest neighbors
    k = 3
    nearest_indices = np.argsort(distances)[:k]
    nearest_points = all_points[nearest_indices]
    nearest_labels = all_labels[nearest_indices]

    # Draw circles to nearest neighbors
    for i, (point, label) in enumerate(zip(nearest_points, nearest_labels)):
        color = COLOR_PALETTE['class1'] if label == 0 else COLOR_PALETTE['class2']
        ax.plot([test_point[0], point[0]], [test_point[1], point[1]],
               'k--', linewidth=2, alpha=0.5, zorder=1)
        circle = Circle(point, 0.15, color=color, fill=True,
                       edgecolor='black', linewidth=3, zorder=4)
        ax.add_patch(circle)

    # Add voting result
    votes_class0 = np.sum(nearest_labels == 0)
    votes_class1 = np.sum(nearest_labels == 1)
    prediction = 0 if votes_class0 > votes_class1 else 1
    pred_color = COLOR_PALETTE['class1'] if prediction == 0 else COLOR_PALETTE['class2']

    vote_text = f"Votes:\nClass 0: {votes_class0}\nClass 1: {votes_class1}\n\nPrediction: Class {prediction}"
    ax.text(0.02, 0.98, vote_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8', facecolor=pred_color,
                    alpha=0.3, edgecolor='black', linewidth=2))

    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title(f'KNN Classification with k={k}',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Right: Algorithm explanation
    ax = axes[1]
    ax.axis('off')

    algorithm_text = """
    K-Nearest Neighbors (KNN) Algorithm:

    1. Choose k (number of neighbors)
       • Small k → more complex boundaries
       • Large k → smoother boundaries

    2. Calculate Distance
       • Compute distance from test point to
         all training points
       • Common: Euclidean distance
         d(x, x') = √(Σ(xᵢ - x'ᵢ)²)

    3. Find k Nearest Neighbors
       • Select k points with smallest distances
       • These neighbors "vote" on classification

    4. Majority Vote
       • Count votes from k neighbors
       • Assign test point to majority class
       • Ties broken by: nearest neighbor,
         random selection, or prior probability

    Key Properties:

    ✓ Non-parametric (no training phase)
    ✓ Instance-based learning
    ✓ Simple and intuitive
    ✓ Works well with small datasets

    ✗ Slow prediction (O(n) per point)
    ✗ Memory intensive (stores all data)
    ✗ Sensitive to irrelevant features
    ✗ Curse of dimensionality

    Hyperparameter: k
    • k = 1: Very flexible, high variance
    • k = n: Very rigid, high bias
    • Optimal k found via cross-validation
    """

    ax.text(0.05, 0.95, algorithm_text, transform=ax.transAxes,
           fontsize=9.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('Understanding K-Nearest Neighbors (KNN)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/knn_concept.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated knn_concept.png")


def plot_knn_decision_boundaries():
    """Visualize KNN decision boundaries with different k values"""
    np.random.seed(42)

    # Generate synthetic data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               flip_y=0.1, random_state=42)

    # Different k values
    k_values = [1, 5, 15]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)

        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu_r')
        ax.contour(xx, yy, Z, colors='black', linewidths=1.5, alpha=0.3)

        # Plot training points
        ax.scatter(X[y==0, 0], X[y==0, 1], c=COLOR_PALETTE['class1'],
                  s=60, alpha=0.7, edgecolors='white', linewidth=1.5,
                  label='Class 0', marker='o')
        ax.scatter(X[y==1, 0], X[y==1, 1], c=COLOR_PALETTE['class2'],
                  s=60, alpha=0.7, edgecolors='white', linewidth=1.5,
                  label='Class 1', marker='s')

        # Calculate training accuracy
        train_acc = knn.score(X, y)

        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(f'k = {k}\nTraining Accuracy: {train_acc:.1%}',
                    fontsize=13, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
        ax.grid(True, alpha=0.3)

        # Add characterization
        if k == 1:
            char = "Very Complex\n(High Variance)"
            color = COLOR_PALETTE['danger']
        elif k == 5:
            char = "Balanced\n(Good Trade-off)"
            color = COLOR_PALETTE['success']
        else:
            char = "Very Smooth\n(High Bias)"
            color = COLOR_PALETTE['warning']

        ax.text(0.02, 0.02, char, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                        alpha=0.3, edgecolor='black', linewidth=2))

    plt.suptitle('KNN Decision Boundaries: Effect of k',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/knn_decision_boundaries.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated knn_decision_boundaries.png")


def plot_knn_bias_variance():
    """Visualize effect of k on bias-variance trade-off"""
    np.random.seed(42)

    # Generate synthetic data
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               flip_y=0.15, random_state=42)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Test different k values
    k_range = range(1, 51)
    train_scores = []
    test_scores = []
    cv_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))

        # Cross-validation score
        cv_score = cross_val_score(knn, X_train, y_train, cv=5).mean()
        cv_scores.append(cv_score)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Training vs Test accuracy
    ax = axes[0]

    ax.plot(k_range, train_scores, 'o-', linewidth=3, markersize=6,
           color=COLOR_PALETTE['class1'], label='Training Accuracy', alpha=0.8)
    ax.plot(k_range, test_scores, 's-', linewidth=3, markersize=6,
           color=COLOR_PALETTE['class2'], label='Test Accuracy', alpha=0.8)
    ax.plot(k_range, cv_scores, '^-', linewidth=3, markersize=6,
           color=COLOR_PALETTE['success'], label='CV Accuracy (5-fold)', alpha=0.8)

    # Find optimal k
    optimal_k = k_range[np.argmax(test_scores)]
    ax.axvline(optimal_k, color='black', linestyle='--', linewidth=2.5,
              label=f'Optimal k={optimal_k}')

    ax.set_xlabel('k (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance vs k',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Add regions
    ax.axvspan(1, 5, alpha=0.1, color=COLOR_PALETTE['danger'])
    ax.text(3, 0.88, 'High Variance\n(Overfitting)', ha='center',
           fontsize=9, fontweight='bold', color=COLOR_PALETTE['danger'])

    ax.axvspan(35, 50, alpha=0.1, color=COLOR_PALETTE['warning'])
    ax.text(42.5, 0.88, 'High Bias\n(Underfitting)', ha='center',
           fontsize=9, fontweight='bold', color=COLOR_PALETTE['warning'])

    # Right: Bias-Variance explanation
    ax = axes[1]
    ax.axis('off')

    explanation = """
    Bias-Variance Trade-off in KNN:

    SMALL k (e.g., k=1):
    • Low Bias
      - Model is very flexible
      - Can capture complex patterns
      - Fits training data closely

    • High Variance
      - Sensitive to noise
      - Decision boundary wiggles
      - Poor generalization
      - Overfits to training data

    LARGE k (e.g., k=50):
    • High Bias
      - Model is too simple
      - Smooths over important patterns
      - Underfits training data

    • Low Variance
      - Stable predictions
      - Less sensitive to noise
      - Decision boundary smooth
      - May miss important patterns

    OPTIMAL k:
    • Balances bias and variance
    • Best generalization performance
    • Found via cross-validation
    • Depends on:
      - Dataset size
      - Noise level
      - Problem complexity
      - Number of features

    Rule of Thumb:
    • k ≈ √n (where n = training samples)
    • Always use odd k for binary problems
      (avoids ties)
    """

    ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
           fontsize=9.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('KNN: Bias-Variance Trade-off',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/knn_bias_variance.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated knn_bias_variance.png")


def plot_distance_metrics():
    """Compare different distance metrics in KNN"""
    np.random.seed(42)

    # Generate synthetic data with specific structure
    X, y = make_classification(n_samples=150, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               flip_y=0.1, random_state=42)

    # Scale features differently to emphasize metric differences
    X[:, 0] = X[:, 0] * 2
    X[:, 1] = X[:, 1] * 0.5

    # Distance metrics to compare
    metrics = [
        ('euclidean', 'Euclidean\nd = √(Σ(xᵢ-yᵢ)²)'),
        ('manhattan', 'Manhattan (L1)\nd = Σ|xᵢ-yᵢ|'),
        ('minkowski', 'Minkowski (p=3)\nd = (Σ|xᵢ-yᵢ|ᵖ)^(1/p)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    k = 5

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]

        # Train KNN with specific metric
        if metric == 'minkowski':
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, p=3)
        else:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X, y)

        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu_r')

        # Plot training points
        ax.scatter(X[y==0, 0], X[y==0, 1], c=COLOR_PALETTE['class1'],
                  s=70, alpha=0.7, edgecolors='white', linewidth=1.5,
                  label='Class 0', marker='o')
        ax.scatter(X[y==1, 0], X[y==1, 1], c=COLOR_PALETTE['class2'],
                  s=70, alpha=0.7, edgecolors='white', linewidth=1.5,
                  label='Class 1', marker='s')

        # Calculate accuracy
        accuracy = knn.score(X, y)

        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(f'{label}\nAccuracy: {accuracy:.1%}',
                    fontsize=12, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
        ax.grid(True, alpha=0.3)

    # Bottom right: Distance metrics explanation
    ax = axes[3]
    ax.axis('off')

    explanation = """
    Distance Metrics in KNN:

    1. EUCLIDEAN DISTANCE (L2 norm)
       d(x, y) = √(Σᵢ(xᵢ - yᵢ)²)

       • Most common metric
       • "As the crow flies"
       • Sensitive to feature scales
       • Dominated by large differences
       • Best for: continuous features

    2. MANHATTAN DISTANCE (L1 norm)
       d(x, y) = Σᵢ|xᵢ - yᵢ|

       • Also called "city block" distance
       • Sum of absolute differences
       • Less sensitive to outliers
       • Works in grid-like spaces
       • Best for: discrete features

    3. MINKOWSKI DISTANCE (Lp norm)
       d(x, y) = (Σᵢ|xᵢ - yᵢ|ᵖ)^(1/p)

       • Generalization of L1 and L2
       • p=1: Manhattan
       • p=2: Euclidean
       • p→∞: Chebyshev (max difference)
       • Flexible for different problems

    IMPORTANT CONSIDERATIONS:

    ⚠ Feature Scaling Required!
       • Different scales → biased distances
       • Always standardize features
       • z = (x - μ) / σ

    ⚠ Curse of Dimensionality
       • Distances become similar in high-D
       • All points equidistant
       • Use dimensionality reduction

    ⚠ Choosing the Right Metric
       • Domain knowledge matters
       • Cross-validate different metrics
       • Consider feature types
    """

    ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('KNN: Comparing Distance Metrics',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/knn_distance_metrics.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated knn_distance_metrics.png")


def plot_iris_example():
    """Real-world example: KNN on Iris dataset"""
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Use only 2 features for visualization
    X_2d = X[:, [0, 2]]  # sepal length and petal length

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Find optimal k
    k_range = range(1, 31)
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
        cv_scores.append(scores.mean())

    optimal_k = k_range[np.argmax(cv_scores)]

    # Train final model
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = knn.predict(X_test_scaled)
    accuracy = knn.score(X_test_scaled, y_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: Decision boundary
    ax = axes[0, 0]

    # Create mesh
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu_r')

    # Plot training points
    colors = [COLOR_PALETTE['class1'], COLOR_PALETTE['class2'], COLOR_PALETTE['class3']]
    markers = ['o', 's', '^']
    for i, (color, marker, name) in enumerate(zip(colors, markers, target_names)):
        ax.scatter(X_train_scaled[y_train==i, 0], X_train_scaled[y_train==i, 1],
                  c=color, s=70, alpha=0.7, edgecolors='white', linewidth=1.5,
                  label=name, marker=marker)

    ax.set_xlabel(f'{feature_names[0]} (scaled)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{feature_names[2]} (scaled)', fontsize=12, fontweight='bold')
    ax.set_title(f'KNN Decision Boundaries (k={optimal_k})',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Top-right: k selection
    ax = axes[0, 1]

    ax.plot(k_range, cv_scores, 'o-', linewidth=3, markersize=8,
           color=COLOR_PALETTE['primary'], alpha=0.8)
    ax.axvline(optimal_k, color=COLOR_PALETTE['success'], linestyle='--',
              linewidth=2.5, label=f'Optimal k={optimal_k}')
    ax.axhline(max(cv_scores), color=COLOR_PALETTE['danger'], linestyle=':',
              linewidth=2, alpha=0.5, label=f'Best CV Score: {max(cv_scores):.3f}')

    ax.set_xlabel('k (Number of Neighbors)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Tuning: Finding Optimal k',
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

    # Bottom-right: Performance metrics
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate per-class metrics
    report = classification_report(y_test, y_pred, target_names=target_names,
                                   output_dict=True, zero_division=0)

    metrics_text = f"""
    KNN Classification on Iris Dataset

    Configuration:
    • Features: {feature_names[0]}, {feature_names[2]}
    • Optimal k: {optimal_k} (via 5-fold CV)
    • Distance: Euclidean (scaled features)
    • Train/Test Split: 70/30

    Overall Performance:
    • Test Accuracy: {accuracy:.1%}
    • Training samples: {len(X_train)}
    • Test samples: {len(X_test)}

    Per-Class Performance:

    """

    for i, name in enumerate(target_names):
        precision = report[name]['precision']
        recall = report[name]['recall']
        f1 = report[name]['f1-score']
        support = int(report[name]['support'])

        metrics_text += f"""    {name}:
      Precision: {precision:.3f}
      Recall:    {recall:.3f}
      F1-Score:  {f1:.3f}
      Support:   {support}

    """

    metrics_text += f"""
    Macro Average:
    • Precision: {report['macro avg']['precision']:.3f}
    • Recall:    {report['macro avg']['recall']:.3f}
    • F1-Score:  {report['macro avg']['f1-score']:.3f}

    Key Observations:
    • KNN performs excellently on Iris
    • Clear separation between species
    • Some confusion between similar classes
    • Feature scaling is critical!
    """

    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['success'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('KNN on Iris Dataset: Complete Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/knn_iris_example.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated knn_iris_example.png")


def main():
    """Generate all KNN figures"""
    print("="*60)
    print("Generating K-Nearest Neighbors Figures")
    print("="*60)

    plot_knn_concept()
    plot_knn_decision_boundaries()
    plot_knn_bias_variance()
    plot_distance_metrics()
    plot_iris_example()

    print("="*60)
    print("✅ KNN figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
