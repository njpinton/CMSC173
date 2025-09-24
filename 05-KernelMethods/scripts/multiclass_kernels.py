#!/usr/bin/env python3
"""
Generate multiclass kernel methods visualization figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def generate_multiclass_data():
    """Generate multiclass dataset for demonstration."""
    X, y = make_blobs(n_samples=300, centers=4, n_features=2,
                      random_state=42, cluster_std=1.5)
    return X, y

def plot_multiclass_strategies():
    """Compare One-vs-Rest and One-vs-One strategies."""
    X, y = generate_multiclass_data()

    # One-vs-Rest
    ovr_clf = OneVsRestClassifier(svm.SVC(kernel='rbf', gamma=0.5, C=1.0))
    ovr_clf.fit(X, y)

    # One-vs-One
    ovo_clf = OneVsOneClassifier(svm.SVC(kernel='rbf', gamma=0.5, C=1.0))
    ovo_clf.fit(X, y)

    # Direct multiclass SVM
    mc_clf = svm.SVC(kernel='rbf', gamma=0.5, C=1.0, decision_function_shape='ovr')
    mc_clf.fit(X, y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    classifiers = [ovr_clf, ovo_clf, mc_clf]
    titles = ['One-vs-Rest (OvR)', 'One-vs-One (OvO)', 'Direct Multiclass']

    # Create mesh for decision boundaries
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    for i, (clf, title) in enumerate(zip(classifiers, titles)):
        # Plot decision boundary
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[i].contourf(xx, yy, Z, alpha=0.4, cmap='Set3')
        axes[i].contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

        # Plot data points
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='Set3', s=60, alpha=0.8, edgecolors='black')

        axes[i].set_xlabel('Feature 1', fontsize=12)
        axes[i].set_ylabel('Feature 2', fontsize=12)
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/multiclass_strategies.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ovr_detailed():
    """Detailed view of One-vs-Rest strategy."""
    X, y = generate_multiclass_data()

    # Create One-vs-Rest classifier
    ovr_clf = OneVsRestClassifier(svm.SVC(kernel='rbf', gamma=0.5, C=1.0, probability=True))
    ovr_clf.fit(X, y)

    n_classes = len(np.unique(y))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    for class_idx in range(n_classes):
        # Get binary classifier for this class
        binary_clf = ovr_clf.estimators_[class_idx]

        # Create binary labels (class vs rest)
        y_binary = (y == class_idx).astype(int)

        # Get decision function
        Z = binary_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margin
        axes[class_idx].contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        axes[class_idx].contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
                                colors=['lightcoral', 'lightblue'], alpha=0.4)

        # Plot data points
        colors = ['red' if label == 0 else 'blue' for label in y_binary]
        scatter = axes[class_idx].scatter(X[:, 0], X[:, 1], c=colors, s=60, alpha=0.8, edgecolors='black')

        # Highlight support vectors
        if hasattr(binary_clf, 'support_vectors_'):
            axes[class_idx].scatter(binary_clf.support_vectors_[:, 0],
                                   binary_clf.support_vectors_[:, 1],
                                   s=120, facecolors='none', edgecolors='black', linewidth=2)

        axes[class_idx].set_xlabel('Feature 1', fontsize=11)
        axes[class_idx].set_ylabel('Feature 2', fontsize=11)
        axes[class_idx].set_title(f'Class {class_idx} vs Rest', fontsize=12, fontweight='bold')
        axes[class_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/ovr_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_kernel_multiclass_comparison():
    """Compare different kernels for multiclass classification."""
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              n_classes=3, random_state=42)

    kernels = ['linear', 'poly', 'rbf']
    kernel_names = ['Linear', 'Polynomial (degree=3)', 'RBF (γ=1.0)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        if kernel == 'poly':
            clf = svm.SVC(kernel=kernel, degree=3, C=1.0)
        elif kernel == 'rbf':
            clf = svm.SVC(kernel=kernel, gamma=1.0, C=1.0)
        else:
            clf = svm.SVC(kernel=kernel, C=1.0)

        clf.fit(X, y)

        # Plot decision boundary
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[i].contourf(xx, yy, Z, alpha=0.4, cmap='Set3')
        axes[i].contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

        # Plot data points
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='Set3', s=60, alpha=0.8, edgecolors='black')

        # Plot support vectors
        axes[i].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                       s=120, facecolors='none', edgecolors='black', linewidth=2)

        axes[i].set_xlabel('Feature 1', fontsize=12)
        axes[i].set_ylabel('Feature 2', fontsize=12)
        axes[i].set_title(f'{name} Kernel', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

        # Add accuracy score
        accuracy = clf.score(X, y)
        axes[i].text(0.05, 0.95, f'Accuracy: {accuracy:.3f}',
                    transform=axes[i].transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')

    plt.tight_layout()
    plt.savefig('../figures/kernel_multiclass_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiclass_confidence():
    """Visualize confidence/probability for multiclass predictions."""
    X, y = generate_multiclass_data()

    # Train SVM with probability estimates
    clf = svm.SVC(kernel='rbf', gamma=0.5, C=1.0, probability=True)
    clf.fit(X, y)

    # Create mesh
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Get probability predictions
    Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z_pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    n_classes = len(np.unique(y))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    # Plot probability for each class
    for class_idx in range(n_classes):
        Z_class = Z_proba[:, class_idx].reshape(xx.shape)

        im = axes[class_idx].contourf(xx, yy, Z_class, levels=20, cmap='Blues', alpha=0.8)
        axes[class_idx].contour(xx, yy, Z_class, levels=[0.5], colors='red', linewidths=2)

        # Plot data points
        scatter = axes[class_idx].scatter(X[:, 0], X[:, 1], c=y, cmap='Set3', s=60, alpha=0.8, edgecolors='black')

        axes[class_idx].set_xlabel('Feature 1', fontsize=11)
        axes[class_idx].set_ylabel('Feature 2', fontsize=11)
        axes[class_idx].set_title(f'P(Class {class_idx})', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[class_idx])

    # Plot final prediction with confidence
    Z_max_proba = np.max(Z_proba, axis=1).reshape(xx.shape)
    Z_pred = Z_pred.reshape(xx.shape)

    im = axes[3].contourf(xx, yy, Z_max_proba, levels=20, cmap='viridis', alpha=0.6)
    axes[3].contour(xx, yy, Z_pred, colors='black', linewidths=1, alpha=0.8)

    scatter = axes[3].scatter(X[:, 0], X[:, 1], c=y, cmap='Set3', s=60, alpha=0.8, edgecolors='black')

    axes[3].set_xlabel('Feature 1', fontsize=11)
    axes[3].set_ylabel('Feature 2', fontsize=11)
    axes[3].set_title('Prediction Confidence', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[3], label='Max Probability')

    plt.tight_layout()
    plt.savefig('../figures/multiclass_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating multiclass kernel figures...")
    plot_multiclass_strategies()
    plot_ovr_detailed()
    plot_kernel_multiclass_comparison()
    plot_multiclass_confidence()
    print("Multiclass kernel figures saved to ../figures/")