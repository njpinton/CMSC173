#!/usr/bin/env python3
"""
Classification Methods - Naïve Bayes Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates visualizations for Naïve Bayes classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import make_classification, load_iris, fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import norm
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


def plot_bayes_theorem():
    """Visualize Bayes' theorem concept"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Conditional probability visualization
    ax = axes[0]

    # Create overlapping distributions
    x = np.linspace(-4, 8, 1000)

    # Class 0 (negative)
    mu0, sigma0 = 0, 1.2
    p_x_given_c0 = norm.pdf(x, mu0, sigma0)

    # Class 1 (positive)
    mu1, sigma1 = 3, 1.5
    p_x_given_c1 = norm.pdf(x, mu1, sigma1)

    # Prior probabilities
    prior_c0 = 0.6
    prior_c1 = 0.4

    # Plot likelihoods
    ax.fill_between(x, p_x_given_c0, alpha=0.3, color=COLOR_PALETTE['class1'],
                     label=f'P(x|C=0) · P(C=0) = {prior_c0}')
    ax.plot(x, p_x_given_c0, color=COLOR_PALETTE['class1'], linewidth=3)

    ax.fill_between(x, p_x_given_c1, alpha=0.3, color=COLOR_PALETTE['class2'],
                     label=f'P(x|C=1) · P(C=1) = {prior_c1}')
    ax.plot(x, p_x_given_c1, color=COLOR_PALETTE['class2'], linewidth=3)

    # Decision boundary
    # Find intersection (approximately)
    intersection_idx = np.argmin(np.abs(p_x_given_c0 * prior_c0 - p_x_given_c1 * prior_c1))
    decision_x = x[intersection_idx]

    ax.axvline(decision_x, color='black', linestyle='--', linewidth=2.5,
              label=f'Decision Boundary (x={decision_x:.2f})')

    ax.set_xlabel('Feature Value (x)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Bayes Theorem: Likelihood × Prior → Posterior',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Add text annotations
    ax.text(mu0, max(p_x_given_c0)*0.8, 'Class 0\n(Negative)',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['class1'],
                    alpha=0.7, edgecolor='black', linewidth=2))

    ax.text(mu1, max(p_x_given_c1)*0.8, 'Class 1\n(Positive)',
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['class2'],
                    alpha=0.7, edgecolor='black', linewidth=2))

    # Right: Formula breakdown
    ax = axes[1]
    ax.axis('off')

    formula_text = """
    Bayes' Theorem:

    P(C|x) = P(x|C) × P(C) / P(x)

    Where:
    • P(C|x)  = Posterior (what we want)
               = Probability of class C given features x

    • P(x|C)  = Likelihood
               = Probability of observing x given class C

    • P(C)    = Prior
               = Initial probability of class C

    • P(x)    = Evidence (normalization)
               = Total probability of observing x

    Naïve Bayes Assumption:

    Features are conditionally independent given class:

    P(x₁, x₂, ..., xₙ | C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)

    This simplifies computation dramatically!

    Classification Rule:

    ŷ = argmax P(C) × ∏ P(xᵢ|C)
         C            i

    Choose class with highest posterior probability.
    """

    ax.text(0.05, 0.95, formula_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('Understanding Naïve Bayes: Bayes\' Theorem',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/nb_bayes_theorem.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated nb_bayes_theorem.png")


def plot_gaussian_nb():
    """Visualize Gaussian Naïve Bayes decision boundaries"""
    # Generate synthetic data
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               flip_y=0.1, random_state=42)

    # Train Gaussian NB
    gnb = GaussianNB()
    gnb.fit(X, y)

    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Get prediction probabilities
    Z_proba = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Decision boundary
    ax = axes[0]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu_r')

    # Plot training points
    scatter0 = ax.scatter(X[y==0, 0], X[y==0, 1], c=COLOR_PALETTE['class1'],
                         s=80, alpha=0.7, edgecolors='white', linewidth=1.5,
                         label='Class 0', marker='o')
    scatter1 = ax.scatter(X[y==1, 0], X[y==1, 1], c=COLOR_PALETTE['class2'],
                         s=80, alpha=0.7, edgecolors='white', linewidth=1.5,
                         label='Class 1', marker='s')

    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title('Gaussian Naïve Bayes Decision Boundary',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Right: Probability contours
    ax = axes[1]
    contour = ax.contourf(xx, yy, Z_proba, levels=20, cmap='RdYlBu_r', alpha=0.8)
    ax.contour(xx, yy, Z_proba, levels=[0.5], colors='black', linewidths=3,
              linestyles='--')

    # Plot training points
    ax.scatter(X[y==0, 0], X[y==0, 1], c=COLOR_PALETTE['class1'],
              s=80, alpha=0.7, edgecolors='white', linewidth=1.5,
              label='Class 0', marker='o')
    ax.scatter(X[y==1, 0], X[y==1, 1], c=COLOR_PALETTE['class2'],
              s=80, alpha=0.7, edgecolors='white', linewidth=1.5,
              label='Class 1', marker='s')

    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Probability P(y=1|x)',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('P(Class=1)', fontsize=11, fontweight='bold')

    plt.suptitle('Gaussian Naïve Bayes Classification',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/nb_gaussian_decision.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated nb_gaussian_decision.png")


def plot_naive_assumption():
    """Visualize the naïve independence assumption"""
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Generate correlated data
    mean = [0, 0]

    # Strong correlation (violates naïve assumption)
    cov_strong = [[1, 0.8], [0.8, 1]]
    X_strong = np.random.multivariate_normal(mean, cov_strong, 500)

    # Weak correlation (closer to naïve assumption)
    cov_weak = [[1, 0.1], [0.1, 1]]
    X_weak = np.random.multivariate_normal(mean, cov_weak, 500)

    # Plot 1: Strongly correlated features
    ax = axes[0, 0]
    ax.scatter(X_strong[:, 0], X_strong[:, 1], alpha=0.5, s=30,
              c=COLOR_PALETTE['danger'], edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax.set_title('Strong Correlation (r=0.8)\n❌ Violates Naïve Assumption',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Compute actual correlation
    corr_strong = np.corrcoef(X_strong.T)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr_strong:.3f}',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Plot 2: Weakly correlated features
    ax = axes[0, 1]
    ax.scatter(X_weak[:, 0], X_weak[:, 1], alpha=0.5, s=30,
              c=COLOR_PALETTE['success'], edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax.set_title('Weak Correlation (r=0.1)\n✓ Naïve Assumption OK',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    corr_weak = np.corrcoef(X_weak.T)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr_weak:.3f}',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Plot 3: Explanation of assumption
    ax = axes[1, 0]
    ax.axis('off')

    explanation = """
    The Naïve Independence Assumption:

    Features are conditionally independent
    given the class label.

    Mathematical Expression:

    P(x₁, x₂ | C) = P(x₁ | C) × P(x₂ | C)

    Why "Naïve"?
    • This assumption is often violated
    • Real features are usually correlated
    • Yet NB still works surprisingly well!

    When it works well:
    ✓ Features weakly correlated
    ✓ High-dimensional data (text)
    ✓ Large datasets
    ✓ Fast predictions needed

    When it struggles:
    ✗ Strong feature dependencies
    ✗ Complex interactions
    ✗ Small datasets
    """

    ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
           fontsize=9.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    # Plot 4: Performance comparison
    ax = axes[1, 1]

    # Create binary classification problem
    y_strong = (X_strong[:, 0] + X_strong[:, 1] > 0).astype(int)
    y_weak = (X_weak[:, 0] + X_weak[:, 1] > 0).astype(int)

    # Train NB on both
    gnb_strong = GaussianNB().fit(X_strong, y_strong)
    gnb_weak = GaussianNB().fit(X_weak, y_weak)

    # Scores
    score_strong = gnb_strong.score(X_strong, y_strong)
    score_weak = gnb_weak.score(X_weak, y_weak)

    bars = ax.bar(['Strong\nCorrelation', 'Weak\nCorrelation'],
                  [score_strong, score_weak],
                  color=[COLOR_PALETTE['danger'], COLOR_PALETTE['success']],
                  alpha=0.7, edgecolor='black', linewidth=2)

    ax.set_ylabel('Classification Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Naïve Bayes Performance vs Feature Correlation',
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, score in zip(bars, [score_strong, score_weak]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.1%}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')

    plt.suptitle('Understanding the Naïve Independence Assumption',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/nb_naive_assumption.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated nb_naive_assumption.png")


def plot_iris_example():
    """Real-world example: Naïve Bayes on Iris dataset"""
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

    # Train Gaussian NB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # Predictions
    y_pred = gnb.predict(X_test)
    accuracy = gnb.score(X_test, y_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Confusion matrix
    ax = axes[0]
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
    ax.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.1%}',
                fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Right: Feature importance (class means)
    ax = axes[1]

    # Get class means and stds
    class_means = []
    for c in range(3):
        class_means.append(X_train[y_train == c].mean(axis=0))

    class_means = np.array(class_means)

    x = np.arange(len(feature_names))
    width = 0.25

    for i, (name, color) in enumerate(zip(target_names,
                                           [COLOR_PALETTE['class1'],
                                            COLOR_PALETTE['class2'],
                                            COLOR_PALETTE['class3']])):
        offset = (i - 1) * width
        ax.bar(x + offset, class_means[i], width, label=name,
              alpha=0.8, edgecolor='black', linewidth=1.5, color=color)

    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Value (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Class-Conditional Feature Means\n(Learned by Gaussian NB)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.split(' (')[0] for name in feature_names],
                       rotation=45, ha='right')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Naïve Bayes on Iris Dataset',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/nb_iris_example.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated nb_iris_example.png")


def main():
    """Generate all Naïve Bayes figures"""
    print("="*60)
    print("Generating Naïve Bayes Figures")
    print("="*60)

    plot_bayes_theorem()
    plot_gaussian_nb()
    plot_naive_assumption()
    plot_iris_example()

    print("="*60)
    print("✅ Naïve Bayes figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
