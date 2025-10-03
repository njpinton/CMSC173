"""
Model validation and evaluation visualizations.

This script generates illustrations for:
- Cross-validation schemes
- Evaluation metrics
- ROC curves and AUC
- Confusion matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve,
                              mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_cross_validation_scheme():
    """Visualize different cross-validation schemes."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    n_samples = 50
    n_folds = 5

    # K-Fold CV
    fold_size = n_samples // n_folds
    for i in range(n_folds):
        train_indices = list(range(0, i*fold_size)) + list(range((i+1)*fold_size, n_samples))
        val_indices = list(range(i*fold_size, (i+1)*fold_size))

        axes[0].broken_barh([(idx, 1) for idx in train_indices],
                            (i*2, 1.5), facecolors='skyblue', edgecolors='black', linewidth=0.5)
        axes[0].broken_barh([(idx, 1) for idx in val_indices],
                            (i*2, 1.5), facecolors='lightcoral', edgecolors='black', linewidth=0.5)

    axes[0].set_ylim(0, n_folds*2)
    axes[0].set_xlim(0, n_samples)
    axes[0].set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Fold', fontsize=11, fontweight='bold')
    axes[0].set_title('K-Fold Cross-Validation (K=5)', fontsize=13, fontweight='bold')
    axes[0].set_yticks([i*2 + 0.75 for i in range(n_folds)])
    axes[0].set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])

    # Leave-One-Out CV (showing first 10 folds)
    n_show = 10
    for i in range(n_show):
        train_indices = list(range(0, i)) + list(range(i+1, n_samples))
        axes[1].broken_barh([(idx, 1) for idx in train_indices],
                            (i*2, 1.5), facecolors='skyblue', edgecolors='black', linewidth=0.3)
        axes[1].broken_barh([(i, 1)], (i*2, 1.5),
                            facecolors='lightcoral', edgecolors='black', linewidth=0.5)

    axes[1].set_ylim(0, n_show*2)
    axes[1].set_xlim(0, n_samples)
    axes[1].set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Fold', fontsize=11, fontweight='bold')
    axes[1].set_title('Leave-One-Out CV (First 10 iterations shown)', fontsize=13, fontweight='bold')
    axes[1].set_yticks([i*2 + 0.75 for i in range(n_show)])
    axes[1].set_yticklabels([f'Iter {i+1}' for i in range(n_show)])

    # Stratified K-Fold (simulated)
    for i in range(n_folds):
        # Simulate stratified splits (alternating samples)
        train_indices = [j for j in range(n_samples) if j % n_folds != i]
        val_indices = [j for j in range(n_samples) if j % n_folds == i]

        axes[2].broken_barh([(idx, 1) for idx in train_indices],
                            (i*2, 1.5), facecolors='skyblue', edgecolors='black', linewidth=0.5)
        axes[2].broken_barh([(idx, 1) for idx in val_indices],
                            (i*2, 1.5), facecolors='lightcoral', edgecolors='black', linewidth=0.5)

    axes[2].set_ylim(0, n_folds*2)
    axes[2].set_xlim(0, n_samples)
    axes[2].set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Fold', fontsize=11, fontweight='bold')
    axes[2].set_title('Stratified K-Fold Cross-Validation (K=5)', fontsize=13, fontweight='bold')
    axes[2].set_yticks([i*2 + 0.75 for i in range(n_folds)])
    axes[2].set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', edgecolor='black', label='Training Set'),
                      Patch(facecolor='lightcoral', edgecolor='black', label='Validation Set')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig('../figures/cross_validation_schemes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: cross_validation_schemes.png")


def plot_confusion_matrix_example():
    """Generate example confusion matrix."""
    # Generate classification data
    X, y = make_classification(n_samples=300, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                ax=ax, linewidths=1, linecolor='black')

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix Example', fontsize=15, fontweight='bold')
    ax.set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
    ax.set_yticklabels(['Class 0', 'Class 1', 'Class 2'])

    plt.tight_layout()
    plt.savefig('../figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: confusion_matrix.png")


def plot_roc_curve_example():
    """Generate ROC curve example."""
    # Generate binary classification data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train classifiers
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: roc_curve.png")


def plot_precision_recall_curve():
    """Generate precision-recall curve."""
    # Generate imbalanced classification data
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, weights=[0.9, 0.1],
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, 'b-', linewidth=2.5, label='Precision-Recall Curve')
    ax.fill_between(recall, precision, alpha=0.2)

    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curve (Imbalanced Data)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('../figures/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: precision_recall_curve.png")


def plot_validation_curve_example():
    """Generate validation curve example."""
    # Generate regression data
    X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)

    param_range = np.arange(1, 21)
    train_scores, val_scores = validation_curve(
        RandomForestRegressor(random_state=42), X, y,
        param_name="max_depth", param_range=param_range,
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(param_range, train_scores_mean, 'b-o', linewidth=2.5,
            label='Training Score', markersize=6)
    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color='blue')

    ax.plot(param_range, val_scores_mean, 'r-s', linewidth=2.5,
            label='Validation Score', markersize=6)
    ax.fill_between(param_range, val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std, alpha=0.2, color='red')

    # Mark optimal
    optimal_idx = np.argmin(val_scores_mean)
    ax.axvline(param_range[optimal_idx], color='green', linestyle='--',
               linewidth=2, label=f'Optimal = {param_range[optimal_idx]}')

    ax.set_xlabel('Max Depth (Hyperparameter)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=13, fontweight='bold')
    ax.set_title('Validation Curve Example', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/validation_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: validation_curve.png")


def plot_metrics_comparison():
    """Compare different evaluation metrics."""
    metrics_data = {
        'Accuracy': [0.85, 0.78, 0.92, 0.88],
        'Precision': [0.82, 0.85, 0.89, 0.86],
        'Recall': [0.88, 0.72, 0.94, 0.90],
        'F1-Score': [0.85, 0.78, 0.91, 0.88]
    }
    models = ['Model A', 'Model B', 'Model C', 'Model D']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (metric, values) in enumerate(metrics_data.items()):
        ax.bar(x + idx*width, values, width, label=metric, color=colors[idx], alpha=0.8)

    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Classification Metrics Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig('../figures/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: metrics_comparison.png")


if __name__ == "__main__":
    print("Generating validation method visualizations...")
    plot_cross_validation_scheme()
    plot_confusion_matrix_example()
    plot_roc_curve_example()
    plot_precision_recall_curve()
    plot_validation_curve_example()
    plot_metrics_comparison()
    print("\nAll validation figures generated successfully!")
