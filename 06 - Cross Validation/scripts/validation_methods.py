#!/usr/bin/env python3
"""
Cross Validation Visualization Scripts
CMSC 173 - Machine Learning

This script generates visualizations for cross validation concepts including:
- Holdout validation splits
- K-fold cross validation
- Leave-one-out cross validation
- Stratified k-fold validation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.datasets import make_classification, make_regression
import seaborn as sns

# Set style for consistent plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

def create_holdout_visualization():
    """Create visualization of holdout validation split."""
    np.random.seed(42)

    # Create sample dataset
    n_samples = 100
    indices = np.arange(n_samples)

    # Split data
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Create visualization
    y_pos = 0
    bar_height = 0.8

    # Plot training set
    for i, idx in enumerate(sorted(train_idx)):
        ax.barh(y_pos, 1, left=idx, height=bar_height, color='#2E8B57', alpha=0.8)

    # Plot validation set
    for i, idx in enumerate(sorted(val_idx)):
        ax.barh(y_pos, 1, left=idx, height=bar_height, color='#FF6347', alpha=0.8)

    # Plot test set
    for i, idx in enumerate(sorted(test_idx)):
        ax.barh(y_pos, 1, left=idx, height=bar_height, color='#4169E1', alpha=0.8)

    # Formatting
    ax.set_xlim(-2, n_samples + 2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
    ax.set_title('Holdout Validation Split (70% Train, 14% Val, 16% Test)',
                 fontsize=16, fontweight='bold', pad=20)

    # Remove y-axis
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add legend
    train_patch = mpatches.Patch(color='#2E8B57', alpha=0.8, label=f'Training Set ({len(train_idx)} samples)')
    val_patch = mpatches.Patch(color='#FF6347', alpha=0.8, label=f'Validation Set ({len(val_idx)} samples)')
    test_patch = mpatches.Patch(color='#4169E1', alpha=0.8, label=f'Test Set ({len(test_idx)} samples)')
    ax.legend(handles=[train_patch, val_patch, test_patch],
             loc='upper right', bbox_to_anchor=(1, 1))

    # Add annotations
    ax.text(len(train_idx)/2, 0.2, 'Train', ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')
    ax.text(np.mean(val_idx), 0.2, 'Val', ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')
    ax.text(np.mean(test_idx), 0.2, 'Test', ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')

    plt.tight_layout()
    plt.savefig('../figures/holdout_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_kfold_visualization():
    """Create visualization of k-fold cross validation."""
    np.random.seed(42)
    n_samples = 50
    k = 5

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(n_samples)

    fig, axes = plt.subplots(k, 1, figsize=(14, 10))
    colors = ['#2E8B57', '#FF6347']  # Train: green, Val: red

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        ax = axes[fold]

        # Create bars for each sample
        for i in range(n_samples):
            if i in train_idx:
                ax.barh(0, 1, left=i, height=0.8, color=colors[0], alpha=0.8)
            else:
                ax.barh(0, 1, left=i, height=0.8, color=colors[1], alpha=0.8)

        # Formatting
        ax.set_xlim(-1, n_samples + 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel(f'Fold {fold + 1}', fontweight='bold', rotation=0,
                     ha='right', va='center')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if fold == k - 1:
            ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
        else:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

    # Add title and legend
    fig.suptitle('5-Fold Cross Validation', fontsize=16, fontweight='bold', y=0.95)

    train_patch = mpatches.Patch(color=colors[0], alpha=0.8, label='Training Set')
    val_patch = mpatches.Patch(color=colors[1], alpha=0.8, label='Validation Set')
    fig.legend(handles=[train_patch, val_patch],
              loc='upper right', bbox_to_anchor=(0.98, 0.88))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('../figures/kfold_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_stratified_kfold_comparison():
    """Create comparison between regular and stratified k-fold."""
    np.random.seed(42)

    # Create imbalanced dataset
    X, y = make_classification(n_samples=60, n_classes=3, n_informative=3,
                              n_redundant=0, n_clusters_per_class=1,
                              weights=[0.6, 0.3, 0.1], random_state=42)

    k = 5

    # Regular K-Fold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Colors for each class
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    class_names = ['Class 0', 'Class 1', 'Class 2']

    # Regular K-Fold
    fold_distributions = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        val_classes = y[val_idx]
        class_counts = [np.sum(val_classes == i) for i in range(3)]
        fold_distributions.append(class_counts)

    fold_distributions = np.array(fold_distributions)

    # Plot regular k-fold
    bottom = np.zeros(k)
    for class_idx in range(3):
        ax1.bar(range(k), fold_distributions[:, class_idx], bottom=bottom,
                color=colors[class_idx], alpha=0.8, label=class_names[class_idx])
        bottom += fold_distributions[:, class_idx]

    ax1.set_title('Regular K-Fold Cross Validation', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontweight='bold')
    ax1.set_xticks(range(k))
    ax1.set_xticklabels([f'Fold {i+1}' for i in range(k)])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Stratified K-Fold
    fold_distributions_strat = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        val_classes = y[val_idx]
        class_counts = [np.sum(val_classes == i) for i in range(3)]
        fold_distributions_strat.append(class_counts)

    fold_distributions_strat = np.array(fold_distributions_strat)

    # Plot stratified k-fold
    bottom = np.zeros(k)
    for class_idx in range(3):
        ax2.bar(range(k), fold_distributions_strat[:, class_idx], bottom=bottom,
                color=colors[class_idx], alpha=0.8, label=class_names[class_idx])
        bottom += fold_distributions_strat[:, class_idx]

    ax2.set_title('Stratified K-Fold Cross Validation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fold', fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.set_xticks(range(k))
    ax2.set_xticklabels([f'Fold {i+1}' for i in range(k)])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../figures/stratified_kfold_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_loocv_visualization():
    """Create visualization of Leave-One-Out Cross Validation."""
    np.random.seed(42)
    n_samples = 20  # Small number for visualization

    loo = LeaveOneOut()
    indices = np.arange(n_samples)

    # Show only first 5 folds for clarity
    fig, axes = plt.subplots(5, 1, figsize=(14, 8))
    colors = ['#2E8B57', '#FF6347']  # Train: green, Val: red

    for fold, (train_idx, val_idx) in enumerate(loo.split(indices)):
        if fold >= 5:  # Only show first 5 folds
            break

        ax = axes[fold]

        # Create bars for each sample
        for i in range(n_samples):
            if i in train_idx:
                ax.barh(0, 1, left=i, height=0.8, color=colors[0], alpha=0.8)
            else:
                ax.barh(0, 1, left=i, height=0.8, color=colors[1], alpha=0.8)

        # Formatting
        ax.set_xlim(-1, n_samples + 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel(f'Fold {fold + 1}', fontweight='bold', rotation=0,
                     ha='right', va='center')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if fold == 4:
            ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
            ax.text(n_samples//2, -0.3, f'... continuing for all {n_samples} samples',
                   ha='center', fontsize=10, style='italic')
        else:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

    # Add title and legend
    fig.suptitle(f'Leave-One-Out Cross Validation (First 5 of {n_samples} Folds)',
                fontsize=16, fontweight='bold', y=0.95)

    train_patch = mpatches.Patch(color=colors[0], alpha=0.8, label='Training Set (n-1 samples)')
    val_patch = mpatches.Patch(color=colors[1], alpha=0.8, label='Validation Set (1 sample)')
    fig.legend(handles=[train_patch, val_patch],
              loc='upper right', bbox_to_anchor=(0.98, 0.88))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('../figures/loocv_validation.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating validation method visualizations...")

    create_holdout_visualization()
    print("✓ Holdout validation visualization created")

    create_kfold_visualization()
    print("✓ K-fold validation visualization created")

    create_stratified_kfold_comparison()
    print("✓ Stratified k-fold comparison created")

    create_loocv_visualization()
    print("✓ LOOCV visualization created")

    print("All validation method visualizations completed!")