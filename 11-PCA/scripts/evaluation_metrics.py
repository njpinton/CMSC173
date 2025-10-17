#!/usr/bin/env python3
"""
Principal Component Analysis - Evaluation Metrics Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates evaluation and performance visualizations for PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, fetch_openml
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
    'train': '#1976D2',
    'val': '#E53935',
    'test': '#43A047',
}

def create_output_dir():
    """Create output directory for figures"""
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_scree_plot():
    """Generate scree plot showing explained variance"""
    # Load digits dataset
    digits = load_digits()
    X = digits.data

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA with all components
    pca = PCA()
    pca.fit(X_scaled)

    # Get explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Scree plot
    ax = axes[0]
    n_components = len(explained_var)
    x = np.arange(1, n_components + 1)

    ax.plot(x, explained_var * 100, marker='o', markersize=8,
           color=COLOR_PALETTE['primary'], linewidth=3,
           markeredgecolor='white', markeredgewidth=1.5,
           label='Individual variance')

    # Highlight elbow point (where slope changes)
    elbow_idx = 10  # Typical elbow for digits dataset
    ax.axvline(elbow_idx, color=COLOR_PALETTE['danger'], linewidth=2.5,
              linestyle='--', label=f'Elbow at PC{elbow_idx}')

    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Scree Plot: Individual Explained Variance\nDigits Dataset (64 features)',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Annotate elbow
    ax.annotate('Elbow point:\nDiminishing returns\nafter this',
               xy=(elbow_idx, explained_var[elbow_idx-1]*100),
               xytext=(elbow_idx+10, explained_var[elbow_idx-1]*100+2),
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor=COLOR_PALETTE['warning'],
                        edgecolor='black', linewidth=2, alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2,
                             color=COLOR_PALETTE['danger']))

    # Right: Cumulative variance
    ax = axes[1]
    ax.plot(x, cumulative_var * 100, marker='s', markersize=8,
           color=COLOR_PALETTE['success'], linewidth=3,
           markeredgecolor='white', markeredgewidth=1.5,
           label='Cumulative variance')

    # Add threshold lines
    thresholds = [80, 90, 95]
    colors = [COLOR_PALETTE['info'], COLOR_PALETTE['warning'], COLOR_PALETTE['danger']]
    for threshold, color in zip(thresholds, colors):
        ax.axhline(threshold, color=color, linewidth=2, linestyle=':',
                  alpha=0.7, label=f'{threshold}% threshold')

        # Find number of components needed
        n_comp_needed = np.argmax(cumulative_var >= threshold/100) + 1
        ax.axvline(n_comp_needed, color=color, linewidth=2,
                  linestyle=':', alpha=0.7)
        ax.plot(n_comp_needed, threshold, 'o', markersize=10,
               color=color, markeredgecolor='black', markeredgewidth=2)

    ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Explained Variance\nHow many components do we need?',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.suptitle('Scree Plot: Choosing the Number of Components',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/scree_plot.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated scree_plot.png")


def plot_reconstruction_error():
    """Visualize reconstruction error vs number of components"""
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate reconstruction error for different numbers of components
    n_components_range = range(1, 65, 2)
    reconstruction_errors = []
    variance_retained = []

    for n_comp in n_components_range:
        pca = PCA(n_components=n_comp)
        X_transformed = pca.fit_transform(X_scaled)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # MSE reconstruction error
        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        reconstruction_errors.append(mse)
        variance_retained.append(np.sum(pca.explained_variance_ratio_) * 100)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Top: Reconstruction error
    ax = axes[0]
    ax.plot(n_components_range, reconstruction_errors, marker='o',
           markersize=8, color=COLOR_PALETTE['danger'], linewidth=3,
           markeredgecolor='white', markeredgewidth=1.5)

    ax.fill_between(n_components_range, reconstruction_errors, alpha=0.2,
                    color=COLOR_PALETTE['danger'])

    ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title('Reconstruction Error vs Number of Components\nLower is better',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate key points
    ax.annotate(f'10 components:\nMSE = {reconstruction_errors[5]:.4f}',
               xy=(n_components_range[5], reconstruction_errors[5]),
               xytext=(n_components_range[5]+10, reconstruction_errors[5]+0.05),
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor=COLOR_PALETTE['warning'],
                        edgecolor='black', linewidth=2, alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2,
                             color=COLOR_PALETTE['danger']))

    # Bottom: Variance retained vs reconstruction error trade-off
    ax = axes[1]

    # Create dual axis
    ax2 = ax.twinx()

    line1 = ax.plot(n_components_range, reconstruction_errors, marker='o',
                   markersize=7, color=COLOR_PALETTE['danger'], linewidth=3,
                   markeredgecolor='white', markeredgewidth=1.5,
                   label='Reconstruction Error (↓)')

    line2 = ax2.plot(n_components_range, variance_retained, marker='s',
                    markersize=7, color=COLOR_PALETTE['success'], linewidth=3,
                    markeredgecolor='white', markeredgewidth=1.5,
                    label='Variance Retained (↑)')

    ax.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12,
                 fontweight='bold', color=COLOR_PALETTE['danger'])
    ax2.set_ylabel('Variance Retained (%)', fontsize=12,
                  fontweight='bold', color=COLOR_PALETTE['success'])

    ax.tick_params(axis='y', labelcolor=COLOR_PALETTE['danger'])
    ax2.tick_params(axis='y', labelcolor=COLOR_PALETTE['success'])

    ax.set_title('Trade-off: Reconstruction Error vs Variance Retained',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, frameon=True, shadow=True, fancybox=True,
             framealpha=0.95, loc='center right')

    plt.suptitle('PCA Reconstruction Quality Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/reconstruction_error.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated reconstruction_error.png")


def plot_component_correlation():
    """Visualize correlation between original features and PCs"""
    np.random.seed(42)

    # Generate correlated features
    n_samples = 500
    n_features = 10

    # Create groups of correlated features
    X = np.zeros((n_samples, n_features))
    X[:, 0:3] = np.random.randn(n_samples, 1) + np.random.randn(n_samples, 3) * 0.3  # Group 1
    X[:, 3:6] = np.random.randn(n_samples, 1) + np.random.randn(n_samples, 3) * 0.3  # Group 2
    X[:, 6:] = np.random.randn(n_samples, 1) + np.random.randn(n_samples, 4) * 0.3   # Group 3

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=5)
    pca.fit(X_scaled)

    # Compute correlations between original features and PCs
    X_pca = pca.transform(X_scaled)
    correlations = np.zeros((n_features, 5))
    for i in range(n_features):
        for j in range(5):
            correlations[i, j] = np.corrcoef(X_scaled[:, i], X_pca[:, j])[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Heatmap of correlations
    ax = axes[0]
    im = ax.imshow(correlations.T, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')

    ax.set_xlabel('Original Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_title('Feature-Component Correlation Matrix\nLoadings Visualization',
                fontsize=13, fontweight='bold')

    ax.set_xticks(range(n_features))
    ax.set_xticklabels([f'F{i+1}' for i in range(n_features)])
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'PC{i+1}' for i in range(5)])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')

    # Right: Bar plot of loadings for PC1
    ax = axes[1]
    pc1_loadings = pca.components_[0]

    colors = [COLOR_PALETTE['danger'] if x > 0 else COLOR_PALETTE['primary']
              for x in pc1_loadings]
    bars = ax.barh(range(n_features), pc1_loadings, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Loading Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Original Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'PC1 Loadings (explains {pca.explained_variance_ratio_[0]*100:.1f}% variance)\nInterpretation: Which features contribute most?',
                fontsize=13, fontweight='bold')
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([f'Feature {i+1}' for i in range(n_features)])
    ax.axvline(0, color='black', linewidth=2)
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Understanding Principal Component Loadings',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/component_correlation.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated component_correlation.png")


def plot_digits_visualization():
    """Visualize digits dataset in PCA space"""
    # Load digits
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Scatter plot
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10',
                        s=60, alpha=0.7, edgecolors='white', linewidth=1)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)',
                 fontsize=12, fontweight='bold')
    ax.set_title('Digits Dataset in PCA Space (2D)\n64D → 2D projection',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10), fraction=0.046, pad=0.04)
    cbar.set_label('Digit Class', fontsize=11, fontweight='bold')

    # Right: Show some digit examples
    ax = axes[1]
    ax.axis('off')

    # Select representative digits
    n_examples = 10
    examples_per_digit = []
    for digit in range(10):
        idx = np.where(y == digit)[0][0]  # First occurrence
        examples_per_digit.append(idx)

    # Create grid of digit images
    for i, idx in enumerate(examples_per_digit):
        ax_sub = plt.subplot(5, 4, i+1)
        ax_sub.imshow(digits.images[idx], cmap='gray')
        ax_sub.axis('off')
        ax_sub.set_title(f'Digit {y[idx]}', fontsize=9, fontweight='bold')

    # Add title for right panel
    fig.text(0.75, 0.95, 'Sample Images from Dataset\n(8×8 pixel images = 64 features)',
            ha='center', fontsize=13, fontweight='bold')

    plt.suptitle('PCA Visualization: Handwritten Digits Dataset',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/digits_visualization.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated digits_visualization.png")


def plot_computational_complexity():
    """Visualize computational complexity of PCA"""
    # Simulate timing for different data sizes
    n_samples_range = np.array([100, 500, 1000, 2500, 5000, 10000])
    n_features = 100

    # Theoretical complexities (normalized)
    # PCA via covariance: O(p^2 * n + p^3)
    # PCA via SVD: O(min(n^2*p, np^2))

    cov_complexity = n_samples_range * n_features**2 + n_features**3
    svd_complexity = np.minimum(n_samples_range**2 * n_features,
                                n_samples_range * n_features**2)

    # Normalize
    cov_complexity = cov_complexity / cov_complexity[0] * 100
    svd_complexity = svd_complexity / svd_complexity[0] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Time complexity comparison
    ax = axes[0]
    ax.plot(n_samples_range, cov_complexity, marker='o', markersize=10,
           color=COLOR_PALETTE['danger'], linewidth=3,
           markeredgecolor='white', markeredgewidth=2,
           label='Covariance method: O(p²n + p³)')
    ax.plot(n_samples_range, svd_complexity, marker='s', markersize=10,
           color=COLOR_PALETTE['success'], linewidth=3,
           markeredgecolor='white', markeredgewidth=2,
           label='SVD method: O(min(n²p, np²))')

    ax.set_xlabel('Number of Samples (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Computation Time', fontsize=12, fontweight='bold')
    ax.set_title(f'Computational Complexity Comparison\n(p = {n_features} features)',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Right: Space complexity
    ax = axes[1]
    ax.axis('off')

    complexity_text = """
    Computational Complexity Analysis:

    Covariance Matrix Method:
    1. Compute covariance: O(p² × n)
    2. Eigendecomposition: O(p³)
    Total: O(p² × n + p³)
    Space: O(p²)

    Singular Value Decomposition:
    1. Compute SVD: O(min(n² × p, n × p²))
    Total: O(min(n² × p, n × p²))
    Space: O(min(n × p, p²))

    Which to use?
    • If n >> p: Covariance method (faster)
    • If p >> n: SVD method (faster)
    • In practice: SVD is more numerically stable

    Memory Considerations:
    • Covariance: Must store p×p matrix
    • SVD: More memory-efficient for tall matrices
    • Incremental PCA: O(batch_size × p) memory

    Optimization Tips:
    ✓ Use randomized SVD for approximate results
    ✓ Consider sparse methods for sparse data
    ✓ Use incremental PCA for large datasets
    """

    ax.text(0.05, 0.95, complexity_text, transform=ax.transAxes,
           fontsize=9.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('PCA Computational Complexity',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/computational_complexity.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated computational_complexity.png")


def main():
    """Generate all evaluation metric figures"""
    print("="*60)
    print("Generating Evaluation Metric Figures for PCA")
    print("="*60)

    plot_scree_plot()
    plot_reconstruction_error()
    plot_component_correlation()
    plot_digits_visualization()
    plot_computational_complexity()

    print("="*60)
    print("✅ Evaluation metric figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
