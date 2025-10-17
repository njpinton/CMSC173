#!/usr/bin/env python3
"""
Principal Component Analysis - Core Concepts Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates core conceptual visualizations for PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


def plot_high_dimensional_curse():
    """Visualize the curse of dimensionality"""
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    dimensions = [2, 10, 50]
    n_samples = 100

    for idx, (ax, d) in enumerate(zip(axes, dimensions)):
        # Generate random points
        X = np.random.randn(n_samples, d)

        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(X)

        # Plot histogram
        ax.hist(distances, bins=30, color=COLOR_PALETTE['primary'],
                alpha=0.7, edgecolor='white', linewidth=1.5)
        ax.axvline(np.mean(distances), color=COLOR_PALETTE['danger'],
                  linewidth=3, linestyle='--', label=f'Mean: {np.mean(distances):.2f}')
        ax.axvline(np.median(distances), color=COLOR_PALETTE['success'],
                  linewidth=3, linestyle='--', label=f'Median: {np.median(distances):.2f}')

        ax.set_xlabel('Pairwise Distance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{d}D Space\n(Spread: {np.std(distances):.2f})',
                    fontsize=13, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
        ax.grid(True, alpha=0.3)

    plt.suptitle('The Curse of Dimensionality\nDistances Become Less Meaningful in High Dimensions',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/curse_of_dimensionality.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated curse_of_dimensionality.png")


def plot_variance_visualization():
    """Visualize variance and principal components"""
    np.random.seed(42)

    # Generate correlated 2D data
    mean = [0, 0]
    cov = [[3, 2], [2, 2]]
    X = np.random.multivariate_normal(mean, cov, 300)

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(X)

    # Get principal components
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]

    # Project data onto PCs
    X_pca = pca.transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Original data with PCs
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=80, alpha=0.6,
              c=COLOR_PALETTE['primary'], edgecolors='white', linewidth=1.5)

    # Plot principal components
    scale = 3
    origin = np.mean(X, axis=0)
    ax.arrow(origin[0], origin[1], pc1[0]*scale*np.sqrt(pca.explained_variance_[0]),
            pc1[1]*scale*np.sqrt(pca.explained_variance_[0]),
            head_width=0.3, head_length=0.2, fc=COLOR_PALETTE['danger'],
            ec=COLOR_PALETTE['danger'], linewidth=3, label='PC1')
    ax.arrow(origin[0], origin[1], pc2[0]*scale*np.sqrt(pca.explained_variance_[1]),
            pc2[1]*scale*np.sqrt(pca.explained_variance_[1]),
            head_width=0.3, head_length=0.2, fc=COLOR_PALETTE['success'],
            ec=COLOR_PALETTE['success'], linewidth=3, label='PC2')

    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title('Original Data with Principal Components\nArrow Length ∝ √Variance',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Right plot: Transformed data
    ax = axes[1]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=80, alpha=0.6,
              c=COLOR_PALETTE['secondary'], edgecolors='white', linewidth=1.5)

    ax.axhline(0, color='gray', linewidth=2, linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=2, linestyle='--', alpha=0.5)

    # Add variance explained text
    ax.text(0.05, 0.95, f'PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['danger'],
                    alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(0.05, 0.85, f'PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% variance',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['success'],
                    alpha=0.7, edgecolor='black', linewidth=2))

    ax.set_xlabel('PC1 (Maximum Variance Direction)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2 (Second Maximum Variance)', fontsize=12, fontweight='bold')
    ax.set_title('PCA-Transformed Data\nUncorrelated Components',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle('Principal Component Analysis: Variance Maximization',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/variance_visualization.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated variance_visualization.png")


def plot_covariance_matrix():
    """Visualize covariance matrix and eigenvectors"""
    np.random.seed(42)

    # Generate correlated data
    mean = [0, 0]
    cov = [[2, 1.5], [1.5, 2]]
    X = np.random.multivariate_normal(mean, cov, 500)

    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)

    # Compute covariance matrix
    cov_matrix = np.cov(X_std.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Data scatter
    ax = axes[0]
    ax.scatter(X_std[:, 0], X_std[:, 1], s=60, alpha=0.5,
              c=COLOR_PALETTE['primary'], edgecolors='white', linewidth=1)
    ax.set_xlabel('Feature 1 (standardized)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2 (standardized)', fontsize=11, fontweight='bold')
    ax.set_title('Correlated Data', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Middle: Covariance matrix heatmap
    ax = axes[1]
    im = ax.imshow(cov_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Feat 1', 'Feat 2'])
    ax.set_yticklabels(['Feat 1', 'Feat 2'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cov_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black",
                         fontsize=14, fontweight='bold')

    ax.set_title('Covariance Matrix', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Covariance', fontsize=10, fontweight='bold')

    # Right: Eigenvectors
    ax = axes[2]
    ax.scatter(X_std[:, 0], X_std[:, 1], s=40, alpha=0.3,
              c=COLOR_PALETTE['primary'], edgecolors='white', linewidth=0.5)

    # Plot eigenvectors
    origin = [0, 0]
    scale = 2.5
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        color = COLOR_PALETTE['danger'] if i == 0 else COLOR_PALETTE['success']
        ax.arrow(origin[0], origin[1], eigenvector[0]*scale*np.sqrt(eigenvalue),
                eigenvector[1]*scale*np.sqrt(eigenvalue),
                head_width=0.15, head_length=0.12, fc=color, ec=color,
                linewidth=3, label=f'λ₁={eigenvalue:.2f}' if i==0 else f'λ₂={eigenvalue:.2f}')

    ax.set_xlabel('Feature 1 (standardized)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2 (standardized)', fontsize=11, fontweight='bold')
    ax.set_title('Eigenvectors (scaled by √λ)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle('Covariance Matrix and Eigendecomposition',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/covariance_eigenvectors.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated covariance_eigenvectors.png")


def plot_dimensionality_reduction_concept():
    """Visualize the concept of dimensionality reduction"""
    np.random.seed(42)

    # Generate 3D data that lies approximately on a 2D plane
    n_samples = 200
    t = np.linspace(0, 4*np.pi, n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = 0.5 * t + np.random.randn(n_samples) * 0.3

    X = np.column_stack([x, y, z])

    # Apply PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    fig = plt.figure(figsize=(16, 7))

    # Left: 3D original data
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(X[:, 0], X[:, 1], X[:, 2],
                         c=t, cmap='viridis', s=60, alpha=0.7,
                         edgecolors='white', linewidth=1)
    ax1.set_xlabel('X₁', fontsize=12, fontweight='bold')
    ax1.set_ylabel('X₂', fontsize=12, fontweight='bold')
    ax1.set_zlabel('X₃', fontsize=12, fontweight='bold')
    ax1.set_title('Original 3D Data\n200 points × 3 dimensions',
                 fontsize=13, fontweight='bold', pad=20)
    ax1.view_init(elev=20, azim=45)

    # Right: 2D reduced data
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(X_reduced[:, 0], X_reduced[:, 1],
                         c=t, cmap='viridis', s=80, alpha=0.7,
                         edgecolors='white', linewidth=1.5)
    ax2.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax2.set_title(f'PCA-Reduced 2D Data\n{pca.explained_variance_ratio_.sum()*100:.1f}% variance retained',
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Time Parameter', fontsize=10, fontweight='bold')

    # Add arrow annotation
    ax2.annotate('Dimension Reduction\n3D → 2D',
                xy=(0.5, 0.95), xytext=(0.5, 0.95),
                xycoords='axes fraction',
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round,pad=0.7',
                         facecolor=COLOR_PALETTE['accent'],
                         edgecolor='black', linewidth=2, alpha=0.9))

    plt.suptitle('Dimensionality Reduction with PCA',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/dimensionality_reduction_concept.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated dimensionality_reduction_concept.png")


def plot_standardization_importance():
    """Show why standardization is important for PCA"""
    np.random.seed(42)

    # Generate data with different scales
    n_samples = 150
    feature1 = np.random.randn(n_samples) * 1  # Small scale
    feature2 = np.random.randn(n_samples) * 100 + 500  # Large scale, shifted
    X = np.column_stack([feature1, feature2])

    # PCA without standardization
    pca_no_std = PCA(n_components=2)
    X_pca_no_std = pca_no_std.fit_transform(X)

    # PCA with standardization
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca_std = PCA(n_components=2)
    X_pca_std = pca_std.fit_transform(X_std)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top left: Original data (not standardized)
    ax = axes[0, 0]
    ax.scatter(X[:, 0], X[:, 1], s=80, alpha=0.6,
              c=COLOR_PALETTE['primary'], edgecolors='white', linewidth=1.5)
    ax.set_xlabel('Feature 1 (scale ≈ 1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2 (scale ≈ 100)', fontsize=11, fontweight='bold')
    ax.set_title('Original Data (Different Scales)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Top right: PCA without standardization
    ax = axes[0, 1]
    ax.scatter(X_pca_no_std[:, 0], X_pca_no_std[:, 1], s=80, alpha=0.6,
              c=COLOR_PALETTE['danger'], edgecolors='white', linewidth=1.5)
    ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax.set_title(f'PCA WITHOUT Standardization\nPC1: {pca_no_std.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca_no_std.explained_variance_ratio_[1]*100:.1f}%',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.05, '⚠ Large-scale feature dominates!',
           transform=ax.transAxes, fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['warning'],
                    alpha=0.8, edgecolor='black', linewidth=2))

    # Bottom left: Standardized data
    ax = axes[1, 0]
    ax.scatter(X_std[:, 0], X_std[:, 1], s=80, alpha=0.6,
              c=COLOR_PALETTE['secondary'], edgecolors='white', linewidth=1.5)
    ax.set_xlabel('Feature 1 (standardized)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2 (standardized)', fontsize=11, fontweight='bold')
    ax.set_title('Standardized Data (Same Scale)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Bottom right: PCA with standardization
    ax = axes[1, 1]
    ax.scatter(X_pca_std[:, 0], X_pca_std[:, 1], s=80, alpha=0.6,
              c=COLOR_PALETTE['success'], edgecolors='white', linewidth=1.5)
    ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
    ax.set_title(f'PCA WITH Standardization\nPC1: {pca_std.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca_std.explained_variance_ratio_[1]*100:.1f}%',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.05, '✓ Balanced representation!',
           transform=ax.transAxes, fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['success'],
                    alpha=0.8, edgecolor='black', linewidth=2))
    ax.set_aspect('equal')

    plt.suptitle('Importance of Standardization in PCA',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/standardization_importance.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated standardization_importance.png")


def main():
    """Generate all core concept figures"""
    print("="*60)
    print("Generating Core Concept Figures for PCA")
    print("="*60)

    plot_high_dimensional_curse()
    plot_variance_visualization()
    plot_covariance_matrix()
    plot_dimensionality_reduction_concept()
    plot_standardization_importance()

    print("="*60)
    print("✅ Core concept figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
