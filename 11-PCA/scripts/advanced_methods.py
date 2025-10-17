#!/usr/bin/env python3
"""
Principal Component Analysis - Advanced Methods Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates advanced method visualizations for PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles, make_moons
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


def plot_svd_decomposition():
    """Visualize SVD decomposition for PCA"""
    np.random.seed(42)

    # Create a simple 2D dataset
    mean = [0, 0]
    cov = [[3, 2], [2, 2]]
    X = np.random.multivariate_normal(mean, cov, 200)

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Perform SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Plot 1: Original centered data
    ax = axes[0, 0]
    ax.scatter(X_centered[:, 0], X_centered[:, 1], s=80, alpha=0.6,
              c=COLOR_PALETTE['primary'], edgecolors='white', linewidth=1.5)
    ax.set_xlabel('Feature 1 (centered)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2 (centered)', fontsize=11, fontweight='bold')
    ax.set_title('Centered Data Matrix X\n(200×2)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 2: Singular values
    ax = axes[0, 1]
    bars = ax.bar([0, 1], S, color=[COLOR_PALETTE['danger'], COLOR_PALETTE['success']],
                  edgecolor='black', linewidth=2, alpha=0.8, width=0.5)
    ax.set_xlabel('Component', fontsize=11, fontweight='bold')
    ax.set_ylabel('Singular Value (σ)', fontsize=11, fontweight='bold')
    ax.set_title('Singular Values from SVD\nσ₁ > σ₂', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['σ₁', 'σ₂'])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, S)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    # Plot 3: Right singular vectors (Principal directions)
    ax = axes[1, 0]
    ax.scatter(X_centered[:, 0], X_centered[:, 1], s=60, alpha=0.4,
              c=COLOR_PALETTE['primary'], edgecolors='white', linewidth=1)

    # Plot principal directions
    origin = [0, 0]
    scale = 4
    for i in range(2):
        color = COLOR_PALETTE['danger'] if i == 0 else COLOR_PALETTE['success']
        ax.arrow(origin[0], origin[1], Vt[i, 0]*scale, Vt[i, 1]*scale,
                head_width=0.3, head_length=0.25, fc=color, ec=color,
                linewidth=3, label=f'v₁ (σ={S[i]:.2f})' if i==0 else f'v₂ (σ={S[i]:.2f})')

    ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax.set_title('Right Singular Vectors V^T\n(Principal Directions)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Plot 4: SVD formula
    ax = axes[1, 1]
    ax.axis('off')

    # Display SVD formula
    formula_text = """
    Singular Value Decomposition (SVD):

    X = U Σ V^T

    Where:
    • X: Data matrix (n×p)
    • U: Left singular vectors (n×k)
    • Σ: Diagonal matrix of singular values (k×k)
    • V^T: Right singular vectors (k×p)

    Relationship to PCA:
    • Principal components: columns of V
    • Eigenvalues: λᵢ = σᵢ² / (n-1)
    • Explained variance: σᵢ² / Σσⱼ²

    Advantages:
    ✓ Numerically stable
    ✓ Efficient for large datasets
    ✓ No need to compute covariance matrix
    """

    ax.text(0.1, 0.9, formula_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.2, edgecolor='black', linewidth=2))

    plt.suptitle('PCA via Singular Value Decomposition (SVD)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/svd_decomposition.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated svd_decomposition.png")


def plot_kernel_pca_concept():
    """Visualize Kernel PCA for nonlinear dimensionality reduction"""
    np.random.seed(42)

    # Generate nonlinear data (circles)
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05)

    # Apply linear PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Apply Kernel PCA with RBF kernel
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_kpca = kpca.fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Original data
    ax = axes[0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                        s=80, alpha=0.7, edgecolors='white', linewidth=1.5)
    ax.set_xlabel('X₁', fontsize=12, fontweight='bold')
    ax.set_ylabel('X₂', fontsize=12, fontweight='bold')
    ax.set_title('Original Data\n(Nonlinear Structure)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Linear PCA
    ax = axes[1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm',
                        s=80, alpha=0.7, edgecolors='white', linewidth=1.5)
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Linear PCA\n❌ Cannot separate classes', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.05, '⚠ Linear method fails!',
           transform=ax.transAxes, fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['danger'],
                    alpha=0.7, edgecolor='black', linewidth=2))

    # Kernel PCA
    ax = axes[2]
    scatter = ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='coolwarm',
                        s=80, alpha=0.7, edgecolors='white', linewidth=1.5)
    ax.set_xlabel('Kernel PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kernel PC2', fontsize=12, fontweight='bold')
    ax.set_title('Kernel PCA (RBF)\n✓ Classes separable', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.05, '✓ Nonlinear kernel succeeds!',
           transform=ax.transAxes, fontsize=11, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['success'],
                    alpha=0.7, edgecolor='black', linewidth=2))

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Class Label', fontsize=11, fontweight='bold')

    plt.suptitle('Kernel PCA: Nonlinear Dimensionality Reduction',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/kernel_pca_concept.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated kernel_pca_concept.png")


def plot_incremental_pca():
    """Visualize incremental PCA for large datasets"""
    from sklearn.decomposition import IncrementalPCA
    np.random.seed(42)

    # Simulate processing in batches
    n_samples = 1000
    n_features = 50
    n_components = 10
    batch_size = 100

    # Generate random data
    X = np.random.randn(n_samples, n_features)

    # Regular PCA (all at once)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    var_regular = pca.explained_variance_ratio_

    # Incremental PCA (batch by batch)
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(X)
    var_incremental = ipca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Variance comparison
    ax = axes[0]
    x = np.arange(n_components)
    width = 0.35
    bars1 = ax.bar(x - width/2, var_regular * 100, width,
                   label='Regular PCA', color=COLOR_PALETTE['primary'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, var_incremental * 100, width,
                   label='Incremental PCA', color=COLOR_PALETTE['secondary'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Regular PCA vs Incremental PCA\nNearly Identical Results',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)])
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Batch processing visualization
    ax = axes[1]
    ax.axis('off')

    # Create batch processing diagram
    info_text = f"""
    Incremental PCA Workflow:

    Dataset: {n_samples} samples × {n_features} features
    Batch size: {batch_size} samples
    Number of batches: {n_samples // batch_size}

    Process:
    1️⃣ Load batch 1 ({batch_size} samples)
    2️⃣ Partial fit PCA model
    3️⃣ Discard batch from memory
    4️⃣ Load batch 2 ({batch_size} samples)
    5️⃣ Update PCA model
    6️⃣ Repeat until all batches processed

    Advantages:
    ✓ Memory efficient (O(batch_size × p))
    ✓ Suitable for out-of-core datasets
    ✓ Can process streaming data
    ✓ Results nearly identical to regular PCA

    Use cases:
    • Datasets too large for memory
    • Online learning scenarios
    • Distributed computing
    """

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
           fontsize=10.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('Incremental PCA for Large-Scale Data',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/incremental_pca.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated incremental_pca.png")


def plot_sparse_pca():
    """Visualize Sparse PCA concept"""
    from sklearn.decomposition import SparsePCA
    np.random.seed(42)

    # Generate data
    n_samples = 300
    n_features = 20
    n_components = 3

    # Create sparse underlying structure
    true_components = np.zeros((n_components, n_features))
    true_components[0, :5] = 1  # First component uses first 5 features
    true_components[1, 5:10] = 1  # Second uses next 5
    true_components[2, 10:15] = 1  # Third uses next 5

    # Generate data from sparse components
    latent = np.random.randn(n_samples, n_components)
    X = latent @ true_components + np.random.randn(n_samples, n_features) * 0.5

    # Regular PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Sparse PCA
    spca = SparsePCA(n_components=n_components, alpha=1, random_state=42)
    spca.fit(X)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Regular PCA components
    ax = axes[0, 0]
    im = ax.imshow(pca.components_, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('Feature Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Component', fontsize=11, fontweight='bold')
    ax.set_title('Regular PCA Components\nDense (all features used)',
                fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['PC1', 'PC2', 'PC3'])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot 2: Sparse PCA components
    ax = axes[0, 1]
    im = ax.imshow(spca.components_, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('Feature Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Component', fontsize=11, fontweight='bold')
    ax.set_title('Sparse PCA Components\nSparse (few features per component)',
                fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['SPC1', 'SPC2', 'SPC3'])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot 3: Sparsity comparison
    ax = axes[1, 0]
    pca_sparsity = [np.sum(np.abs(comp) > 0.1) for comp in pca.components_]
    spca_sparsity = [np.sum(np.abs(comp) > 0.1) for comp in spca.components_]

    x = np.arange(n_components)
    width = 0.35
    bars1 = ax.bar(x - width/2, pca_sparsity, width,
                   label='Regular PCA', color=COLOR_PALETTE['primary'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, spca_sparsity, width,
                   label='Sparse PCA', color=COLOR_PALETTE['success'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Component', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Active Features', fontsize=11, fontweight='bold')
    ax.set_title('Sparsity Comparison\n(|weight| > 0.1)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Comp {i+1}' for i in range(n_components)])
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    # Plot 4: Benefits
    ax = axes[1, 1]
    ax.axis('off')

    benefits_text = """
    Sparse PCA Benefits:

    Interpretability:
    • Easier to understand which features matter
    • Clear feature selection for each component
    • Reduces cognitive load in interpretation

    Mathematical Formulation:
    min ||X - XV^T||² + α||V||₁
     V

    where α controls sparsity (higher α → sparser)

    Trade-offs:
    ✓ More interpretable components
    ✓ Automatic feature selection
    ✓ Better for high-dimensional data
    ✗ Slightly less variance explained
    ✗ More computationally expensive

    Applications:
    • Gene expression analysis
    • Finance (factor models)
    • Image processing
    • Text mining
    """

    ax.text(0.05, 0.95, benefits_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['accent'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('Sparse PCA: Interpretable Dimensionality Reduction',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/sparse_pca.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated sparse_pca.png")


def plot_pca_reconstruction():
    """Visualize data reconstruction with different numbers of components"""
    np.random.seed(42)

    # Generate correlated 2D data
    mean = [0, 0]
    cov = [[2, 1.5], [1.5, 1]]
    X_original = np.random.multivariate_normal(mean, cov, 200)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    n_components_list = [2, 1, 0]
    titles = ['Original Data\n(2 dimensions)',
              'Reconstructed from PC1 only\n(1 component)',
              'Mean only\n(0 components)']

    for ax, n_comp, title in zip(axes, n_components_list, titles):
        if n_comp > 0:
            pca = PCA(n_components=n_comp)
            X_transformed = pca.fit_transform(X_original)
            X_reconstructed = pca.inverse_transform(X_transformed)

            # Calculate reconstruction error
            mse = np.mean((X_original - X_reconstructed) ** 2)
            var_retained = np.sum(pca.explained_variance_ratio_) * 100
        else:
            # Zero components = just the mean
            X_reconstructed = np.tile(X_original.mean(axis=0), (len(X_original), 1))
            mse = np.mean((X_original - X_reconstructed) ** 2)
            var_retained = 0

        # Plot original data
        ax.scatter(X_original[:, 0], X_original[:, 1], s=80, alpha=0.3,
                  c=COLOR_PALETTE['primary'], edgecolors='white',
                  linewidth=1, label='Original', zorder=1)

        # Plot reconstructed data
        if n_comp < 2:
            ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], s=80, alpha=0.7,
                      c=COLOR_PALETTE['danger'], edgecolors='white',
                      linewidth=1.5, label='Reconstructed', zorder=2)

            # Draw reconstruction lines
            for i in range(0, len(X_original), 5):  # Sample for clarity
                ax.plot([X_original[i, 0], X_reconstructed[i, 0]],
                       [X_original[i, 1], X_reconstructed[i, 1]],
                       'k-', alpha=0.2, linewidth=1, zorder=0)

        ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\nMSE: {mse:.3f} | Variance: {var_retained:.1f}%',
                    fontsize=12, fontweight='bold')
        ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Data Reconstruction with Different Numbers of Components',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/pca_reconstruction.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated pca_reconstruction.png")


def main():
    """Generate all advanced method figures"""
    print("="*60)
    print("Generating Advanced Method Figures for PCA")
    print("="*60)

    plot_svd_decomposition()
    plot_kernel_pca_concept()
    plot_incremental_pca()
    plot_sparse_pca()
    plot_pca_reconstruction()

    print("="*60)
    print("✅ Advanced method figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
