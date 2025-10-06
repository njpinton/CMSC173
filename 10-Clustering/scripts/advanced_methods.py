#!/usr/bin/env python3
"""
Clustering - Advanced Methods Figure Generation
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates advanced clustering method visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Ellipse
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

# PROFESSIONAL COLOR PALETTE
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#D32F2F',
    'warning': '#F57C00',
    'info': '#0288D1',
}

CLUSTER_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#E53935',
                  '#8E44AD', '#16A085', '#F39C12', '#C0392B', '#27AE60']

def create_output_dir():
    """Create output directory for figures"""
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def kmeans_initialization_comparison():
    """Compare random vs K-Means++ initialization"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=3, n_features=2,
                      cluster_std=0.8, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Random initialization (with bad seed)
    ax = axes[0]
    kmeans_random = KMeans(n_clusters=3, init='random', n_init=1, random_state=10)
    labels_random = kmeans_random.fit_predict(X)
    centroids_random = kmeans_random.cluster_centers_

    for k in range(3):
        mask = labels_random == k
        ax.scatter(X[mask, 0], X[mask, 1], s=80, alpha=0.6,
                  c=CLUSTER_COLORS[k], edgecolors='white', linewidth=1.5,
                  label=f'Cluster {k+1}')

    ax.scatter(centroids_random[:, 0], centroids_random[:, 1],
              s=500, marker='*', c='black', edgecolors='white',
              linewidth=3, zorder=10, label='Centroids')

    ax.set_title('Random Initialization\n(Suboptimal Result)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)

    inertia_random = kmeans_random.inertia_
    ax.annotate(f'Inertia: {inertia_random:.1f}',
                xy=(0.5, -0.15), xycoords='axes fraction',
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=COLOR_PALETTE['danger'],
                         edgecolor='black', linewidth=2, alpha=0.8))

    # K-Means++ initialization
    ax = axes[1]
    kmeans_pp = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=42)
    labels_pp = kmeans_pp.fit_predict(X)
    centroids_pp = kmeans_pp.cluster_centers_

    for k in range(3):
        mask = labels_pp == k
        ax.scatter(X[mask, 0], X[mask, 1], s=80, alpha=0.6,
                  c=CLUSTER_COLORS[k], edgecolors='white', linewidth=1.5,
                  label=f'Cluster {k+1}')

    ax.scatter(centroids_pp[:, 0], centroids_pp[:, 1],
              s=500, marker='*', c='black', edgecolors='white',
              linewidth=3, zorder=10, label='Centroids')

    ax.set_title('K-Means++ Initialization\n(Optimal Result)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)

    inertia_pp = kmeans_pp.inertia_
    ax.annotate(f'Inertia: {inertia_pp:.1f}\n(Lower is better!)',
                xy=(0.5, -0.15), xycoords='axes fraction',
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=COLOR_PALETTE['success'],
                         edgecolor='black', linewidth=2, alpha=0.8))

    plt.suptitle('K-Means Initialization Strategies Comparison',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/kmeans_initialization_comparison.png",
               dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated kmeans_initialization_comparison.png")


def gmm_soft_clustering():
    """Visualize GMM soft clustering with probability contours"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2,
                      cluster_std=[0.8, 1.2, 0.6], random_state=42)

    # Fit GMM
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Hard clustering (like K-Means)
    ax = axes[0]
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=80, alpha=0.6,
                  c=CLUSTER_COLORS[k], edgecolors='white', linewidth=1.5,
                  label=f'Cluster {k+1}')

    ax.set_title('Hard Clustering\n(Each point assigned to 1 cluster)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)

    # Right: Soft clustering (GMM probabilities)
    ax = axes[1]

    # Plot probability contours for each Gaussian
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Get cluster with max probability for each point
    max_prob_idx = np.argmax(probs, axis=1)

    # Color points by uncertainty (entropy)
    from scipy.stats import entropy
    uncertainties = entropy(probs.T)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=uncertainties, s=80,
                        cmap='YlOrRd', alpha=0.7, edgecolors='white',
                        linewidth=1.5)

    # Add Gaussian contours
    for k in range(3):
        mean = gmm.means_[k]
        covar = gmm.covariances_[k]

        # Eigenvalues and eigenvectors
        v, w = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))

        # Draw ellipses at 1, 2, 3 std
        for n_std in [1, 2, 3]:
            width, height = 2 * n_std * np.sqrt(v)
            ellipse = Ellipse(mean, width, height, angle=angle,
                            facecolor='none', edgecolor=CLUSTER_COLORS[k],
                            linewidth=2.5, linestyle='--', alpha=0.7)
            ax.add_patch(ellipse)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Uncertainty\n(Higher = more uncertain)', fontsize=10)

    ax.set_title('Soft Clustering (GMM)\n(Probabilistic assignments)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)

    plt.suptitle('Gaussian Mixture Model: Soft vs Hard Clustering',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/gmm_soft_clustering.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated gmm_soft_clustering.png")


def hierarchical_dendrogram():
    """Generate hierarchical clustering dendrogram"""
    np.random.seed(42)
    X, y = make_blobs(n_samples=50, centers=3, n_features=2,
                      cluster_std=0.6, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Data points
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=120, alpha=0.7, c='gray',
              edgecolors='white', linewidth=2)

    # Add labels to points
    for i in range(min(10, len(X))):  # Label first 10 points
        ax.text(X[i, 0], X[i, 1], str(i), fontsize=8,
               ha='center', va='center', fontweight='bold')

    ax.set_title('Data Points (50 samples)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)

    # Right: Dendrogram
    ax = axes[1]

    # Compute linkage matrix
    linkage_matrix = linkage(X, method='ward')

    # Plot dendrogram
    dendrogram(linkage_matrix, ax=ax, color_threshold=15,
              above_threshold_color='gray')

    ax.set_title('Dendrogram (Ward Linkage)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Distance (Ward)', fontsize=11)

    # Add horizontal line to show cut
    ax.axhline(y=15, color=COLOR_PALETTE['danger'], linestyle='--',
              linewidth=3, label='Cut threshold (3 clusters)')
    ax.legend(frameon=True, shadow=True, loc='best')

    # Add annotation
    ax.annotate('Cut here to get\n3 clusters',
                xy=(25, 15), xytext=(25, 25),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=COLOR_PALETTE['warning'],
                         edgecolor=COLOR_PALETTE['danger'],
                         linewidth=2, alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=2.5,
                              color=COLOR_PALETTE['danger']))

    plt.suptitle('Hierarchical Clustering: Dendrogram Visualization',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/hierarchical_dendrogram.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated hierarchical_dendrogram.png")


def linkage_methods_comparison():
    """Compare different hierarchical linkage methods"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=60, centers=3, n_features=2,
                      cluster_std=0.7, random_state=42)

    linkage_methods = ['single', 'complete', 'average', 'ward']
    method_names = ['Single Linkage\n(MIN)', 'Complete Linkage\n(MAX)',
                    'Average Linkage', 'Ward Linkage']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (method, name) in enumerate(zip(linkage_methods, method_names)):
        ax = axes[idx]

        # Fit hierarchical clustering
        agg = AgglomerativeClustering(n_clusters=3, linkage=method)
        labels = agg.fit_predict(X)

        # Plot clusters
        for k in range(3):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], s=100, alpha=0.7,
                      c=CLUSTER_COLORS[k], edgecolors='white',
                      linewidth=1.5, label=f'Cluster {k+1}')

        ax.set_title(name, fontweight='bold', fontsize=13, pad=10)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.legend(frameon=True, shadow=True, fancybox=True,
                 framealpha=0.95, loc='best')

        # Add description
        descriptions = [
            'Minimum distance\nbetween clusters',
            'Maximum distance\nbetween clusters',
            'Average distance\nbetween clusters',
            'Minimize within-\ncluster variance'
        ]
        ax.text(0.02, 0.98, descriptions[idx],
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor=COLOR_PALETTE['info'],
                        alpha=0.7))

    plt.suptitle('Hierarchical Clustering: Linkage Methods Comparison',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/linkage_methods_comparison.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated linkage_methods_comparison.png")


def clustering_3d_visualization():
    """Create 3D cluster visualization"""
    np.random.seed(42)

    # Create 3D clusters
    n_samples = 300
    X1 = np.random.randn(n_samples // 3, 3) * 0.5 + np.array([0, 0, 0])
    X2 = np.random.randn(n_samples // 3, 3) * 0.5 + np.array([3, 3, 0])
    X3 = np.random.randn(n_samples // 3, 3) * 0.5 + np.array([0, 3, 3])
    X = np.vstack([X1, X2, X3])

    # Fit K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot clusters
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                  s=80, alpha=0.6, c=CLUSTER_COLORS[k],
                  edgecolors='white', linewidth=1,
                  label=f'Cluster {k+1}')

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
              s=600, marker='*', c='black', edgecolors='white',
              linewidth=3, label='Centroids', zorder=10)

    ax.set_xlabel('Feature 1', fontsize=12, labelpad=10)
    ax.set_ylabel('Feature 2', fontsize=12, labelpad=10)
    ax.set_zlabel('Feature 3', fontsize=12, labelpad=10)
    ax.set_title('3D Cluster Visualization\n(K-Means with 3 Clusters)',
                fontweight='bold', fontsize=16, pad=20)

    ax.legend(frameon=True, shadow=True, fancybox=True,
             framealpha=0.95, loc='best')

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/clustering_3d.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated clustering_3d.png")


def agglomerative_steps():
    """Visualize agglomerative clustering step by step"""
    np.random.seed(42)
    X = np.array([[0, 0], [1, 0], [0, 1], [5, 5], [5, 6], [6, 5]])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    steps = [
        (6, "Initial: 6 clusters"),
        (5, "Step 1: Merge closest (0,1)"),
        (4, "Step 2: Merge (0-1, 2)"),
        (3, "Step 3: Form 3 clusters"),
        (2, "Step 4: Merge to 2 clusters"),
        (1, "Step 5: All merged (1 cluster)")
    ]

    for idx, (n_clusters, title) in enumerate(steps):
        ax = axes[idx]

        if n_clusters == 6:
            # Initial state - all separate
            colors = CLUSTER_COLORS[:6]
            for i, (x, color) in enumerate(zip(X, colors)):
                ax.scatter(x[0], x[1], s=300, c=color,
                          edgecolors='white', linewidth=2.5)
                ax.text(x[0], x[1] - 0.5, f'P{i}', ha='center',
                       fontsize=10, fontweight='bold')
        else:
            # Fit agglomerative clustering
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
            labels = agg.fit_predict(X)

            for k in range(n_clusters):
                mask = labels == k
                ax.scatter(X[mask, 0], X[mask, 1], s=300,
                          c=CLUSTER_COLORS[k], edgecolors='white',
                          linewidth=2.5, label=f'C{k+1}')

            # Draw connections within clusters
            for k in range(n_clusters):
                mask = labels == k
                points = X[mask]
                if len(points) > 1:
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            ax.plot([points[i, 0], points[j, 0]],
                                   [points[i, 1], points[j, 1]],
                                   'k--', alpha=0.3, linewidth=2)

        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Agglomerative Clustering: Bottom-Up Merging Process',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/agglomerative_steps.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated agglomerative_steps.png")


def main():
    """Generate all advanced method figures"""
    print("\n" + "="*60)
    print("Generating Advanced Methods Figures for Clustering")
    print("="*60 + "\n")

    kmeans_initialization_comparison()
    gmm_soft_clustering()
    hierarchical_dendrogram()
    linkage_methods_comparison()
    clustering_3d_visualization()
    agglomerative_steps()

    print("\n" + "="*60)
    print("✅ Advanced methods figures generated successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
