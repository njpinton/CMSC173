#!/usr/bin/env python3
"""
Clustering - Core Concepts Figure Generation
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates core concept visualizations for clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial import Voronoi, voronoi_plot_2d
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

# Additional cluster colors
CLUSTER_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#E53935',
                  '#8E44AD', '#16A085', '#F39C12', '#C0392B', '#27AE60']

def create_output_dir():
    """Create output directory for figures"""
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def clustering_motivation():
    """Generate motivation figure showing clustering concept"""
    np.random.seed(42)

    # Create three distinct clusters
    X, y = make_blobs(n_samples=300, centers=3, n_features=2,
                      cluster_std=0.6, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Unlabeled data
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=80, alpha=0.7, c='gray',
               edgecolors='white', linewidth=1.5)
    ax.set_title('Unlabeled Data\n(What patterns exist?)',
                 fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.annotate('Goal: Find natural groupings',
                xy=(0.5, -0.15), xycoords='axes fraction',
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=COLOR_PALETTE['warning'],
                         edgecolor=COLOR_PALETTE['accent'],
                         linewidth=2, alpha=0.8))

    # Right: Discovered clusters
    ax = axes[1]
    for i in range(3):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], s=80, alpha=0.7,
                  c=CLUSTER_COLORS[i], edgecolors='white',
                  linewidth=1.5, label=f'Cluster {i+1}')

    ax.set_title('Discovered Clusters\n(Unsupervised Learning)',
                 fontweight='bold', fontsize=14, pad=15)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.annotate('Found 3 distinct groups!',
                xy=(0.5, -0.15), xycoords='axes fraction',
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=COLOR_PALETTE['success'],
                         edgecolor=COLOR_PALETTE['primary'],
                         linewidth=2, alpha=0.8))

    plt.suptitle('Clustering: Discovering Hidden Structure in Data',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/clustering_motivation.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated clustering_motivation.png")


def kmeans_iterations():
    """Visualize K-Means iterations step by step"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=150, centers=3, n_features=2,
                      cluster_std=0.7, random_state=42)

    # Manually simulate K-Means iterations
    n_iterations = 4
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Initialize random centroids
    np.random.seed(0)
    initial_idx = np.random.choice(len(X), 3, replace=False)
    centroids = X[initial_idx].copy()

    for iteration in range(n_iterations):
        ax = axes[iteration]

        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Plot points with current labels
        for k in range(3):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], s=80, alpha=0.6,
                      c=CLUSTER_COLORS[k], edgecolors='white',
                      linewidth=1.5, label=f'Cluster {k+1}')

        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], s=400, marker='*',
                  c='black', edgecolors='white', linewidth=3,
                  label='Centroids', zorder=10)

        if iteration == 0:
            title = f'Iteration {iteration}: Random Initialization'
        else:
            title = f'Iteration {iteration}: Update Centroids & Reassign'

        ax.set_title(title, fontweight='bold', fontsize=13, pad=10)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.legend(frameon=True, shadow=True, fancybox=True,
                 framealpha=0.95, loc='best')

        # Update centroids for next iteration
        if iteration < n_iterations - 1:
            for k in range(3):
                mask = labels == k
                if mask.sum() > 0:
                    centroids[k] = X[mask].mean(axis=0)

    plt.suptitle('K-Means Algorithm: Iterative Refinement Process',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/kmeans_iterations.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated kmeans_iterations.png")


def voronoi_diagram():
    """Generate Voronoi diagram showing K-Means decision boundaries"""
    np.random.seed(42)
    X, y = make_blobs(n_samples=200, centers=4, n_features=2,
                      cluster_std=0.6, random_state=42)

    # Fit K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Create dense grid for decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundaries with color regions
    ax.contourf(xx, yy, Z, alpha=0.15, levels=3,
                colors=[CLUSTER_COLORS[i] for i in range(4)])
    ax.contour(xx, yy, Z, levels=3, colors='black',
               linewidths=2, linestyles='--', alpha=0.4)

    # Plot data points
    for k in range(4):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=80, alpha=0.7,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=f'Cluster {k+1}')

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], s=500, marker='*',
              c='black', edgecolors='white', linewidth=3,
              label='Centroids', zorder=10)

    ax.set_title('Voronoi Diagram: K-Means Decision Boundaries',
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.legend(frameon=True, shadow=True, fancybox=True,
             framealpha=0.95, loc='best')

    ax.annotate('Each region contains\npoints closest to\nthat centroid',
                xy=(centroids[0, 0], centroids[0, 1]),
                xytext=(centroids[0, 0] - 2, centroids[0, 1] + 2),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=COLOR_PALETTE['info'],
                         edgecolor=COLOR_PALETTE['primary'],
                         linewidth=2, alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=2.5,
                              color=COLOR_PALETTE['primary']))

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/voronoi_diagram.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated voronoi_diagram.png")


def elbow_method():
    """Generate elbow method plot for optimal K selection"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                      cluster_std=0.7, random_state=42)

    # Compute inertia for different K values
    K_range = range(1, 11)
    inertias = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot inertia curve
    ax.plot(K_range, inertias, marker='o', markersize=12,
            linewidth=3, color=COLOR_PALETTE['primary'],
            markeredgecolor='white', markeredgewidth=2,
            label='Within-Cluster Sum of Squares (WCSS)')

    # Highlight the elbow point (K=4)
    elbow_k = 4
    elbow_idx = elbow_k - 1
    ax.scatter([elbow_k], [inertias[elbow_idx]], s=500, marker='*',
              c=COLOR_PALETTE['danger'], edgecolors='white',
              linewidth=3, zorder=10, label='Elbow Point')

    # Annotate elbow
    ax.annotate('Elbow at K=4\n(Optimal choice)',
                xy=(elbow_k, inertias[elbow_idx]),
                xytext=(elbow_k + 1.5, inertias[elbow_idx] + 50),
                fontsize=12, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=COLOR_PALETTE['warning'],
                         edgecolor=COLOR_PALETTE['danger'],
                         linewidth=2.5, alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=3,
                              color=COLOR_PALETTE['danger']))

    # Add vertical line at elbow
    ax.axvline(x=elbow_k, color=COLOR_PALETTE['danger'],
               linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Number of Clusters (K)', fontsize=13)
    ax.set_ylabel('Inertia (WCSS)', fontsize=13)
    ax.set_title('Elbow Method for Optimal K Selection',
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xticks(K_range)
    ax.legend(frameon=True, shadow=True, fancybox=True,
             framealpha=0.95, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add explanation box
    textstr = 'Look for the "elbow" where\nadding more clusters gives\ndiminishing returns'
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.7',
                     facecolor=COLOR_PALETTE['info'],
                     edgecolor=COLOR_PALETTE['primary'],
                     linewidth=2, alpha=0.85))

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/elbow_method.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated elbow_method.png")


def silhouette_analysis():
    """Generate silhouette analysis plot"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                      cluster_std=0.7, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for idx, n_clusters in enumerate([2, 3, 4, 5]):
        ax = axes[idx // 2, idx % 2]

        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # Compute silhouette scores
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate silhouette scores for samples in cluster i
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=CLUSTER_COLORS[i], edgecolor='white',
                            alpha=0.7, linewidth=1.5)

            # Label clusters
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1),
                   fontsize=11, fontweight='bold')

            y_lower = y_upper + 10

        # Average silhouette score line
        ax.axvline(x=silhouette_avg, color=COLOR_PALETTE['danger'],
                  linestyle='--', linewidth=3, label=f'Avg Score: {silhouette_avg:.3f}')

        ax.set_xlabel('Silhouette Coefficient', fontsize=11)
        ax.set_ylabel('Cluster Label', fontsize=11)
        ax.set_title(f'K={n_clusters} (Score={silhouette_avg:.3f})',
                    fontweight='bold', fontsize=13, pad=10)
        ax.set_xlim([-0.1, 1])
        ax.set_yticks([])
        ax.legend(frameon=True, shadow=True, loc='best')

        # Highlight best K
        if n_clusters == 4:
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_PALETTE['success'])
                spine.set_linewidth(4)

    plt.suptitle('Silhouette Analysis for Optimal K Selection',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/silhouette_analysis.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated silhouette_analysis.png")


def distance_metrics():
    """Visualize different distance metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Create two points
    p1 = np.array([1, 1])
    p2 = np.array([4, 5])

    metrics = ['Euclidean', 'Manhattan', 'Chebyshev']

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        # Plot points
        ax.scatter(*p1, s=300, c=COLOR_PALETTE['primary'],
                  edgecolors='white', linewidth=2.5, zorder=10, marker='o')
        ax.scatter(*p2, s=300, c=COLOR_PALETTE['accent'],
                  edgecolors='white', linewidth=2.5, zorder=10, marker='s')

        ax.text(p1[0], p1[1] - 0.4, 'Point A', ha='center',
               fontsize=11, fontweight='bold')
        ax.text(p2[0], p2[1] + 0.4, 'Point B', ha='center',
               fontsize=11, fontweight='bold')

        if metric == 'Euclidean':
            # Straight line
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-',
                   linewidth=3, label='Euclidean distance')
            dist = np.sqrt(np.sum((p2 - p1)**2))
            ax.annotate(f'd = {dist:.2f}',
                       xy=((p1[0] + p2[0])/2, (p1[1] + p2[1])/2),
                       fontsize=11, ha='center',
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='yellow', alpha=0.8))

        elif metric == 'Manhattan':
            # L-shaped path
            ax.plot([p1[0], p2[0]], [p1[1], p1[1]], 'g-',
                   linewidth=3, label='Horizontal')
            ax.plot([p2[0], p2[0]], [p1[1], p2[1]], 'b-',
                   linewidth=3, label='Vertical')
            dist = np.abs(p2[0] - p1[0]) + np.abs(p2[1] - p1[1])
            ax.annotate(f'd = {dist:.2f}',
                       xy=(p2[0] + 0.3, (p1[1] + p2[1])/2),
                       fontsize=11, ha='left',
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='yellow', alpha=0.8))

        else:  # Chebyshev
            # Max coordinate difference
            ax.plot([p1[0], p2[0]], [p1[1], p1[1]], 'g--',
                   linewidth=2, alpha=0.5)
            ax.plot([p2[0], p2[0]], [p1[1], p2[1]], 'b-',
                   linewidth=3, label='Max dimension')
            dist = np.max(np.abs(p2 - p1))
            ax.annotate(f'd = {dist:.2f}',
                       xy=(p2[0] + 0.3, (p1[1] + p2[1])/2),
                       fontsize=11, ha='left',
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='yellow', alpha=0.8))

        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 6.5)
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_title(f'{metric} Distance', fontweight='bold',
                    fontsize=13, pad=10)
        ax.legend(frameon=True, shadow=True, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Distance Metrics in Clustering',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/distance_metrics.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated distance_metrics.png")


def main():
    """Generate all core concept figures"""
    print("\n" + "="*60)
    print("Generating Core Concepts Figures for Clustering")
    print("="*60 + "\n")

    clustering_motivation()
    kmeans_iterations()
    voronoi_diagram()
    elbow_method()
    silhouette_analysis()
    distance_metrics()

    print("\n" + "="*60)
    print("✅ Core concepts figures generated successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
