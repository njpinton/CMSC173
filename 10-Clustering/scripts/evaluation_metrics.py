#!/usr/bin/env python3
"""
Clustering - Evaluation Metrics Figure Generation
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates evaluation metric visualizations for clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score, confusion_matrix)
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

CLUSTER_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#E53935']

def create_output_dir():
    """Create output directory for figures"""
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def internal_validation_metrics():
    """Compare internal validation metrics across different K values"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=400, centers=4, n_features=2,
                      cluster_std=0.7, random_state=42)

    K_range = range(2, 11)
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Silhouette Score (higher is better)
    ax = axes[0]
    ax.plot(K_range, silhouette_scores, marker='o', markersize=10,
            linewidth=3, color=COLOR_PALETTE['primary'],
            markeredgecolor='white', markeredgewidth=2)

    best_k_sil = list(K_range)[np.argmax(silhouette_scores)]
    ax.scatter([best_k_sil], [max(silhouette_scores)], s=400, marker='*',
              c=COLOR_PALETTE['success'], edgecolors='white',
              linewidth=3, zorder=10, label=f'Best K={best_k_sil}')

    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score\n(Higher is Better)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(K_range)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Davies-Bouldin Index (lower is better)
    ax = axes[1]
    ax.plot(K_range, davies_bouldin_scores, marker='s', markersize=10,
            linewidth=3, color=COLOR_PALETTE['secondary'],
            markeredgecolor='white', markeredgewidth=2)

    best_k_db = list(K_range)[np.argmin(davies_bouldin_scores)]
    ax.scatter([best_k_db], [min(davies_bouldin_scores)], s=400, marker='*',
              c=COLOR_PALETTE['success'], edgecolors='white',
              linewidth=3, zorder=10, label=f'Best K={best_k_db}')

    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=12)
    ax.set_title('Davies-Bouldin Index\n(Lower is Better)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(K_range)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Calinski-Harabasz Score (higher is better)
    ax = axes[2]
    ax.plot(K_range, calinski_harabasz_scores, marker='^', markersize=10,
            linewidth=3, color=COLOR_PALETTE['accent'],
            markeredgecolor='white', markeredgewidth=2)

    best_k_ch = list(K_range)[np.argmax(calinski_harabasz_scores)]
    ax.scatter([best_k_ch], [max(calinski_harabasz_scores)], s=400, marker='*',
              c=COLOR_PALETTE['success'], edgecolors='white',
              linewidth=3, zorder=10, label=f'Best K={best_k_ch}')

    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Calinski-Harabasz Score', fontsize=12)
    ax.set_title('Calinski-Harabasz Score\n(Higher is Better)',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(K_range)
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Internal Validation Metrics Comparison',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/internal_validation_metrics.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated internal_validation_metrics.png")


def external_validation_metrics():
    """Demonstrate external validation metrics"""
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                           cluster_std=0.7, random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    K_values = [2, 3, 4, 5, 6, 8]

    for idx, k in enumerate(K_values):
        ax = axes[idx // 3, idx % 3]

        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)

        # Compute external metrics
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

        # Plot clusters
        for cluster_id in range(k):
            mask = y_pred == cluster_id
            ax.scatter(X[mask, 0], X[mask, 1], s=80, alpha=0.6,
                      c=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                      edgecolors='white', linewidth=1.5)

        ax.set_title(f'K={k}\nARI={ari:.3f}, NMI={nmi:.3f}',
                    fontweight='bold', fontsize=12, pad=10)
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)

        # Highlight best K
        if k == 4:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(COLOR_PALETTE['success'])
                spine.set_linewidth(4)
            ax.annotate('True K!', xy=(0.5, 0.95), xycoords='axes fraction',
                       fontsize=11, ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor=COLOR_PALETTE['success'],
                                alpha=0.8))

    plt.suptitle('External Validation: ARI and NMI Scores\n(True K=4)',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/external_validation_metrics.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated external_validation_metrics.png")


def cluster_quality_heatmap():
    """Generate heatmap showing cluster quality metrics"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                      cluster_std=0.7, random_state=42)

    K_range = range(2, 11)
    metrics_data = {
        'Silhouette': [],
        'Davies-Bouldin': [],
        'Calinski-Harabasz': [],
    }

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        metrics_data['Silhouette'].append(silhouette_score(X, labels))
        metrics_data['Davies-Bouldin'].append(davies_bouldin_score(X, labels))
        metrics_data['Calinski-Harabasz'].append(calinski_harabasz_score(X, labels))

    # Normalize metrics to [0, 1] for comparison
    normalized_data = []
    for metric_name, values in metrics_data.items():
        values_array = np.array(values)
        if metric_name == 'Davies-Bouldin':
            # Lower is better, so invert
            normalized = 1 - (values_array - values_array.min()) / (values_array.max() - values_array.min())
        else:
            # Higher is better
            normalized = (values_array - values_array.min()) / (values_array.max() - values_array.min())
        normalized_data.append(normalized)

    normalized_data = np.array(normalized_data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Create heatmap
    im = ax.imshow(normalized_data, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=1, interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(range(len(K_range)))
    ax.set_xticklabels([f'K={k}' for k in K_range])
    ax.set_yticks(range(len(metrics_data)))
    ax.set_yticklabels(list(metrics_data.keys()))

    # Add text annotations
    for i in range(len(metrics_data)):
        for j in range(len(K_range)):
            text = ax.text(j, i, f'{normalized_data[i, j]:.2f}',
                          ha="center", va="center", color="black",
                          fontsize=10, fontweight='bold')

    # Highlight best K column
    best_k_col = 2  # K=4 (index 2 in range 2-10)
    ax.add_patch(plt.Rectangle((best_k_col - 0.5, -0.5), 1,
                               len(metrics_data), fill=False,
                               edgecolor=COLOR_PALETTE['primary'],
                               linewidth=4))

    ax.set_title('Normalized Cluster Quality Metrics Heatmap\n(Green = Better)',
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Number of Clusters', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Score (0-1)', fontsize=11)

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/cluster_quality_heatmap.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated cluster_quality_heatmap.png")


def optimal_k_comparison():
    """Compare multiple methods for selecting optimal K"""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=400, centers=4, n_features=2,
                      cluster_std=0.7, random_state=42)

    K_range = range(2, 11)

    # Compute metrics
    inertias = []
    silhouette_scores_list = []
    db_scores = []
    ch_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores_list.append(silhouette_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Elbow method
    ax = axes[0, 0]
    ax.plot(K_range, inertias, marker='o', markersize=10, linewidth=3,
            color=COLOR_PALETTE['primary'], markeredgecolor='white',
            markeredgewidth=2)
    ax.scatter([4], [inertias[2]], s=400, marker='*',
              c=COLOR_PALETTE['danger'], edgecolors='white',
              linewidth=3, zorder=10)
    ax.axvline(x=4, color=COLOR_PALETTE['danger'], linestyle='--',
              linewidth=2, alpha=0.5)
    ax.set_xlabel('K', fontsize=11)
    ax.set_ylabel('Inertia', fontsize=11)
    ax.set_title('Elbow Method\n(Look for elbow)', fontweight='bold',
                fontsize=12, pad=10)
    ax.set_xticks(K_range)
    ax.grid(True, alpha=0.3)

    # Silhouette
    ax = axes[0, 1]
    ax.plot(K_range, silhouette_scores_list, marker='s', markersize=10,
            linewidth=3, color=COLOR_PALETTE['success'],
            markeredgecolor='white', markeredgewidth=2)
    best_k_sil = list(K_range)[np.argmax(silhouette_scores_list)]
    ax.scatter([best_k_sil], [max(silhouette_scores_list)], s=400,
              marker='*', c=COLOR_PALETTE['danger'], edgecolors='white',
              linewidth=3, zorder=10)
    ax.axvline(x=best_k_sil, color=COLOR_PALETTE['danger'],
              linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('K', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Silhouette Method\n(Maximize score)',
                fontweight='bold', fontsize=12, pad=10)
    ax.set_xticks(K_range)
    ax.grid(True, alpha=0.3)

    # Davies-Bouldin
    ax = axes[1, 0]
    ax.plot(K_range, db_scores, marker='^', markersize=10, linewidth=3,
            color=COLOR_PALETTE['warning'], markeredgecolor='white',
            markeredgewidth=2)
    best_k_db = list(K_range)[np.argmin(db_scores)]
    ax.scatter([best_k_db], [min(db_scores)], s=400, marker='*',
              c=COLOR_PALETTE['danger'], edgecolors='white',
              linewidth=3, zorder=10)
    ax.axvline(x=best_k_db, color=COLOR_PALETTE['danger'],
              linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('K', fontsize=11)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=11)
    ax.set_title('Davies-Bouldin Method\n(Minimize index)',
                fontweight='bold', fontsize=12, pad=10)
    ax.set_xticks(K_range)
    ax.grid(True, alpha=0.3)

    # Calinski-Harabasz
    ax = axes[1, 1]
    ax.plot(K_range, ch_scores, marker='D', markersize=10, linewidth=3,
            color=COLOR_PALETTE['info'], markeredgecolor='white',
            markeredgewidth=2)
    best_k_ch = list(K_range)[np.argmax(ch_scores)]
    ax.scatter([best_k_ch], [max(ch_scores)], s=400, marker='*',
              c=COLOR_PALETTE['danger'], edgecolors='white',
              linewidth=3, zorder=10)
    ax.axvline(x=best_k_ch, color=COLOR_PALETTE['danger'],
              linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('K', fontsize=11)
    ax.set_ylabel('Calinski-Harabasz Score', fontsize=11)
    ax.set_title('Calinski-Harabasz Method\n(Maximize score)',
                fontweight='bold', fontsize=12, pad=10)
    ax.set_xticks(K_range)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Multiple Methods for Selecting Optimal K\n(True K=4)',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/optimal_k_comparison.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated optimal_k_comparison.png")


def clustering_confusion_matrix():
    """Show confusion matrix for clustering evaluation"""
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                           cluster_std=0.7, random_state=42)

    # Fit K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Clustered data
    ax = axes[0]
    for k in range(4):
        mask = y_pred == k
        ax.scatter(X[mask, 0], X[mask, 1], s=100, alpha=0.7,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=f'Cluster {k}')

    ax.set_title('K-Means Clustering Result',
                fontweight='bold', fontsize=13, pad=15)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)

    # Right: Confusion matrix
    ax = axes[1]
    im = ax.imshow(cm, cmap='Blues', aspect='auto', interpolation='nearest')

    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, str(cm[i, j]),
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=13, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f'Cluster {i}' for i in range(4)])
    ax.set_yticklabels([f'True {i}' for i in range(4)])
    ax.set_xlabel('Predicted Cluster', fontsize=12)
    ax.set_ylabel('True Cluster', fontsize=12)
    ax.set_title('Confusion Matrix\n(External Validation)',
                fontweight='bold', fontsize=13, pad=15)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count', fontsize=11)

    # Compute and display metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    textstr = f'ARI: {ari:.3f}\nNMI: {nmi:.3f}'
    ax.text(1.25, 0.5, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.7',
                     facecolor=COLOR_PALETTE['info'],
                     edgecolor=COLOR_PALETTE['primary'],
                     linewidth=2, alpha=0.85))

    plt.suptitle('Clustering Evaluation with Confusion Matrix',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/clustering_confusion_matrix.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated clustering_confusion_matrix.png")


def main():
    """Generate all evaluation metric figures"""
    print("\n" + "="*60)
    print("Generating Evaluation Metrics Figures for Clustering")
    print("="*60 + "\n")

    internal_validation_metrics()
    external_validation_metrics()
    cluster_quality_heatmap()
    optimal_k_comparison()
    clustering_confusion_matrix()

    print("\n" + "="*60)
    print("✅ Evaluation metrics figures generated successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
