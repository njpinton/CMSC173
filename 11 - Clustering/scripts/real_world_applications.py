#!/usr/bin/env python3
"""
Clustering - Real-World Applications Figure Generation
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates real-world application visualizations for clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import image libraries
try:
    from skimage import io, color
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

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


def customer_segmentation():
    """Simulate customer segmentation analysis"""
    np.random.seed(42)

    # Simulate customer data (spending vs frequency)
    n_customers = 400

    # Segment 1: Low spenders, low frequency
    seg1 = np.random.randn(100, 2) * [[15, 5]] + [[30, 20]]

    # Segment 2: High spenders, low frequency
    seg2 = np.random.randn(100, 2) * [[20, 5]] + [[120, 25]]

    # Segment 3: Medium spenders, high frequency
    seg3 = np.random.randn(100, 2) * [[15, 10]] + [[75, 80]]

    # Segment 4: VIP - high spenders, high frequency
    seg4 = np.random.randn(100, 2) * [[15, 8]] + [[140, 90]]

    X = np.vstack([seg1, seg2, seg3, seg4])
    X = np.clip(X, 0, None)  # No negative values

    # Fit K-Means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Scatter plot with segments
    ax = axes[0]

    segment_names = ['Budget Shoppers', 'Big Ticket Buyers',
                     'Frequent Buyers', 'VIP Customers']

    for k in range(4):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=100, alpha=0.6,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=segment_names[k])

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], s=600, marker='*',
              c='black', edgecolors='white', linewidth=3,
              zorder=10, label='Segment Centers')

    ax.set_xlabel('Average Spending ($)', fontsize=12)
    ax.set_ylabel('Purchase Frequency (per year)', fontsize=12)
    ax.set_title('Customer Segmentation Analysis',
                fontweight='bold', fontsize=14, pad=15)
    ax.legend(frameon=True, shadow=True, fancybox=True,
             framealpha=0.95, loc='best')
    ax.grid(True, alpha=0.3)

    # Right: Segment profiles
    ax = axes[1]
    ax.axis('off')

    # Create segment profile table
    profiles = []
    for k in range(4):
        mask = labels == k
        avg_spending = X[mask, 0].mean()
        avg_frequency = X[mask, 1].mean()
        count = mask.sum()
        profiles.append([segment_names[k], f'${avg_spending:.0f}',
                        f'{avg_frequency:.0f}', f'{count}'])

    # Add title
    ax.text(0.5, 0.95, 'Segment Profiles', ha='center',
           fontsize=16, fontweight='bold', transform=ax.transAxes)

    # Create table
    table_data = [['Segment', 'Avg Spending', 'Avg Frequency', 'Count']] + profiles

    table = ax.table(cellText=table_data, cellLoc='left',
                    loc='center', bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor(COLOR_PALETTE['primary'])
        cell.set_text_props(weight='bold', color='white')

    # Style data rows with cluster colors
    for i in range(1, 5):
        cell = table[(i, 0)]
        cell.set_facecolor(CLUSTER_COLORS[i-1])
        cell.set_text_props(weight='bold', color='white')

    # Add marketing strategies
    strategies_text = '''
    Marketing Strategies:

    • Budget Shoppers: Discount campaigns, loyalty programs
    • Big Ticket Buyers: Premium products, exclusive previews
    • Frequent Buyers: Subscription services, rewards
    • VIP Customers: Personalized service, early access
    '''

    ax.text(0.5, 0.05, strategies_text, ha='center', va='bottom',
           fontsize=10, transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.7',
                    facecolor=COLOR_PALETTE['info'],
                    edgecolor=COLOR_PALETTE['primary'],
                    linewidth=2, alpha=0.85))

    plt.suptitle('Real-World Application: Customer Segmentation',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/customer_segmentation.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated customer_segmentation.png")


def image_color_quantization():
    """Demonstrate image color quantization using K-Means"""
    np.random.seed(42)

    # Create synthetic colorful image
    height, width = 200, 300
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    # Create gradient image with multiple colors
    R = np.sin(2 * np.pi * X) * 0.5 + 0.5
    G = np.cos(2 * np.pi * Y) * 0.5 + 0.5
    B = np.sin(2 * np.pi * (X + Y)) * 0.5 + 0.5

    original_image = np.dstack([R, G, B])

    # Reshape for clustering
    pixels = original_image.reshape(-1, 3)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Original image
    ax = axes[0, 0]
    ax.imshow(original_image)
    ax.set_title('Original Image\n(Continuous colors)',
                fontweight='bold', fontsize=12, pad=10)
    ax.axis('off')

    # Apply K-Means with different K values
    k_values = [2, 4, 8, 16, 32]

    for idx, k in enumerate(k_values):
        if idx < 5:
            row = (idx + 1) // 3
            col = (idx + 1) % 3
            ax = axes[row, col]

            # Fit K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            quantized_colors = kmeans.cluster_centers_[labels]

            # Reshape back to image
            quantized_image = quantized_colors.reshape(height, width, 3)

            ax.imshow(np.clip(quantized_image, 0, 1))
            ax.set_title(f'K={k} colors\n({k} clusters)',
                        fontweight='bold', fontsize=12, pad=10)
            ax.axis('off')

            # Add color palette
            palette_height = 20
            palette = np.zeros((palette_height, k * 30, 3))
            for i, color in enumerate(kmeans.cluster_centers_):
                palette[:, i*30:(i+1)*30] = color

            # Create small inset for palette
            axins = ax.inset_axes([0.05, 0.05, 0.9, 0.15])
            axins.imshow(np.clip(palette, 0, 1))
            axins.axis('off')

    # Hide last subplot
    axes[1, 2].axis('off')

    plt.suptitle('Real-World Application: Image Color Quantization\n(K-Means Clustering)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/image_color_quantization.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated image_color_quantization.png")


def iris_species_clustering():
    """Cluster Iris dataset and compare with true labels"""
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Use first 2 features for visualization
    X_2d = X[:, [0, 1]]  # Sepal length and width

    # Also do PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Fit K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: True labels (first 2 features)
    ax = axes[0, 0]
    for k in range(3):
        mask = y_true == k
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=100, alpha=0.7,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=target_names[k])

    ax.set_xlabel('Sepal Length (cm)', fontsize=11)
    ax.set_ylabel('Sepal Width (cm)', fontsize=11)
    ax.set_title('True Species Labels\n(Ground Truth)',
                fontweight='bold', fontsize=12, pad=10)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Top-right: K-Means clusters (first 2 features)
    ax = axes[0, 1]
    for k in range(3):
        mask = y_pred == k
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=100, alpha=0.7,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=f'Cluster {k}')

    centroids_2d = kmeans.cluster_centers_[:, [0, 1]]
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=500, marker='*',
              c='black', edgecolors='white', linewidth=3, zorder=10)

    ax.set_xlabel('Sepal Length (cm)', fontsize=11)
    ax.set_ylabel('Sepal Width (cm)', fontsize=11)
    ax.set_title('K-Means Clustering\n(Unsupervised)',
                fontweight='bold', fontsize=12, pad=10)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Bottom-left: True labels (PCA)
    ax = axes[1, 0]
    for k in range(3):
        mask = y_true == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=100, alpha=0.7,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=target_names[k])

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax.set_title('True Labels (PCA Projection)',
                fontweight='bold', fontsize=12, pad=10)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Bottom-right: K-Means (PCA)
    ax = axes[1, 1]

    # Re-cluster on PCA features
    kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_pred_pca = kmeans_pca.fit_predict(X_pca)

    for k in range(3):
        mask = y_pred_pca == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=100, alpha=0.7,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=f'Cluster {k}')

    ax.scatter(kmeans_pca.cluster_centers_[:, 0],
              kmeans_pca.cluster_centers_[:, 1],
              s=500, marker='*', c='black', edgecolors='white',
              linewidth=3, zorder=10)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=11)
    ax.set_title('K-Means (PCA Projection)',
                fontweight='bold', fontsize=12, pad=10)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Compute metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    plt.suptitle(f'Iris Dataset Clustering (ARI={ari:.3f}, NMI={nmi:.3f})',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/iris_species_clustering.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated iris_species_clustering.png")


def document_clustering_simulation():
    """Simulate document clustering using synthetic topic vectors"""
    np.random.seed(42)

    # Simulate document-term matrix (TF-IDF-like)
    # 3 topics: Sports, Politics, Technology
    n_docs_per_topic = 40

    # Topic vectors (dominant features for each topic)
    sports_words = [0.8, 0.7, 0.6, 0.1, 0.1, 0.2]  # High in sports features
    politics_words = [0.1, 0.2, 0.1, 0.8, 0.7, 0.2]  # High in politics features
    tech_words = [0.2, 0.1, 0.1, 0.2, 0.1, 0.9]  # High in tech features

    # Generate documents with noise
    sports_docs = np.random.randn(n_docs_per_topic, 6) * 0.2 + sports_words
    politics_docs = np.random.randn(n_docs_per_topic, 6) * 0.2 + politics_words
    tech_docs = np.random.randn(n_docs_per_topic, 6) * 0.2 + tech_words

    X = np.vstack([sports_docs, politics_docs, tech_docs])
    X = np.clip(X, 0, 1)  # TF-IDF values between 0 and 1

    true_labels = np.array([0]*n_docs_per_topic + [1]*n_docs_per_topic + [2]*n_docs_per_topic)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Fit K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    topic_names = ['Sports', 'Politics', 'Technology']

    # Left: True topics
    ax = axes[0]
    for k in range(3):
        mask = true_labels == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=100, alpha=0.7,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=topic_names[k])

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                 fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                 fontsize=11)
    ax.set_title('True Document Topics\n(Labeled Data)',
                fontweight='bold', fontsize=13, pad=15)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Right: Discovered clusters
    ax = axes[1]
    for k in range(3):
        mask = pred_labels == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=100, alpha=0.7,
                  c=CLUSTER_COLORS[k], edgecolors='white',
                  linewidth=1.5, label=f'Cluster {k}')

    # Plot centroids in PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
              s=600, marker='*', c='black', edgecolors='white',
              linewidth=3, zorder=10, label='Centroids')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                 fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                 fontsize=11)
    ax.set_title('K-Means Document Clustering\n(Unsupervised Discovery)',
                fontweight='bold', fontsize=13, pad=15)
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Compute metrics
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    plt.suptitle(f'Real-World Application: Document Topic Clustering\n(ARI={ari:.3f}, NMI={nmi:.3f})',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/document_clustering.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated document_clustering.png")


def main():
    """Generate all real-world application figures"""
    print("\n" + "="*60)
    print("Generating Real-World Applications Figures for Clustering")
    print("="*60 + "\n")

    customer_segmentation()
    image_color_quantization()
    iris_species_clustering()
    document_clustering_simulation()

    print("\n" + "="*60)
    print("✅ Real-world applications figures generated successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
