#!/usr/bin/env python3
"""
Principal Component Analysis - Real-World Applications
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates real-world application visualizations for PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, fetch_lfw_people
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


def plot_image_compression():
    """Demonstrate PCA for image compression"""
    # Load digits and select one image
    digits = load_digits()
    original_image = digits.images[0]  # 8x8 image
    X = original_image.reshape(1, -1)  # Flatten to (1, 64)

    # For demonstration, create multiple versions with different compressions
    # We'll use the entire dataset to learn PCA, then reconstruct one image
    X_all = digits.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Test image scaled
    X_test = scaler.transform(X)

    # Different compression levels
    n_components_list = [64, 32, 16, 8, 4, 2]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (ax, n_comp) in enumerate(zip(axes, n_components_list)):
        if n_comp < 64:
            pca = PCA(n_components=n_comp)
            pca.fit(X_scaled)
            X_compressed = pca.transform(X_test)
            X_reconstructed = pca.inverse_transform(X_compressed)
            X_reconstructed = scaler.inverse_transform(X_reconstructed)
            reconstructed_image = X_reconstructed.reshape(8, 8)

            # Calculate compression ratio and error
            compression_ratio = (64 - n_comp) / 64 * 100
            mse = np.mean((original_image - reconstructed_image) ** 2)
            var_retained = np.sum(pca.explained_variance_ratio_) * 100
        else:
            reconstructed_image = original_image
            compression_ratio = 0
            mse = 0
            var_retained = 100

        # Plot
        im = ax.imshow(reconstructed_image, cmap='gray', vmin=0, vmax=16)
        ax.axis('off')

        # Color-code title based on quality
        if compression_ratio == 0:
            color = COLOR_PALETTE['primary']
            quality = 'Original'
        elif var_retained >= 95:
            color = COLOR_PALETTE['success']
            quality = 'Excellent'
        elif var_retained >= 85:
            color = COLOR_PALETTE['warning']
            quality = 'Good'
        else:
            color = COLOR_PALETTE['danger']
            quality = 'Poor'

        title = f'{n_comp} components ({quality})\n'
        title += f'Compression: {compression_ratio:.0f}% | MSE: {mse:.2f}\n'
        title += f'Variance: {var_retained:.1f}%'

        ax.set_title(title, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color,
                            alpha=0.3, edgecolor='black', linewidth=2))

    plt.suptitle('PCA for Image Compression: Digits Dataset\nTrade-off between compression and quality',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/image_compression.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated image_compression.png")


def plot_face_recognition_eigenfaces():
    """Demonstrate eigenfaces for face recognition"""
    try:
        # Try to load faces dataset
        faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        X = faces.data
        images = faces.images
        use_faces = True
    except:
        # Fallback to digits if faces dataset unavailable
        print("  (Using digits dataset as fallback)")
        digits = load_digits()
        X = digits.data
        images = digits.images
        use_faces = False

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    n_components = 12
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Get eigenfaces/eigenvectors
    eigenfaces = pca.components_

    fig = plt.figure(figsize=(16, 10))

    # Plot eigenfaces/components
    for i in range(min(n_components, 12)):
        ax = plt.subplot(3, 4, i + 1)

        if use_faces:
            eigenface = eigenfaces[i].reshape(faces.images[0].shape)
        else:
            eigenface = eigenfaces[i].reshape(8, 8)

        ax.imshow(eigenface, cmap='RdBu_r')
        ax.axis('off')

        var_explained = pca.explained_variance_ratio_[i] * 100
        title = f'PC{i+1}\n{var_explained:.1f}% var'

        ax.set_title(title, fontsize=10, fontweight='bold')

    dataset_name = 'Eigenfaces (LFW Dataset)' if use_faces else 'Eigendigits (Digits Dataset)'
    plt.suptitle(f'{dataset_name}\nPrincipal Components as Basis Images',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/eigenfaces_application.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated eigenfaces_application.png")


def plot_noise_filtering():
    """Demonstrate PCA for noise filtering"""
    np.random.seed(42)

    # Generate clean signal
    t = np.linspace(0, 2*np.pi, 100)
    clean_signal = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t),
        np.cos(2*t),
    ])

    # Add noise
    noise_level = 0.5
    noisy_signal = clean_signal + np.random.randn(*clean_signal.shape) * noise_level

    # Apply PCA for denoising
    pca = PCA(n_components=2)  # Keep only 2 main components
    signal_pca = pca.fit_transform(noisy_signal)
    denoised_signal = pca.inverse_transform(signal_pca)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Clean signal
    ax = axes[0, 0]
    for i in range(4):
        ax.plot(t, clean_signal[:, i], linewidth=2.5, alpha=0.8,
               label=f'Feature {i+1}')
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title('Original Clean Signal\n4 correlated features',
                fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Plot 2: Noisy signal
    ax = axes[0, 1]
    for i in range(4):
        ax.plot(t, noisy_signal[:, i], linewidth=2, alpha=0.6,
               label=f'Feature {i+1}')
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title(f'Noisy Signal (SNR ≈ {1/noise_level:.1f})\nCorrupted with Gaussian noise',
                fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Plot 3: Denoised signal
    ax = axes[1, 0]
    for i in range(4):
        ax.plot(t, denoised_signal[:, i], linewidth=2.5, alpha=0.8,
               label=f'Feature {i+1}')
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title(f'PCA-Denoised Signal (2 components)\nVariance retained: {pca.explained_variance_ratio_.sum()*100:.1f}%',
                fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Plot 4: Comparison for one feature
    ax = axes[1, 1]
    feature_idx = 0
    ax.plot(t, clean_signal[:, feature_idx], linewidth=3,
           color=COLOR_PALETTE['primary'], label='Original',
           alpha=0.8)
    ax.plot(t, noisy_signal[:, feature_idx], linewidth=2,
           color=COLOR_PALETTE['danger'], alpha=0.5, label='Noisy')
    ax.plot(t, denoised_signal[:, feature_idx], linewidth=2.5,
           color=COLOR_PALETTE['success'], linestyle='--',
           label='PCA Denoised')

    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title(f'Comparison: Feature {feature_idx+1}\nPCA removes random noise while preserving structure',
                fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Calculate MSE
    mse_noisy = np.mean((clean_signal - noisy_signal) ** 2)
    mse_denoised = np.mean((clean_signal - denoised_signal) ** 2)

    ax.text(0.5, 0.05, f'MSE Noisy: {mse_noisy:.3f} → Denoised: {mse_denoised:.3f}\nImprovement: {(1-mse_denoised/mse_noisy)*100:.1f}%',
           transform=ax.transAxes, fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['success'],
                    alpha=0.7, edgecolor='black', linewidth=2))

    plt.suptitle('PCA for Noise Filtering in Multivariate Signals',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/noise_filtering.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated noise_filtering.png")


def plot_data_visualization_application():
    """Show PCA for data visualization of high-dimensional data"""
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA with different numbers of components
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)

    pca_3d = PCA(n_components=3)
    X_3d = pca_3d.fit_transform(X_scaled)

    fig = plt.figure(figsize=(16, 7))

    # Left: 2D visualization
    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10',
                         s=50, alpha=0.6, edgecolors='white', linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax1.set_title(f'2D Visualization\nTotal variance: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, ticks=range(10), fraction=0.046, pad=0.04)
    cbar.set_label('Digit Class', fontsize=11, fontweight='bold')

    # Right: 3D visualization
    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                         c=y, cmap='tab10', s=40, alpha=0.5,
                         edgecolors='white', linewidth=0.5)
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)',
                  fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)',
                  fontsize=11, fontweight='bold')
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)',
                  fontsize=11, fontweight='bold')
    ax2.set_title(f'3D Visualization\nTotal variance: {pca_3d.explained_variance_ratio_.sum()*100:.1f}%',
                 fontsize=13, fontweight='bold', pad=20)
    ax2.view_init(elev=20, azim=45)

    plt.suptitle('PCA for High-Dimensional Data Visualization\nDigits: 64D → 2D/3D',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/data_visualization_application.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated data_visualization_application.png")


def plot_feature_engineering():
    """Demonstrate PCA for feature engineering in ML pipelines"""
    np.random.seed(42)

    # Simulate a classification task
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X, y = make_classification(n_samples=500, n_features=50,
                              n_informative=10, n_redundant=40,
                              random_state=42)

    # Test different numbers of PCA components
    n_components_range = range(5, 51, 5)
    scores_with_pca = []
    scores_without_pca = []

    # Without PCA
    clf = LogisticRegression(max_iter=1000, random_state=42)
    score_no_pca = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

    # With different numbers of PCA components
    for n_comp in n_components_range:
        pipeline = Pipeline([
            ('pca', PCA(n_components=n_comp)),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        score = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy').mean()
        scores_with_pca.append(score)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Accuracy vs number of components
    ax = axes[0]
    ax.plot(n_components_range, np.array(scores_with_pca)*100,
           marker='o', markersize=10, color=COLOR_PALETTE['primary'],
           linewidth=3, markeredgecolor='white', markeredgewidth=2,
           label='With PCA')
    ax.axhline(score_no_pca*100, color=COLOR_PALETTE['danger'],
              linewidth=3, linestyle='--', label=f'Without PCA ({score_no_pca*100:.1f}%)')

    # Find optimal
    optimal_idx = np.argmax(scores_with_pca)
    optimal_n = n_components_range[optimal_idx]
    optimal_score = scores_with_pca[optimal_idx]

    ax.plot(optimal_n, optimal_score*100, 'o', markersize=15,
           color=COLOR_PALETTE['success'], markeredgecolor='black',
           markeredgewidth=2, zorder=10)

    ax.annotate(f'Optimal: {optimal_n} components\nAccuracy: {optimal_score*100:.1f}%',
               xy=(optimal_n, optimal_score*100),
               xytext=(optimal_n+8, optimal_score*100-2),
               fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor=COLOR_PALETTE['success'],
                        edgecolor='black', linewidth=2, alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2,
                             color=COLOR_PALETTE['success']))

    ax.set_xlabel('Number of PCA Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('PCA as Feature Engineering\nClassification Accuracy vs Dimensionality',
                fontsize=13, fontweight='bold')
    ax.legend(frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    # Right: Benefits summary
    ax = axes[1]
    ax.axis('off')

    benefits_text = f"""
    PCA in ML Pipelines:

    Original Dataset:
    • 500 samples × 50 features
    • 10 informative, 40 redundant

    Results:
    • No PCA: {score_no_pca*100:.2f}% accuracy
    • Optimal PCA ({optimal_n} comp): {optimal_score*100:.2f}%
    • Improvement: {(optimal_score-score_no_pca)*100:.2f}%

    Benefits:
    ✓ Removes redundant features
    ✓ Reduces overfitting
    ✓ Faster training (fewer features)
    ✓ Better generalization
    ✓ Noise reduction

    When to use PCA in pipelines:
    1. High-dimensional data (p > 100)
    2. Many correlated features
    3. Noisy measurements
    4. Computational constraints
    5. Interpretability needs

    Best Practices:
    • Always fit PCA on training set only
    • Standardize before PCA
    • Use cross-validation to select k
    • Consider domain knowledge
    """

    ax.text(0.05, 0.95, benefits_text, transform=ax.transAxes,
           fontsize=9.5, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=COLOR_PALETTE['info'],
                    alpha=0.15, edgecolor='black', linewidth=2))

    plt.suptitle('PCA for Feature Engineering in Machine Learning',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/feature_engineering_application.png", dpi=200,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated feature_engineering_application.png")


def main():
    """Generate all real-world application figures"""
    print("="*60)
    print("Generating Real-World Application Figures for PCA")
    print("="*60)

    plot_image_compression()
    plot_face_recognition_eigenfaces()
    plot_noise_filtering()
    plot_data_visualization_application()
    plot_feature_engineering()

    print("="*60)
    print("✅ Real-world application figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
