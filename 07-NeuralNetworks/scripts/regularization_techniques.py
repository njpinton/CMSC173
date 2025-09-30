#!/usr/bin/env python3
"""
Regularization Techniques for Neural Networks
CMSC 173 - Machine Learning

This script generates visualizations for:
- Overfitting vs regularization
- Dropout technique
- L1/L2 regularization effects
- Training curves with regularization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import os

# Set style for consistent figures
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def create_output_dir():
    """Create output directory for figures"""
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_overfitting_demo():
    """Demonstrate overfitting and regularization effects"""
    # Generate synthetic regression data
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create range for plotting
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Different model complexities
    configs = [
        {'hidden_layer_sizes': (5,), 'alpha': 0.0, 'title': 'Simple Model (5 neurons)'},
        {'hidden_layer_sizes': (50,), 'alpha': 0.0, 'title': 'Complex Model (50 neurons) - Overfitting'},
        {'hidden_layer_sizes': (50,), 'alpha': 0.01, 'title': 'Complex Model + L2 Regularization'},
        {'hidden_layer_sizes': (50,), 'alpha': 0.1, 'title': 'Strong Regularization'}
    ]

    for idx, config in enumerate(configs):
        ax = axes[idx // 2, idx % 2]

        # Train model
        model = MLPRegressor(
            hidden_layer_sizes=config['hidden_layer_sizes'],
            alpha=config['alpha'],
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_plot_pred = model.predict(X_plot)

        # Plot data and predictions
        ax.scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data')
        ax.scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data')
        ax.plot(X_plot, y_plot_pred, color='green', linewidth=2, label='Model Prediction')

        # Calculate errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        ax.set_title(f"{config['title']}\\nTrain MSE: {train_mse:.1f}, Test MSE: {test_mse:.1f}")
        ax.set_xlabel('Feature')
        ax.set_ylabel('Target')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/overfitting_regularization_demo.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated overfitting_regularization_demo.png")

def create_dropout_visualization():
    """Visualize dropout technique"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Network without dropout
    layer_sizes = [4, 6, 4, 2]
    layer_x = [1, 3, 5, 7]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    # Draw network without dropout
    node_positions = []
    for i, (x, size, color) in enumerate(zip(layer_x, layer_sizes, colors)):
        positions = []
        y_start = (6 - size) / 2
        for j in range(size):
            y = y_start + j
            circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=1.5)
            ax1.add_patch(circle)
            positions.append((x, y))
        node_positions.append(positions)

    # Draw all connections
    for i in range(len(layer_sizes) - 1):
        for pos1 in node_positions[i]:
            for pos2 in node_positions[i + 1]:
                ax1.plot([pos1[0] + 0.15, pos2[0] - 0.15], [pos1[1], pos2[1]],
                        'gray', linewidth=1, alpha=0.7)

    ax1.set_xlim(0, 8)
    ax1.set_ylim(-0.5, 6.5)
    ax1.set_title('Training: All Neurons Active', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw network with dropout
    dropped_nodes = {1: [1, 4], 2: [0, 2]}  # Nodes to drop in each layer

    node_positions_dropout = []
    for i, (x, size, color) in enumerate(zip(layer_x, layer_sizes, colors)):
        positions = []
        y_start = (6 - size) / 2
        for j in range(size):
            y = y_start + j

            # Check if this node should be dropped
            if i in dropped_nodes and j in dropped_nodes[i]:
                # Draw dropped node
                circle = plt.Circle((x, y), 0.15, color='lightgray', ec='red',
                                  linewidth=2, linestyle='--', alpha=0.5)
                ax2.add_patch(circle)
                ax2.plot([x-0.1, x+0.1], [y-0.1, y+0.1], 'r-', linewidth=3)
                ax2.plot([x-0.1, x+0.1], [y+0.1, y-0.1], 'r-', linewidth=3)
            else:
                # Draw active node
                circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=1.5)
                ax2.add_patch(circle)

            positions.append((x, y))
        node_positions_dropout.append(positions)

    # Draw connections, but skip connections to/from dropped nodes
    for i in range(len(layer_sizes) - 1):
        for j, pos1 in enumerate(node_positions_dropout[i]):
            for k, pos2 in enumerate(node_positions_dropout[i + 1]):
                # Skip if either node is dropped
                if ((i in dropped_nodes and j in dropped_nodes[i]) or
                    (i+1 in dropped_nodes and k in dropped_nodes[i+1])):
                    continue

                ax2.plot([pos1[0] + 0.15, pos2[0] - 0.15], [pos1[1], pos2[1]],
                        'gray', linewidth=1, alpha=0.7)

    ax2.set_xlim(0, 8)
    ax2.set_ylim(-0.5, 6.5)
    ax2.set_title('Training with Dropout (50% rate)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                  markersize=10, label='Active Neuron'),
        plt.Line2D([0], [0], marker='X', color='red', markersize=10,
                  label='Dropped Neuron', markerfacecolor='lightgray')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/dropout_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated dropout_visualization.png")

def create_regularization_comparison():
    """Compare different regularization techniques"""
    # Generate classification data
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=10, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Different regularization strengths
    alphas = [0.0, 0.001, 0.01, 0.1, 1.0]
    train_scores = []
    test_scores = []

    for alpha in alphas:
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            alpha=alpha,
            max_iter=500,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

        train_scores.append(train_acc)
        test_scores.append(test_acc)

    # Plot regularization effect
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy vs regularization strength
    ax1.semilogx(alphas, train_scores, 'o-', linewidth=2, markersize=8, label='Training Accuracy')
    ax1.semilogx(alphas, test_scores, 's-', linewidth=2, markersize=8, label='Test Accuracy')
    ax1.set_xlabel('Regularization Strength (α)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Effect of L2 Regularization Strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.0001, 10)

    # Weight magnitude visualization
    # Train models with different regularization and show weight magnitudes
    regularization_types = ['No Regularization', 'L2 (α=0.01)', 'L2 (α=0.1)']
    alpha_values = [0.0, 0.01, 0.1]
    weight_norms = []

    for alpha in alpha_values:
        model = MLPClassifier(
            hidden_layer_sizes=(50,),
            alpha=alpha,
            max_iter=500,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Calculate weight magnitudes
        weights = np.concatenate([coef.flatten() for coef in model.coefs_])
        weight_norms.append(np.abs(weights))

    # Box plot of weight magnitudes
    ax2.boxplot(weight_norms, labels=regularization_types)
    ax2.set_ylabel('Weight Magnitude')
    ax2.set_title('Weight Distribution with Different Regularization')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/regularization_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated regularization_comparison.png")

def create_training_curves_regularization():
    """Show training curves with and without regularization"""
    # Generate data
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                              n_redundant=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Simulate training curves (epochs)
    epochs = range(1, 101)

    # Without regularization - shows overfitting
    np.random.seed(42)
    train_acc_no_reg = 0.5 + 0.45 * (1 - np.exp(-np.array(epochs) / 20)) + np.random.normal(0, 0.01, len(epochs))
    test_acc_no_reg = 0.5 + 0.3 * (1 - np.exp(-np.array(epochs) / 30)) - 0.1 * np.maximum(0, (np.array(epochs) - 50) / 50) + np.random.normal(0, 0.015, len(epochs))

    # With regularization - more stable
    train_acc_reg = 0.5 + 0.35 * (1 - np.exp(-np.array(epochs) / 25)) + np.random.normal(0, 0.008, len(epochs))
    test_acc_reg = 0.5 + 0.32 * (1 - np.exp(-np.array(epochs) / 30)) + np.random.normal(0, 0.01, len(epochs))

    # Smooth the curves
    from scipy.ndimage import uniform_filter1d
    train_acc_no_reg = uniform_filter1d(train_acc_no_reg, size=5)
    test_acc_no_reg = uniform_filter1d(test_acc_no_reg, size=5)
    train_acc_reg = uniform_filter1d(train_acc_reg, size=5)
    test_acc_reg = uniform_filter1d(test_acc_reg, size=5)

    # Plot without regularization
    ax1.plot(epochs, train_acc_no_reg, 'b-', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs, test_acc_no_reg, 'r-', linewidth=2, label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training Curves: No Regularization\\n(Shows Overfitting)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)

    # Highlight overfitting region
    overfitting_start = 50
    ax1.axvspan(overfitting_start, 100, alpha=0.2, color='red', label='Overfitting Region')
    ax1.text(75, 0.6, 'Overfitting\\nRegion', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))

    # Plot with regularization
    ax2.plot(epochs, train_acc_reg, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, test_acc_reg, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Curves: With L2 Regularization\\n(Prevents Overfitting)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)

    # Highlight stable region
    ax2.axhspan(0.78, 0.85, alpha=0.2, color='green')
    ax2.text(50, 0.815, 'Stable\\nGeneralization', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.3))

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/training_curves_regularization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated training_curves_regularization.png")

def create_l1_vs_l2_comparison():
    """Compare L1 and L2 regularization effects on weights"""
    # Generate some sample weights for demonstration
    np.random.seed(42)
    n_weights = 100

    # Original weights (before regularization)
    original_weights = np.random.normal(0, 1, n_weights)

    # Simulate L1 regularization effect (promotes sparsity)
    l1_weights = original_weights.copy()
    threshold = 0.5
    l1_weights[np.abs(l1_weights) < threshold] = 0  # L1 sets small weights to zero
    l1_weights[l1_weights != 0] *= 0.8  # Shrink remaining weights

    # Simulate L2 regularization effect (uniform shrinkage)
    l2_weights = original_weights * 0.6  # L2 shrinks all weights uniformly

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Weight distributions
    bins = np.linspace(-2, 2, 30)

    axes[0, 0].hist(original_weights, bins=bins, alpha=0.7, color='blue', label='Original')
    axes[0, 0].set_title('Original Weight Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(l1_weights, bins=bins, alpha=0.7, color='red', label='L1 Regularized')
    axes[0, 1].set_title('L1 Regularization (Promotes Sparsity)')
    axes[0, 1].set_xlabel('Weight Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(l2_weights, bins=bins, alpha=0.7, color='green', label='L2 Regularized')
    axes[1, 0].set_title('L2 Regularization (Uniform Shrinkage)')
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # Comparison plot
    x_pos = np.arange(n_weights)
    axes[1, 1].scatter(x_pos, original_weights, alpha=0.6, color='blue', s=20, label='Original')
    axes[1, 1].scatter(x_pos, l1_weights, alpha=0.8, color='red', s=15, label='L1 Regularized')
    axes[1, 1].scatter(x_pos, l2_weights, alpha=0.8, color='green', s=10, label='L2 Regularized')
    axes[1, 1].set_title('Weight Values Comparison')
    axes[1, 1].set_xlabel('Weight Index')
    axes[1, 1].set_ylabel('Weight Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Add text annotations
    sparsity_l1 = np.sum(l1_weights == 0) / len(l1_weights) * 100
    sparsity_original = np.sum(original_weights == 0) / len(original_weights) * 100

    axes[0, 1].text(0.02, 0.98, f'Sparsity: {sparsity_l1:.1f}%\\n(Zero weights)',
                   transform=axes[0, 1].transAxes, va='top', ha='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    axes[1, 0].text(0.02, 0.98, f'Mean |weight|: {np.mean(np.abs(l2_weights)):.2f}\\n(Shrunk uniformly)',
                   transform=axes[1, 0].transAxes, va='top', ha='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/l1_vs_l2_regularization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated l1_vs_l2_regularization.png")

def main():
    """Generate all regularization technique figures"""
    print("Generating regularization technique visualizations...")

    create_overfitting_demo()
    create_dropout_visualization()
    create_regularization_comparison()
    create_training_curves_regularization()
    create_l1_vs_l2_comparison()

    print("✅ All regularization figures generated successfully!")

if __name__ == "__main__":
    main()