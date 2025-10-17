#!/usr/bin/env python3
"""
Neural Network Basics - Figure Generation
CMSC 173 - Machine Learning

This script generates visualizations for basic neural network concepts:
- Neuron structure and architecture
- Activation functions
- Simple network diagrams
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches
import os
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
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_perceptron_diagram():
    """Create a diagram showing the structure of a perceptron"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Input nodes
    input_positions = [(1, 3), (1, 2), (1, 1), (1, 0)]
    input_labels = ['$x_1$', '$x_2$', '$x_3$', '$x_0$ (bias)']
    input_colors = ['lightblue', 'lightblue', 'lightblue', 'lightcoral']

    # Draw input nodes
    for pos, label, color in zip(input_positions, input_labels, input_colors):
        circle = Circle(pos, 0.15, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=14, fontweight='bold')

    # Output node
    output_pos = (4, 1.5)
    output_circle = Circle(output_pos, 0.2, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(output_circle)
    ax.text(output_pos[0], output_pos[1], '$y$', ha='center', va='center', fontsize=16, fontweight='bold')

    # Draw connections with weights
    weights = ['$w_1$', '$w_2$', '$w_3$', '$b$']
    for i, (input_pos, weight) in enumerate(zip(input_positions, weights)):
        # Draw arrow
        ax.annotate('', xy=(output_pos[0]-0.2, output_pos[1]), xytext=(input_pos[0]+0.15, input_pos[1]),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))

        # Add weight label
        mid_x = (input_pos[0] + output_pos[0]) / 2
        mid_y = (input_pos[1] + output_pos[1]) / 2 + 0.1
        ax.text(mid_x, mid_y, weight, ha='center', va='center', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor='none'))

    # Add summation and activation function
    ax.text(2.8, 1.8, '$\\sum$', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(3.5, 1.8, '$\\sigma$', ha='center', va='center', fontsize=20, fontweight='bold')

    # Add formula
    ax.text(2.5, 0.2, '$z = \\sum_{i=1}^{n} w_i x_i + b$', ha='center', va='center',
           fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    ax.text(2.5, -0.2, '$y = \\sigma(z)$', ha='center', va='center',
           fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))

    ax.set_xlim(0.5, 5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Perceptron Structure', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/perceptron_structure.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated perceptron_structure.png")

def create_activation_functions():
    """Create plots showing different activation functions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = np.linspace(-5, 5, 1000)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    axes[0, 0].plot(x, sigmoid, color=COLOR_PALETTE['primary'], linewidth=3,
                    label='$\\sigma(x) = \\frac{1}{1+e^{-x}}$')
    axes[0, 0].set_title('Sigmoid Activation', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Input (x)')
    axes[0, 0].set_ylabel('Output')
    axes[0, 0].legend(frameon=True, shadow=True)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 0].spines['bottom'].set_linewidth(1.5)
    axes[0, 0].spines['left'].set_linewidth(1.5)

    # Tanh
    tanh = np.tanh(x)
    axes[0, 1].plot(x, tanh, color=COLOR_PALETTE['danger'], linewidth=3,
                    label='$\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$')
    axes[0, 1].set_title('Tanh Activation', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Input (x)')
    axes[0, 1].set_ylabel('Output')
    axes[0, 1].legend(frameon=True, shadow=True)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 1].spines['bottom'].set_linewidth(1.5)
    axes[0, 1].spines['left'].set_linewidth(1.5)

    # ReLU
    relu = np.maximum(0, x)
    axes[1, 0].plot(x, relu, color=COLOR_PALETTE['success'], linewidth=3,
                    label='$\\text{ReLU}(x) = \\max(0, x)$')
    axes[1, 0].set_title('ReLU Activation', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Input (x)')
    axes[1, 0].set_ylabel('Output')
    axes[1, 0].legend(frameon=True, shadow=True)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 0].spines['bottom'].set_linewidth(1.5)
    axes[1, 0].spines['left'].set_linewidth(1.5)

    # Leaky ReLU
    leaky_relu = np.where(x > 0, x, 0.1 * x)
    axes[1, 1].plot(x, leaky_relu, color=COLOR_PALETTE['secondary'], linewidth=3,
                    label='LeakyReLU(x)')
    axes[1, 1].set_title('Leaky ReLU Activation', fontweight='bold', fontsize=14)
    axes[1, 1].set_xlabel('Input (x)')
    axes[1, 1].set_ylabel('Output')
    axes[1, 1].legend(frameon=True, shadow=True)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 1].spines['bottom'].set_linewidth(1.5)
    axes[1, 1].spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/activation_functions.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated activation_functions.png")

def create_activation_derivatives():
    """Create plots showing derivatives of activation functions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x = np.linspace(-5, 5, 1000)

    # Sigmoid derivative
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)
    axes[0, 0].plot(x, sigmoid_derivative, color=COLOR_PALETTE['primary'], linewidth=3,
                    label="$\\sigma'(x) = \\sigma(x)(1-\\sigma(x))$")
    axes[0, 0].set_title('Sigmoid Derivative', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Input (x)')
    axes[0, 0].set_ylabel("$\\sigma'(x)$")
    axes[0, 0].legend(frameon=True, shadow=True)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 0].spines['bottom'].set_linewidth(1.5)
    axes[0, 0].spines['left'].set_linewidth(1.5)

    # Tanh derivative
    tanh = np.tanh(x)
    tanh_derivative = 1 - tanh**2
    axes[0, 1].plot(x, tanh_derivative, color=COLOR_PALETTE['danger'], linewidth=3,
                    label="$\\tanh'(x) = 1 - \\tanh^2(x)$")
    axes[0, 1].set_title('Tanh Derivative', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Input (x)')
    axes[0, 1].set_ylabel("$\\tanh'(x)$")
    axes[0, 1].legend(frameon=True, shadow=True)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 1].spines['bottom'].set_linewidth(1.5)
    axes[0, 1].spines['left'].set_linewidth(1.5)

    # ReLU derivative
    relu_derivative = np.where(x > 0, 1, 0)
    axes[1, 0].plot(x, relu_derivative, color=COLOR_PALETTE['success'], linewidth=3,
                    label="ReLU'(x)")
    axes[1, 0].set_title('ReLU Derivative', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Input (x)')
    axes[1, 0].set_ylabel("$\\text{ReLU}'(x)$")
    axes[1, 0].legend(frameon=True, shadow=True)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].spines['bottom'].set_linewidth(1.5)
    axes[1, 0].spines['left'].set_linewidth(1.5)

    # Comparison of gradients
    axes[1, 1].plot(x, sigmoid_derivative, color=COLOR_PALETTE['primary'], linewidth=3, label='Sigmoid')
    axes[1, 1].plot(x, tanh_derivative, color=COLOR_PALETTE['danger'], linewidth=3, label='Tanh')
    axes[1, 1].plot(x, relu_derivative, color=COLOR_PALETTE['success'], linewidth=3, label='ReLU')
    axes[1, 1].set_title('Derivative Comparison', fontweight='bold', fontsize=14)
    axes[1, 1].set_xlabel('Input (x)')
    axes[1, 1].set_ylabel('Derivative')
    axes[1, 1].legend(frameon=True, shadow=True)
    axes[1, 1].spines['bottom'].set_linewidth(1.5)
    axes[1, 1].spines['left'].set_linewidth(1.5)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/activation_derivatives.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated activation_derivatives.png")

def create_multilayer_network():
    """Create a diagram of a multi-layer neural network"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define layers
    layer_sizes = [4, 5, 3, 2]
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    layer_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    # Layer positions
    layer_x = [1, 3, 5, 7]

    # Draw nodes
    node_positions = []
    for i, (x, size, color) in enumerate(zip(layer_x, layer_sizes, layer_colors)):
        positions = []
        y_start = (5 - size) / 2
        for j in range(size):
            y = y_start + j
            circle = Circle((x, y), 0.15, color=color, ec='black', linewidth=1.5)
            ax.add_patch(circle)
            positions.append((x, y))

            # Add labels for input and output
            if i == 0:
                ax.text(x, y, f'$x_{j+1}$', ha='center', va='center', fontsize=10, fontweight='bold')
            elif i == len(layer_sizes) - 1:
                ax.text(x, y, f'$y_{j+1}$', ha='center', va='center', fontsize=10, fontweight='bold')

        node_positions.append(positions)

        # Layer labels
        ax.text(x, -0.8, layer_names[i], ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw connections
    for i in range(len(layer_sizes) - 1):
        for pos1 in node_positions[i]:
            for pos2 in node_positions[i + 1]:
                ax.plot([pos1[0] + 0.15, pos2[0] - 0.15], [pos1[1], pos2[1]],
                       'gray', linewidth=0.5, alpha=0.7)

    # Add mathematical notation
    ax.text(2, 5.5, '$W^{(1)}, b^{(1)}$', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray'))
    ax.text(4, 5.5, '$W^{(2)}, b^{(2)}$', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray'))
    ax.text(6, 5.5, '$W^{(3)}, b^{(3)}$', ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='gray'))

    # Add forward propagation arrows
    for i in range(len(layer_x) - 1):
        ax.annotate('', xy=(layer_x[i+1] - 0.5, -0.3), xytext=(layer_x[i] + 0.5, -0.3),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax.text(4, -0.1, 'Forward Propagation', ha='center', va='center', fontsize=12,
           color='red', fontweight='bold')

    ax.set_xlim(0, 8)
    ax.set_ylim(-1.2, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multi-Layer Neural Network Architecture', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/multilayer_network.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Generated multilayer_network.png")

def main():
    """Generate all basic neural network figures"""
    print("Generating basic neural network visualizations...")

    create_perceptron_diagram()
    create_activation_functions()
    create_activation_derivatives()
    create_multilayer_network()

    print("✅ All basic neural network figures generated successfully!")

if __name__ == "__main__":
    main()