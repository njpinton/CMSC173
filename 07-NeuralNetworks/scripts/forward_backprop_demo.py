#!/usr/bin/env python3
"""
Forward Propagation and Backpropagation Demonstrations
CMSC 173 - Machine Learning

This script generates visualizations for:
- Forward propagation flow
- Backpropagation algorithm
- Gradient flow and computation graphs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches
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

def create_forward_propagation_flow():
    """Create a detailed forward propagation visualization"""
    fig, ax = plt.subplots(figsize=(15, 10))

    # Network structure: 3 -> 4 -> 2
    layers = [(1, 3), (4, 4), (7, 2)]
    layer_names = ['Input Layer\n($a^{(0)}$)', 'Hidden Layer\n($a^{(1)}$)', 'Output Layer\n($a^{(2)}$)']

    # Draw nodes with values
    node_positions = []
    sample_values = [
        [0.8, 0.3, 0.5],  # Input values
        [0.7, 0.9, 0.2, 0.6],  # Hidden values
        [0.85, 0.15]  # Output values
    ]

    colors = ['lightblue', 'lightgreen', 'lightcoral']

    for layer_idx, ((x, size), values, color) in enumerate(zip(layers, sample_values, colors)):
        positions = []
        y_start = (4 - size) / 2 + 0.5
        for i in range(size):
            y = y_start + i * 1.2
            circle = Circle((x, y), 0.2, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f'{values[i]:.2f}', ha='center', va='center',
                   fontsize=10, fontweight='bold')
            positions.append((x, y))
        node_positions.append(positions)

        # Layer labels
        ax.text(x, 0, layer_names[layer_idx], ha='center', va='center',
               fontsize=12, fontweight='bold')

    # Draw connections with sample weights
    np.random.seed(42)

    # Input to hidden
    for i, pos1 in enumerate(node_positions[0]):
        for j, pos2 in enumerate(node_positions[1]):
            weight = np.random.uniform(0.2, 0.8)
            ax.plot([pos1[0] + 0.2, pos2[0] - 0.2], [pos1[1], pos2[1]],
                   'blue', linewidth=weight*3, alpha=0.7)

            # Add some weight labels
            if i == 0 and j < 2:
                mid_x = (pos1[0] + pos2[0]) / 2
                mid_y = (pos1[1] + pos2[1]) / 2
                ax.text(mid_x, mid_y + 0.15, f'w={weight:.2f}', ha='center', va='center',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))

    # Hidden to output
    for i, pos1 in enumerate(node_positions[1]):
        for j, pos2 in enumerate(node_positions[2]):
            weight = np.random.uniform(0.3, 0.7)
            ax.plot([pos1[0] + 0.2, pos2[0] - 0.2], [pos1[1], pos2[1]],
                   'green', linewidth=weight*3, alpha=0.7)

    # Add computation steps
    computation_steps = [
        (2.5, 4, '$z^{(1)} = W^{(1)}a^{(0)} + b^{(1)}$'),
        (2.5, 3.5, '$a^{(1)} = \\sigma(z^{(1)})$'),
        (5.5, 4, '$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$'),
        (5.5, 3.5, '$a^{(2)} = \\sigma(z^{(2)})$')
    ]

    for x, y, text in computation_steps:
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='orange'))

    # Add arrows for forward flow
    arrow_props = dict(arrowstyle='->', lw=3, color='red')
    ax.annotate('', xy=(3.5, 2.5), xytext=(1.5, 2.5), arrowprops=arrow_props)
    ax.annotate('', xy=(6.5, 2.5), xytext=(4.5, 2.5), arrowprops=arrow_props)

    ax.text(4, 1.8, 'Forward Propagation Flow', ha='center', va='center',
           fontsize=14, fontweight='bold', color='red')

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Forward Propagation: Information Flow Through Network',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/forward_propagation_flow.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated forward_propagation_flow.png")

def create_backpropagation_flow():
    """Create a detailed backpropagation visualization"""
    fig, ax = plt.subplots(figsize=(15, 10))

    # Network structure: 3 -> 4 -> 2
    layers = [(1, 3), (4, 4), (7, 2)]
    layer_names = ['Input Layer', 'Hidden Layer', 'Output Layer']

    # Draw nodes
    node_positions = []
    colors = ['lightblue', 'lightgreen', 'lightcoral']

    for layer_idx, ((x, size), color) in enumerate(zip(layers, colors)):
        positions = []
        y_start = (4 - size) / 2 + 0.5
        for i in range(size):
            y = y_start + i * 1.2
            circle = Circle((x, y), 0.2, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            positions.append((x, y))
        node_positions.append(positions)

        # Layer labels
        ax.text(x, 0, layer_names[layer_idx], ha='center', va='center',
               fontsize=12, fontweight='bold')

    # Draw connections
    for i in range(len(layers) - 1):
        for pos1 in node_positions[i]:
            for pos2 in node_positions[i + 1]:
                ax.plot([pos1[0] + 0.2, pos2[0] - 0.2], [pos1[1], pos2[1]],
                       'gray', linewidth=1, alpha=0.5)

    # Add gradient flow arrows (backward)
    arrow_props_back = dict(arrowstyle='->', lw=4, color='purple')
    ax.annotate('', xy=(4.5, 2.5), xytext=(6.5, 2.5), arrowprops=arrow_props_back)
    ax.annotate('', xy=(1.5, 2.5), xytext=(3.5, 2.5), arrowprops=arrow_props_back)

    # Add computation steps for backpropagation
    backprop_steps = [
        (5.5, 4.2, 'Compute Output Error'),
        (5.5, 3.8, '$\\delta^{(2)} = \\frac{\\partial L}{\\partial z^{(2)}}$'),
        (2.5, 4.2, 'Propagate Error Backward'),
        (2.5, 3.8, '$\\delta^{(1)} = (W^{(2)})^T \\delta^{(2)} \\odot \\sigma\'(z^{(1)})$'),
        (5.5, 1, 'Update Weights'),
        (5.5, 0.6, '$W^{(2)} := W^{(2)} - \\alpha \\frac{\\partial L}{\\partial W^{(2)}}$'),
        (2.5, 1, 'Update Weights'),
        (2.5, 0.6, '$W^{(1)} := W^{(1)} - \\alpha \\frac{\\partial L}{\\partial W^{(1)}}$')
    ]

    for x, y, text in backprop_steps:
        color = 'lightpink' if 'Error' in text or 'Propagate' in text else 'lightcyan'
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor='purple'))

    # Add error gradient visualization
    ax.text(7, 4.5, 'Loss\n$L$', ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))

    # Draw gradient arrows from loss
    for pos in node_positions[2]:
        ax.annotate('', xy=pos, xytext=(7, 4.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.7))

    ax.text(4, 1.3, 'Backpropagation: Error Gradient Flow', ha='center', va='center',
           fontsize=14, fontweight='bold', color='purple')

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Backpropagation: Gradient Flow and Weight Updates',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/backpropagation_flow.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated backpropagation_flow.png")

def create_computational_graph():
    """Create a computational graph showing forward and backward passes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Forward pass computational graph
    nodes = {
        'x': (1, 3),
        'w': (1, 2),
        'b': (1, 1),
        'z': (3, 2),
        'a': (5, 2),
        'L': (7, 2)
    }

    # Draw nodes for forward pass
    for name, (x, y) in nodes.items():
        if name in ['x', 'w', 'b']:
            color = 'lightblue'
        elif name in ['z', 'a']:
            color = 'lightgreen'
        else:
            color = 'lightcoral'

        circle = Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold')

    # Draw operations
    operations = {
        '*': (2, 2.5),
        '+': (2, 1.5),
        'σ': (4, 2),
        'Loss': (6, 2)
    }

    for op, (x, y) in operations.items():
        if op == 'Loss':
            rect = Rectangle((x-0.3, y-0.2), 0.6, 0.4, color='yellow', ec='black', linewidth=2)
            ax1.add_patch(rect)
        else:
            rect = Rectangle((x-0.2, y-0.2), 0.4, 0.4, color='white', ec='black', linewidth=2)
            ax1.add_patch(rect)
        ax1.text(x, y, op, ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw edges for forward pass
    forward_edges = [
        ('x', '*'), ('w', '*'), ('*', '+'), ('b', '+'), ('+', 'σ'), ('σ', 'Loss')
    ]

    for start, end in forward_edges:
        if start in nodes:
            x1, y1 = nodes[start]
            x1 += 0.3
        else:
            x1, y1 = operations[start]
            x1 += 0.2

        if end in nodes:
            x2, y2 = nodes[end]
            x2 -= 0.3
        else:
            x2, y2 = operations[end]
            x2 -= 0.2

        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 4)
    ax1.set_title('Forward Pass: Computational Graph', fontweight='bold')
    ax1.axis('off')

    # Backward pass computational graph
    # Draw same nodes but with gradients
    for name, (x, y) in nodes.items():
        if name in ['x', 'w', 'b']:
            color = 'lightblue'
        elif name in ['z', 'a']:
            color = 'lightgreen'
        else:
            color = 'lightcoral'

        circle = Circle((x, y), 0.3, color=color, ec='black', linewidth=2)
        ax2.add_patch(circle)
        gradient_name = f'∂L/∂{name}' if name != 'L' else '1'
        ax2.text(x, y, gradient_name, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw operations for backward pass
    for op, (x, y) in operations.items():
        if op == 'Loss':
            rect = Rectangle((x-0.3, y-0.2), 0.6, 0.4, color='yellow', ec='black', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x, y, '∂L', ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            rect = Rectangle((x-0.2, y-0.2), 0.4, 0.4, color='white', ec='black', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x, y, op, ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw edges for backward pass (reverse direction)
    backward_edges = [
        ('Loss', 'σ'), ('σ', '+'), ('+', '*'), ('+', 'b'), ('*', 'x'), ('*', 'w')
    ]

    for start, end in backward_edges:
        if start in nodes:
            x1, y1 = nodes[start]
            x1 -= 0.3
        else:
            x1, y1 = operations[start]
            x1 -= 0.2

        if end in nodes:
            x2, y2 = nodes[end]
            x2 += 0.3
        else:
            x2, y2 = operations[end]
            x2 += 0.2

        ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 4)
    ax2.set_title('Backward Pass: Gradient Computation', fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/computational_graph.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated computational_graph.png")

def create_gradient_descent_visualization():
    """Create visualization of gradient descent optimization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1D gradient descent
    x = np.linspace(-2, 2, 100)
    y = x**2 + 0.1*x + 0.5  # Simple quadratic function

    ax1.plot(x, y, 'b-', linewidth=2, label='Loss Function')

    # Gradient descent steps
    x_steps = [1.5, 1.0, 0.6, 0.3, 0.1, 0.05]
    y_steps = [xi**2 + 0.1*xi + 0.5 for xi in x_steps]

    for i, (xi, yi) in enumerate(zip(x_steps, y_steps)):
        ax1.plot(xi, yi, 'ro', markersize=8)
        if i < len(x_steps) - 1:
            ax1.annotate('', xy=(x_steps[i+1], y_steps[i+1]), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax1.text(xi, yi + 0.3, f'Step {i}', ha='center', va='bottom', fontsize=10)

    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Loss')
    ax1.set_title('1D Gradient Descent', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2D contour plot for multiple weights
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = (W1**2 + W2**2 + 0.5*W1*W2 + 0.3*W1 + 0.2*W2)

    contour = ax2.contour(W1, W2, Z, levels=20, colors='blue', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)

    # Gradient descent path in 2D
    path_x = [1.5, 1.1, 0.7, 0.3, 0.1, 0.02]
    path_y = [1.2, 0.8, 0.5, 0.2, 0.05, 0.01]

    ax2.plot(path_x, path_y, 'ro-', markersize=6, linewidth=2, label='Gradient Descent Path')
    ax2.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
    ax2.plot(path_x[-1], path_y[-1], 'ko', markersize=10, label='End')

    # Add arrows
    for i in range(len(path_x) - 1):
        ax2.annotate('', xy=(path_x[i+1], path_y[i+1]), xytext=(path_x[i], path_y[i]),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

    ax2.set_xlabel('Weight 1')
    ax2.set_ylabel('Weight 2')
    ax2.set_title('2D Gradient Descent (Contour Plot)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/gradient_descent_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated gradient_descent_visualization.png")

def main():
    """Generate all forward propagation and backpropagation figures"""
    print("Generating forward propagation and backpropagation visualizations...")

    create_forward_propagation_flow()
    create_backpropagation_flow()
    create_computational_graph()
    create_gradient_descent_visualization()

    print("✅ All forward/backpropagation figures generated successfully!")

if __name__ == "__main__":
    main()