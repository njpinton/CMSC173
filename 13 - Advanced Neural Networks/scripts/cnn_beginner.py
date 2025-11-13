#!/usr/bin/env python3
"""
Convolutional Neural Networks - Beginner-Friendly Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

Simple, intuitive visualizations for understanding CNNs.
Focus on concepts, not complex math!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow
import warnings
warnings.filterwarnings('ignore')

# Simple, beginner-friendly styling
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = False

COLORS = {
    'blue': '#4A90E2',
    'green': '#7ED321',
    'orange': '#F5A623',
    'red': '#E74C3C',
    'purple': '#9B59B6',
    'yellow': '#F1C40F',
}

def create_output_dir():
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_cnn_intuition():
    """Simple visual explanation of what CNNs do"""
    fig = plt.figure(figsize=(16, 10))

    # Create a simple "image" (smiley face pattern)
    image = np.zeros((8, 8))
    # Eyes
    image[2, 2] = 1
    image[2, 5] = 1
    # Smile
    image[5, 2:6] = 1
    image[6, 3:5] = 0.5

    # Left side: Show the concept
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image, cmap='gray', interpolation='nearest')
    ax1.set_title('Input Image\n(Like a tiny photo)', fontsize=13, fontweight='bold')
    ax1.axis('off')
    ax1.text(0.5, -0.15, '😊 Can we teach a computer to see this?',
            transform=ax1.transAxes, ha='center', fontsize=11, style='italic')

    # Show a filter (detector for vertical lines)
    ax2 = plt.subplot(2, 3, 2)
    filter_vertical = np.array([[-1, 1], [-1, 1]])
    im = ax2.imshow(filter_vertical, cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_title('Filter (Detector)\n"Looking for edges"', fontsize=13, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    ax2.text(0.5, -0.15, '🔍 Like a magnifying glass scanning the image',
            transform=ax2.transAxes, ha='center', fontsize=11, style='italic')

    # Show what the filter detects
    ax3 = plt.subplot(2, 3, 3)
    # Simple convolution result
    detected = np.abs(image[:-1, :-1] - image[1:, 1:])
    ax3.imshow(detected, cmap='hot', interpolation='nearest')
    ax3.set_title('What Filter Found\n"Edge map"', fontsize=13, fontweight='bold')
    ax3.axis('off')
    ax3.text(0.5, -0.15, '✨ Highlighted important patterns!',
            transform=ax3.transAxes, ha='center', fontsize=11, style='italic')

    # Bottom: Real-world analogy
    ax4 = plt.subplot(2, 1, 2)
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    explanation = """
    🎯 CNN IN SIMPLE TERMS:

    Think of CNNs like having many "smart glasses" that look at an image:

    👓 Glass #1: "I spot vertical lines!" (detects |)
    👓 Glass #2: "I see horizontal lines!" (detects —)
    👓 Glass #3: "I found circles!" (detects ⭕)
    👓 Glass #4: "I detect corners!" (detects ⌐)

    ➡️ Step 1: SCAN - Each glass scans the entire image
    ➡️ Step 2: DETECT - They highlight what they find
    ➡️ Step 3: COMBINE - All findings are combined
    ➡️ Step 4: DECIDE - "This looks like a cat!" 🐱

    📝 Key Ideas (No Complex Math!):
    • CNNs look at SMALL PIECES of an image at a time (like using a magnifying glass)
    • They learn to recognize PATTERNS (edges, shapes, textures)
    • Simple patterns → Complex patterns → Full objects
    • Example: edges → curves → wheels → CAR! 🚗

    💡 Why CNNs are Special:
    ✓ They learn WHAT to look for (unlike old methods where humans had to tell them)
    ✓ They understand images the way humans do (step by step)
    ✓ They work great for photos, videos, medical scans, anything visual!
    """

    ax4.text(0.5, 9, explanation, fontsize=10.5, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round,pad=1',
            facecolor='lightyellow', edgecolor='black', linewidth=2))

    plt.suptitle('Understanding CNNs: The "Smart Glasses" for Computers 👓',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/cnn_simple_intuition.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated cnn_simple_intuition.png")


def plot_cnn_architecture_simple():
    """Super simple CNN architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Draw simple boxes representing layers
    layers = [
        ("📷 Input\nImage\n(Cat photo)", 1, COLORS['blue']),
        ("🔍 Layer 1\nFind edges\n(lines, curves)", 3.5, COLORS['green']),
        ("🔍 Layer 2\nFind shapes\n(ears, eyes)", 6, COLORS['orange']),
        ("🔍 Layer 3\nFind parts\n(face, body)", 8.5, COLORS['purple']),
        ("🧠 Decision\n\"It's a CAT!\"\n🐱", 11.5, COLORS['red']),
    ]

    for i, (label, x, color) in enumerate(layers):
        # Draw box
        box = FancyBboxPatch((x-0.6, 2), 1.2, 4, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=3, alpha=0.7)
        ax.add_patch(box)

        # Add label
        ax.text(x, 4, label, ha='center', va='center', fontsize=11,
               fontweight='bold', color='white')

        # Add arrow to next layer
        if i < len(layers) - 1:
            ax.arrow(x + 0.7, 4, 1, 0, head_width=0.3, head_length=0.2,
                    fc='black', ec='black', linewidth=2)

    # Add explanation below
    explanation = """
    🎓 HOW IT WORKS (Simple Version):

    1️⃣ START: Computer sees image as numbers (pixels)
    2️⃣ EARLY LAYERS: Find simple things (Is it bright? Dark? Where are edges?)
    3️⃣ MIDDLE LAYERS: Combine simple things into shapes (circles, rectangles, curves)
    4️⃣ LATE LAYERS: Recognize complex objects (eyes, wheels, faces)
    5️⃣ END: Make decision ("This is a cat with 95% confidence!")

    💭 Think of it like LEARNING TO READ:
    First you learn letters (A, B, C) → Then words (CAT, DOG) → Then sentences → Then stories!
    CNNs do the same with images: pixels → edges → shapes → objects!
    """

    ax.text(7.5, 0.5, explanation, ha='center', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue',
                    edgecolor='black', linewidth=2, alpha=0.8))

    plt.title('CNN Architecture: From Simple to Complex (Layer by Layer)',
             fontsize=16, fontweight='bold', pad=20)

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/cnn_simple_architecture.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated cnn_simple_architecture.png")


def plot_cnn_applications():
    """Show real-world CNN applications with icons"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    applications = [
        ("📸 Photo Recognition", "Your phone:\n'This is a beach!'", COLORS['blue']),
        ("🚗 Self-Driving Cars", "Detecting pedestrians,\nstop signs, other cars", COLORS['green']),
        ("🏥 Medical Diagnosis", "Finding tumors in\nX-rays and MRI scans", COLORS['red']),
        ("🔐 Face Unlock", "Your phone recognizes\nYOUR face to unlock", COLORS['purple']),
        ("🎨 Image Filters", "Snapchat/Instagram\nfun face filters", COLORS['orange']),
        ("🔍 Visual Search", "Google Lens:\nFind similar products", COLORS['yellow']),
    ]

    for ax, (title, desc, color) in zip(axes, applications):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Draw colored box
        box = FancyBboxPatch((1, 2), 8, 6, boxstyle="round,pad=0.3",
                            facecolor=color, edgecolor='black', linewidth=3, alpha=0.6)
        ax.add_patch(box)

        # Add title and description
        ax.text(5, 7, title, ha='center', va='center', fontsize=14,
               fontweight='bold', color='white')
        ax.text(5, 4, desc, ha='center', va='center', fontsize=11,
               color='white', style='italic')

    plt.suptitle('CNN Applications: Where You See Them Every Day! 🌟',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/cnn_applications.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated cnn_applications.png")


def plot_cnn_vs_traditional():
    """Compare CNN approach vs traditional approach"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Traditional approach
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Draw flow
    y_pos = 11
    boxes_old = [
        ("📷 Image", y_pos, COLORS['blue']),
        ("👨‍💻 Humans design\nfeature detectors\n(hard work!)", y_pos-2.5, COLORS['red']),
        ("📊 Extract features\nmanually", y_pos-5, COLORS['orange']),
        ("🤖 Simple classifier\n(like decision tree)", y_pos-7.5, COLORS['purple']),
        ("✅ Result", y_pos-10, COLORS['green']),
    ]

    for label, y, color in boxes_old:
        box = FancyBboxPatch((2, y-0.6), 6, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(5, y, label, ha='center', va='center', fontsize=10,
               fontweight='bold', color='white')

        if y > y_pos - 10:
            ax.arrow(5, y-0.7, 0, -0.6, head_width=0.3, head_length=0.2,
                    fc='black', ec='black', linewidth=2)

    ax.text(5, 0.5, "❌ Problem: Humans had to figure\nout what features to look for!",
           ha='center', fontsize=10, fontweight='bold', color='red',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow'))

    ax.set_title('OLD WAY (Before CNNs)', fontsize=14, fontweight='bold', pad=20)

    # Right: CNN approach
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    boxes_new = [
        ("📷 Image", y_pos, COLORS['blue']),
        ("🧠 CNN learns\nfeatures automatically\n(magic!)", y_pos-2.5, COLORS['green']),
        ("🔍 Discovers patterns\non its own", y_pos-5, COLORS['orange']),
        ("🎯 Makes decisions\nend-to-end", y_pos-7.5, COLORS['purple']),
        ("✅ Result", y_pos-10, COLORS['green']),
    ]

    for label, y, color in boxes_new:
        box = FancyBboxPatch((2, y-0.6), 6, 1.2, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(5, y, label, ha='center', va='center', fontsize=10,
               fontweight='bold', color='white')

        if y > y_pos - 10:
            ax.arrow(5, y-0.7, 0, -0.6, head_width=0.3, head_length=0.2,
                    fc='black', ec='black', linewidth=2)

    ax.text(5, 0.5, "✅ Advantage: CNN learns\nthe best features itself!",
           ha='center', fontsize=10, fontweight='bold', color='green',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen'))

    ax.set_title('NEW WAY (CNNs)', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Why CNNs Changed Everything 🚀',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/cnn_vs_traditional.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated cnn_vs_traditional.png")


def main():
    """Generate all beginner-friendly CNN figures"""
    print("="*60)
    print("Generating Beginner-Friendly CNN Visualizations")
    print("="*60)

    plot_cnn_intuition()
    plot_cnn_architecture_simple()
    plot_cnn_applications()
    plot_cnn_vs_traditional()

    print("="*60)
    print("✅ CNN figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
