#!/usr/bin/env python3
"""
Diffusion Models - Beginner-Friendly Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

Simple, intuitive visualizations for understanding Diffusion Models.
Focus on concepts, not complex math!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, Ellipse
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
    'pink': '#FF6B9D',
    'teal': '#1ABC9C',
}

def create_output_dir():
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_diffusion_concept():
    """Simple diffusion concept - like reverse blurring"""
    fig = plt.figure(figsize=(16, 11))

    plt.suptitle('Diffusion Models: The Magic of "Denoising" 🪄✨',
                fontsize=16, fontweight='bold', y=0.98)

    # Forward process (adding noise)
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 4)
    ax1.axis('off')

    ax1.text(7, 3.7, '➡️ FORWARD PROCESS: Gradually Add Noise (Training)', ha='center',
            fontsize=13, fontweight='bold', color=COLORS['red'])

    # Show progressive noising
    steps = 5
    x_start = 1.5
    spacing = 2.5

    # Create fake "image" that gets progressively noisier
    for i in range(steps):
        x = x_start + i * spacing
        noise_level = i / (steps - 1)

        # Draw box
        box = FancyBboxPatch((x-0.7, 0.8), 1.4, 1.8, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor='black', linewidth=3)
        ax1.add_patch(box)

        # Simulate noise with random dots
        if i == 0:
            # Clean image (smiley)
            ax1.text(x, 1.7, '😊', fontsize=40, ha='center', va='center')
            ax1.text(x, 0.3, 'Clean Image', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['green'])
        elif i == steps - 1:
            # Pure noise
            np.random.seed(42)
            for _ in range(50):
                dx = np.random.uniform(-0.6, 0.6)
                dy = np.random.uniform(-0.8, 0.8)
                ax1.plot(x + dx, 1.7 + dy, 'o', color='gray',
                        markersize=np.random.uniform(1, 3))
            ax1.text(x, 0.3, 'Pure Noise', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['red'])
        else:
            # Partial noise
            ax1.text(x, 1.7, '😊', fontsize=40, ha='center', va='center',
                    alpha=1 - noise_level)
            np.random.seed(42 + i)
            for _ in range(int(30 * noise_level)):
                dx = np.random.uniform(-0.6, 0.6)
                dy = np.random.uniform(-0.8, 0.8)
                ax1.plot(x + dx, 1.7 + dy, 'o', color='gray',
                        markersize=np.random.uniform(1, 3))
            ax1.text(x, 0.3, f'Step {i}', ha='center', fontsize=9,
                    style='italic')

        # Arrow to next step
        if i < steps - 1:
            ax1.annotate('', xy=(x + 1.2, 1.7), xytext=(x + 0.8, 1.7),
                        arrowprops=dict(arrowstyle='->', lw=3, color='red'))
            ax1.text(x + 1, 2.8, '+noise', ha='center', fontsize=8,
                    color='red', style='italic')

    # Reverse process (removing noise)
    ax2 = plt.subplot(3, 1, 2)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 4)
    ax2.axis('off')

    ax2.text(7, 3.7, '⬅️ REVERSE PROCESS: Gradually Remove Noise (Generation)', ha='center',
            fontsize=13, fontweight='bold', color=COLORS['green'])

    # Show progressive denoising
    for i in range(steps):
        x = x_start + i * spacing
        noise_level = 1 - (i / (steps - 1))  # Reverse

        # Draw box
        box = FancyBboxPatch((x-0.7, 0.8), 1.4, 1.8, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor='black', linewidth=3)
        ax2.add_patch(box)

        # Simulate denoising
        if i == 0:
            # Start with pure noise
            np.random.seed(42)
            for _ in range(50):
                dx = np.random.uniform(-0.6, 0.6)
                dy = np.random.uniform(-0.8, 0.8)
                ax2.plot(x + dx, 1.7 + dy, 'o', color='gray',
                        markersize=np.random.uniform(1, 3))
            ax2.text(x, 0.3, 'Start: Random', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['red'])
        elif i == steps - 1:
            # End with clean image
            ax2.text(x, 1.7, '🎨', fontsize=40, ha='center', va='center')
            ax2.text(x, 0.3, 'Final: Art! ✨', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['green'])
        else:
            # Partial denoising
            ax2.text(x, 1.7, '🎨', fontsize=40, ha='center', va='center',
                    alpha=1 - noise_level)
            np.random.seed(42 + i)
            for _ in range(int(30 * noise_level)):
                dx = np.random.uniform(-0.6, 0.6)
                dy = np.random.uniform(-0.8, 0.8)
                ax2.plot(x + dx, 1.7 + dy, 'o', color='gray',
                        markersize=np.random.uniform(1, 3))
            ax2.text(x, 0.3, f'Step {i}', ha='center', fontsize=9,
                    style='italic')

        # Arrow to next step
        if i < steps - 1:
            ax2.annotate('', xy=(x + 1.2, 1.7), xytext=(x + 0.8, 1.7),
                        arrowprops=dict(arrowstyle='->', lw=3, color='green'))
            ax2.text(x + 1, 2.8, '-noise', ha='center', fontsize=8,
                    color='green', style='italic')

    # Explanation
    ax3 = plt.subplot(3, 1, 3)
    ax3.axis('off')

    explanation = """
    🎯 DIFFUSION MODELS IN SIMPLE TERMS:

    💭 THE CORE IDEA: Learn to reverse the process of adding noise!

    📚 ANALOGY - Like a Magic Eraser:
    Imagine you have a beautiful drawing, but someone scribbled all over it with random marks.
    A diffusion model is like a magic eraser that can gradually remove the scribbles
    until the original drawing appears again!

    🔄 TWO PHASES:

    1️⃣ TRAINING (Forward): Add noise step by step
       • Start with real images (like photos of cats)
       • Gradually add random noise until it's completely fuzzy
       • Save each step to learn the pattern

    2️⃣ GENERATION (Reverse): Remove noise step by step
       • Start with pure random noise (like TV static)
       • AI predicts: "What does this look like with LESS noise?"
       • Remove a little noise at a time
       • Eventually reveals a clear image!

    🤖 HOW THE AI LEARNS:
    During training, the AI learns: "Given a noisy image, what's the clean version?"
    It practices this over and over with millions of images.
    Then during generation, it applies this skill repeatedly to create new images!

    ✨ WHY IT'S SPECIAL:
    • Very stable training (unlike GANs that can be tricky)
    • High quality results (super realistic images!)
    • Controllable (you can guide what it creates with text)
    • Flexible (works for images, audio, video, 3D models)

    🎨 THE MAGIC:
    You start with random noise (complete chaos) and step-by-step, the AI
    sculpts it into a beautiful image - like a sculptor revealing a statue
    hidden inside a block of marble!
    """

    ax3.text(0.5, 0.5, explanation, transform=ax3.transAxes,
            fontsize=10.5, verticalalignment='center', ha='center',
            family='monospace', bbox=dict(boxstyle='round,pad=1',
            facecolor='lightyellow', edgecolor='black', linewidth=2))

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/diffusion_simple_concept.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated diffusion_simple_concept.png")


def plot_text_to_image():
    """How text-to-image generation works (Stable Diffusion, DALL-E)"""
    fig = plt.figure(figsize=(16, 10))

    plt.suptitle('Text-to-Image: Turning Words into Pictures! 📝➡️🖼️',
                fontsize=16, fontweight='bold', y=0.98)

    # Main process
    ax = plt.subplot(2, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Step 1: Text input
    y = 9
    text_input = '"A cat astronaut in space"'
    input_box = FancyBboxPatch((2, y-0.4), 10, 0.8, boxstyle="round,pad=0.2",
                              facecolor=COLORS['blue'], edgecolor='black',
                              linewidth=3, alpha=0.7)
    ax.add_patch(input_box)
    ax.text(7, y, f'💬 Your Prompt: {text_input}', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')

    # Arrow down
    ax.arrow(7, 8.3, 0, -0.4, head_width=0.4, head_length=0.2,
            fc='black', ec='black', linewidth=3)

    # Step 2: Text encoder
    y = 7.5
    encoder_box = FancyBboxPatch((3.5, y-0.5), 7, 1, boxstyle="round,pad=0.2",
                                facecolor=COLORS['purple'], edgecolor='black',
                                linewidth=3, alpha=0.7)
    ax.add_patch(encoder_box)
    ax.text(7, y+0.2, '🧠 Text Encoder (like mini-ChatGPT)', ha='center',
           fontsize=11, fontweight='bold', color='white')
    ax.text(7, y-0.2, 'Understands: cat + astronaut + space', ha='center',
           fontsize=9, color='white', style='italic')

    # Arrow down
    ax.arrow(7, 6.8, 0, -0.4, head_width=0.4, head_length=0.2,
            fc='black', ec='black', linewidth=3)

    # Step 3: Initial noise
    y = 6
    ax.text(2, y, '🎲 Start with\nrandom noise:', ha='center', fontsize=10,
           fontweight='bold')

    noise_box = FancyBboxPatch((3, y-0.6), 1.2, 1.2, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(noise_box)
    # Draw noise
    np.random.seed(123)
    for _ in range(60):
        dx = np.random.uniform(-0.5, 0.5)
        dy = np.random.uniform(-0.5, 0.5)
        ax.plot(3.6 + dx, y + dy, 'o', color='gray',
               markersize=np.random.uniform(1, 2))

    # Arrow to process
    ax.arrow(4.5, y, 1, 0, head_width=0.3, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Step 4: Denoising loop
    denoise_box = FancyBboxPatch((6, y-0.7), 3.5, 1.4, boxstyle="round,pad=0.2",
                                facecolor=COLORS['orange'], edgecolor='black',
                                linewidth=3, alpha=0.7)
    ax.add_patch(denoise_box)
    ax.text(7.75, y+0.3, '🔄 Denoising Loop', ha='center', fontsize=11,
           fontweight='bold', color='white')
    ax.text(7.75, y-0.05, 'Remove noise guided', ha='center', fontsize=9,
           color='white')
    ax.text(7.75, y-0.35, 'by your text!', ha='center', fontsize=9,
           color='white')

    # Loop arrow
    ax.annotate('', xy=(6.2, y-0.8), xytext=(9.3, y-0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='white'))
    ax.annotate('', xy=(9.3, y+0.8), xytext=(6.2, y+0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='white'))
    ax.text(7.75, y-1.1, 'Repeat 50-100 times', ha='center', fontsize=8,
           style='italic', fontweight='bold')

    # Arrow to result
    ax.arrow(10, y, 1, 0, head_width=0.3, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Step 5: Final result
    result_box = FancyBboxPatch((11.3, y-0.6), 1.5, 1.2, boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor=COLORS['green'],
                               linewidth=3)
    ax.add_patch(result_box)
    ax.text(12.05, y, '🐱\n🚀', ha='center', va='center', fontsize=25)
    ax.text(12.05, y-1.1, '✨ Final Image!', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['green'])

    # Show the guidance process
    y = 3.8
    ax.text(7, y, '🎯 KEY INSIGHT: Text Guides the Denoising!', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['purple'],
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue',
                    edgecolor='black', linewidth=2))

    # Show examples of guidance
    y = 2.5
    guidance_examples = [
        ('Random noise →', '→ Could be anything', COLORS['red']),
        ('Noise + "cat" →', '→ Becomes cat-like', COLORS['orange']),
        ('Noise + "astronaut" →', '→ Adds space suit', COLORS['blue']),
        ('Noise + "in space" →', '→ Adds stars, planets', COLORS['purple']),
    ]

    for i, (before, after, color) in enumerate(guidance_examples):
        y_pos = y - i * 0.6
        ax.text(3, y_pos, before, ha='left', fontsize=9)
        ax.text(7, y_pos, after, ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                        edgecolor='black', alpha=0.5))

    # Bottom explanation
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')

    explanation = """
    🎨 TEXT-TO-IMAGE IN SIMPLE TERMS:

    🤔 THE CHALLENGE:
    How do you teach a computer to draw what you describe with words?

    ✨ THE SOLUTION (Stable Diffusion, DALL-E, Midjourney):
    Combine THREE powerful components:

    1️⃣ TEXT ENCODER (Understanding):
       • Uses a language model (like ChatGPT's brain)
       • Reads your prompt: "A cat astronaut in space"
       • Understands: [cat features] + [space suit] + [space background]
       • Creates a "concept map" of what you want

    2️⃣ DIFFUSION MODEL (Drawing):
       • Starts with random noise
       • Gradually removes noise in many small steps
       • Each step: "What should this noisy image look like to match the text?"

    3️⃣ GUIDANCE (Steering):
       • The text "guides" the denoising process
       • At each step: "Move the image TOWARD matching the description"
       • Like a GPS guiding you to your destination!

    🎯 HOW THEY WORK TOGETHER:
    • Text encoder says: "This is what a cat astronaut should look like"
    • Diffusion model says: "I'll gradually create it from noise"
    • Guidance says: "Keep moving closer to that description with each step"

    💡 REAL-WORLD ANALOGY:
    Imagine sculpting clay blindfolded:
    • Someone describes what to sculpt (Text Encoder)
    • You make small changes to the clay (Diffusion Steps)
    • They guide you: "Add more here, less there" (Guidance)
    • After many adjustments, you've created what they described!

    🚀 WHY IT'S REVOLUTIONARY:
    ✅ Anyone can create art (no drawing skills needed!)
    ✅ Incredibly detailed and creative results
    ✅ Can mix concepts that don't exist: "robot playing piano underwater"
    ✅ Fast generation (seconds to minutes)

    ⚡ FUN FACT:
    These models were trained on BILLIONS of images from the internet,
    learning what things look like and how words connect to visuals!
    """

    ax2.text(0.5, 0.5, explanation, transform=ax2.transAxes,
            fontsize=10.5, verticalalignment='center', ha='center',
            family='monospace', bbox=dict(boxstyle='round,pad=1',
            facecolor='lightgreen', edgecolor='black', linewidth=2))

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/text_to_image_process.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated text_to_image_process.png")


def plot_diffusion_applications():
    """Show real-world applications of diffusion models"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    applications = [
        ("🎨 AI Art Creation", "Midjourney, DALL-E 3,\nStable Diffusion",
         COLORS['purple'], "Create stunning\nart from text!"),
        ("✏️ Image Editing", "Inpainting, outpainting,\nstyle transfer",
         COLORS['blue'], "Edit photos like\na pro designer!"),
        ("🎬 Video Generation", "Runway, Pika Labs,\nanimation tools",
         COLORS['red'], "Create videos\nfrom text!"),
        ("🏠 Architecture & Design", "Generate room layouts,\nbuilding designs",
         COLORS['green'], "Design spaces\ninstantly!"),
        ("🎮 Game Assets", "Generate textures,\ncharacters, environments",
         COLORS['orange'], "Create game art\nsuper fast!"),
        ("👔 Fashion & Product", "Design clothes,\nproduct mockups",
         COLORS['pink'], "Prototype designs\nin seconds!"),
    ]

    for ax, (title, examples, color, benefit) in zip(axes, applications):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Main box
        box = FancyBboxPatch((0.5, 2), 9, 6, boxstyle="round,pad=0.3",
                            facecolor=color, edgecolor='black',
                            linewidth=3, alpha=0.6)
        ax.add_patch(box)

        # Title
        ax.text(5, 7, title, ha='center', va='center', fontsize=12,
               fontweight='bold', color='white')

        # Examples
        ax.text(5, 5.5, examples, ha='center', va='center', fontsize=10,
               color='white', style='italic')

        # Benefit
        benefit_box = FancyBboxPatch((1.5, 3), 7, 1.2, boxstyle="round,pad=0.2",
                                    facecolor='white', edgecolor='black',
                                    linewidth=2, alpha=0.9)
        ax.add_patch(benefit_box)
        ax.text(5, 3.6, benefit, ha='center', va='center', fontsize=9,
               fontweight='bold', color=color)

        # Decorative elements
        ax.text(1, 7.5, '✨', fontsize=15)
        ax.text(9, 7.5, '✨', fontsize=15)

    plt.suptitle('Diffusion Models: Creating the Impossible! 🚀',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/diffusion_applications.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated diffusion_applications.png")


def plot_diffusion_vs_others():
    """Compare diffusion models with other generative approaches"""
    fig = plt.figure(figsize=(16, 10))

    plt.suptitle('Generative Models Comparison: Which One to Use? 🤔',
                fontsize=16, fontweight='bold', y=0.98)

    ax = plt.subplot(1, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # GANs
    y = 10.5
    gan_box = FancyBboxPatch((0.5, y-1.8), 4, 1.8, boxstyle="round,pad=0.2",
                            facecolor=COLORS['red'], edgecolor='black',
                            linewidth=3, alpha=0.6)
    ax.add_patch(gan_box)
    ax.text(2.5, y-0.3, '🎭 GANs', ha='center', fontsize=13,
           fontweight='bold', color='white')
    ax.text(2.5, y-0.7, '(Generator vs\nDiscriminator)', ha='center',
           fontsize=9, color='white')

    gan_text = """✅ Very realistic
✅ Fast generation
❌ Hard to train
❌ Mode collapse
🎯 Best for: Faces"""
    ax.text(2.5, y-1.4, gan_text, ha='center', va='top', fontsize=8,
           color='white', family='monospace')

    # VAEs
    vae_box = FancyBboxPatch((5, y-1.8), 4, 1.8, boxstyle="round,pad=0.2",
                            facecolor=COLORS['blue'], edgecolor='black',
                            linewidth=3, alpha=0.6)
    ax.add_patch(vae_box)
    ax.text(7, y-0.3, '🗜️ VAEs', ha='center', fontsize=13,
           fontweight='bold', color='white')
    ax.text(7, y-0.7, '(Encoder-Decoder)', ha='center', fontsize=9,
           color='white')

    vae_text = """✅ Stable training
✅ Smooth latent space
❌ Blurry images
❌ Less realistic
🎯 Best for: Compression"""
    ax.text(7, y-1.4, vae_text, ha='center', va='top', fontsize=8,
           color='white', family='monospace')

    # Diffusion
    diff_box = FancyBboxPatch((9.5, y-1.8), 4, 1.8, boxstyle="round,pad=0.2",
                             facecolor=COLORS['green'], edgecolor='black',
                             linewidth=3, alpha=0.6)
    ax.add_patch(diff_box)
    ax.text(11.5, y-0.3, '🌊 Diffusion', ha='center', fontsize=13,
           fontweight='bold', color='white')
    ax.text(11.5, y-0.7, '(Iterative Denoising)', ha='center', fontsize=9,
           color='white')

    diff_text = """✅ Highest quality!
✅ Stable training
✅ Text-controllable
❌ Slower generation
🎯 Best for: Art, images"""
    ax.text(11.5, y-1.4, diff_text, ha='center', va='top', fontsize=8,
           color='white', family='monospace')

    # Comparison table
    y = 6.5
    ax.text(7, y+0.5, '📊 DETAILED COMPARISON:', ha='center', fontsize=13,
           fontweight='bold')

    # Table headers
    headers = ['Feature', 'GANs', 'VAEs', 'Diffusion']
    x_positions = [1.5, 4.5, 7.5, 10.5]
    for i, header in enumerate(headers):
        ax.text(x_positions[i], y, header, ha='center', fontsize=11,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray',
                        edgecolor='black', linewidth=2))

    # Table rows
    rows = [
        ('Image Quality', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐'),
        ('Training Stability', '⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'),
        ('Generation Speed', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐'),
        ('Diversity', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'),
        ('Control (text)', '⭐⭐', '⭐⭐', '⭐⭐⭐⭐⭐'),
    ]

    y_row = y - 0.8
    for row in rows:
        for i, cell in enumerate(row):
            ax.text(x_positions[i], y_row, cell, ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='gray', linewidth=1))
        y_row -= 0.6

    # Bottom summary
    summary = """
    🎯 WHEN TO USE EACH:

    🎭 GANs: When you need fast generation and have lots of training data
         Example: Generating realistic faces, style transfer

    🗜️ VAEs: When you need smooth interpolation and stable training
         Example: Data compression, anomaly detection, simple generation

    🌊 DIFFUSION: When you need the highest quality and text control
         Example: AI art (DALL-E, Midjourney), image editing, creative applications

    🏆 CURRENT WINNER: Diffusion Models!
    They're the technology behind most modern AI art tools because they:
    • Generate the highest quality images
    • Work great with text prompts
    • Are stable and reliable to train
    • Can be controlled and edited easily

    💡 THE TREND:
    Diffusion models have largely replaced GANs for image generation!
    They're slower, but the quality difference is worth it for most applications.
    """

    ax.text(7, 1.5, summary, ha='center', va='center', fontsize=10,
           family='monospace', bbox=dict(boxstyle='round,pad=0.8',
           facecolor='lightyellow', edgecolor='black', linewidth=2))

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/diffusion_vs_others.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated diffusion_vs_others.png")


def main():
    """Generate all beginner-friendly Diffusion Models figures"""
    print("="*60)
    print("Generating Beginner-Friendly Diffusion Models Visualizations")
    print("="*60)

    plot_diffusion_concept()
    plot_text_to_image()
    plot_diffusion_applications()
    plot_diffusion_vs_others()

    print("="*60)
    print("✅ Diffusion Models figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
