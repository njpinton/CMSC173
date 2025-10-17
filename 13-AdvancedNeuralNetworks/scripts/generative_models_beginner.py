#!/usr/bin/env python3
"""
Generative Models - Beginner-Friendly Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

Simple, intuitive visualizations for understanding Generative Models.
Focus on concepts, not complex math!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, Wedge
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


def plot_generative_vs_discriminative():
    """Simple explanation of generative vs discriminative models"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Discriminative (Classifying)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Draw input images
    ax.text(5, 9, '📷 DISCRIMINATIVE MODELS', ha='center', fontsize=14,
           fontweight='bold', color=COLORS['blue'])
    ax.text(5, 8.2, '(The Classifier - "What is this?")', ha='center',
           fontsize=11, style='italic')

    # Show example images
    y_start = 6.5
    examples = [
        ("🐶", "Is this a dog?", "YES! ✅"),
        ("🐱", "Is this a cat?", "YES! ✅"),
        ("🚗", "Is this a dog?", "NO! ❌"),
    ]

    for i, (emoji, question, answer) in enumerate(examples):
        y = y_start - i*1.8
        # Input box
        box1 = FancyBboxPatch((1, y-0.3), 1.5, 0.8, boxstyle="round,pad=0.1",
                             facecolor=COLORS['orange'], edgecolor='black',
                             linewidth=2, alpha=0.7)
        ax.add_patch(box1)
        ax.text(1.75, y, emoji, ha='center', va='center', fontsize=20)

        # Arrow
        ax.arrow(2.8, y, 1.2, 0, head_width=0.2, head_length=0.2,
                fc='black', ec='black', linewidth=2)

        # Question
        ax.text(5, y, question, ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

        # Answer
        answer_color = COLORS['green'] if "YES" in answer else COLORS['red']
        ax.text(7.5, y, answer, ha='center', va='center', fontsize=10,
               fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=answer_color))

    explanation1 = """
    🎯 TASK: Look at things
    and CLASSIFY them

    📚 Like a student taking
    a multiple choice test:
    "Which one is correct?"
    """
    ax.text(5, 1.5, explanation1, ha='center', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                    edgecolor='black', linewidth=2))

    # Right: Generative (Creating)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9, '🎨 GENERATIVE MODELS', ha='center', fontsize=14,
           fontweight='bold', color=COLORS['purple'])
    ax.text(5, 8.2, '(The Creator - "Make me something new!")', ha='center',
           fontsize=11, style='italic')

    # Show creation process
    requests = [
        ("Draw me a dog! 🐕", "🐶"),
        ("Draw me a cat! 🐈", "🐱"),
        ("Draw me a car! 🏎️", "🚗"),
    ]

    for i, (request, result) in enumerate(requests):
        y = y_start - i*1.8
        # Request
        ax.text(2.5, y, request, ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

        # Arrow
        ax.arrow(4.5, y, 1.2, 0, head_width=0.2, head_length=0.2,
                fc='black', ec='black', linewidth=2)

        # Created result
        box2 = FancyBboxPatch((6.5, y-0.3), 1.5, 0.8, boxstyle="round,pad=0.1",
                             facecolor=COLORS['green'], edgecolor='black',
                             linewidth=2, alpha=0.7)
        ax.add_patch(box2)
        ax.text(7.25, y, result, ha='center', va='center', fontsize=20)

        # Sparkles
        ax.text(8.2, y+0.1, '✨', fontsize=12)

    explanation2 = """
    🎨 TASK: CREATE new
    things from scratch!

    👨‍🎨 Like an artist:
    "I'll paint you something
    that has never existed!"
    """
    ax.text(5, 1.5, explanation2, ha='center', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                    edgecolor='black', linewidth=2))

    plt.suptitle('Generative vs Discriminative: Creating vs Classifying 🎨🔍',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/generative_vs_discriminative.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated generative_vs_discriminative.png")


def plot_gan_concept():
    """Simple GAN concept: Generator vs Discriminator like artist vs critic"""
    fig = plt.figure(figsize=(16, 10))

    # Main title
    plt.suptitle('GANs: The Artist vs The Critic Game 🎨⚔️🔍',
                fontsize=16, fontweight='bold', y=0.98)

    # Create the game setup
    ax = plt.subplot(2, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Generator (Artist)
    gen_box = FancyBboxPatch((1, 4.5), 3, 2.5, boxstyle="round,pad=0.2",
                            facecolor=COLORS['purple'], edgecolor='black',
                            linewidth=3, alpha=0.7)
    ax.add_patch(gen_box)
    ax.text(2.5, 6.2, '🎨 GENERATOR', ha='center', fontsize=12,
           fontweight='bold', color='white')
    ax.text(2.5, 5.5, '"The Artist"', ha='center', fontsize=10,
           color='white', style='italic')
    ax.text(2.5, 4.9, 'Creates fake\nimages', ha='center', fontsize=9,
           color='white')

    # Random noise input
    noise_box = FancyBboxPatch((0.5, 1.5), 1, 1, boxstyle="round,pad=0.1",
                              facecolor=COLORS['yellow'], edgecolor='black',
                              linewidth=2)
    ax.add_patch(noise_box)
    ax.text(1, 2, '🎲\nRandom\nNoise', ha='center', va='center', fontsize=8)
    ax.arrow(1, 2.6, 1, 1.5, head_width=0.3, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Generated image
    gen_img = FancyBboxPatch((4.8, 5.2), 1.2, 1.5, boxstyle="round,pad=0.1",
                            facecolor='white', edgecolor=COLORS['purple'],
                            linewidth=3)
    ax.add_patch(gen_img)
    ax.text(5.4, 6, '🖼️', fontsize=25, ha='center', va='center')
    ax.text(5.4, 4.8, 'Fake\nImage', ha='center', fontsize=8,
           fontweight='bold', color=COLORS['purple'])

    # Arrow to discriminator
    ax.arrow(6.2, 5.9, 1.5, 0, head_width=0.3, head_length=0.3,
            fc=COLORS['purple'], ec=COLORS['purple'], linewidth=3)

    # Real image
    real_img = FancyBboxPatch((4.8, 1.5), 1.2, 1.5, boxstyle="round,pad=0.1",
                             facecolor='white', edgecolor=COLORS['green'],
                             linewidth=3)
    ax.add_patch(real_img)
    ax.text(5.4, 2.3, '📷', fontsize=25, ha='center', va='center')
    ax.text(5.4, 1.1, 'Real\nImage', ha='center', fontsize=8,
           fontweight='bold', color=COLORS['green'])

    # Arrow to discriminator
    ax.arrow(6.2, 2.2, 1.5, 0.5, head_width=0.3, head_length=0.3,
            fc=COLORS['green'], ec=COLORS['green'], linewidth=3)

    # Discriminator (Critic)
    disc_box = FancyBboxPatch((8, 3), 3, 3, boxstyle="round,pad=0.2",
                             facecolor=COLORS['blue'], edgecolor='black',
                             linewidth=3, alpha=0.7)
    ax.add_patch(disc_box)
    ax.text(9.5, 5.3, '🔍 DISCRIMINATOR', ha='center', fontsize=12,
           fontweight='bold', color='white')
    ax.text(9.5, 4.7, '"The Critic"', ha='center', fontsize=10,
           color='white', style='italic')
    ax.text(9.5, 4, 'Judges:\nReal or Fake?', ha='center', fontsize=9,
           color='white')

    # Output verdicts
    verdict_y = 4.5
    for verdict, color, emoji in [("REAL! ✅", COLORS['green'], "📷"),
                                   ("FAKE! ❌", COLORS['red'], "🎨")]:
        verdict_box = FancyBboxPatch((11.5, verdict_y-0.3), 2, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black',
                                    linewidth=2, alpha=0.8)
        ax.add_patch(verdict_box)
        ax.text(12.5, verdict_y, verdict, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
        verdict_y -= 1.2

    # Feedback loop
    ax.annotate('', xy=(2.5, 4.3), xytext=(9.5, 2.8),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['red'],
                             connectionstyle='arc3,rad=-.3'))
    ax.text(6, 2, '💬 "Try harder!\nI caught you!"',
           ha='center', fontsize=9, fontweight='bold', color=COLORS['red'],
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    # Bottom explanation
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')

    explanation = """
    🎮 HOW THE "GAME" WORKS:

    🎨 THE GENERATOR (Artist):
    • Starts with random noise (like random brushstrokes)
    • Creates fake images trying to look real
    • Goal: "Fool the critic into thinking my art is real!"

    🔍 THE DISCRIMINATOR (Critic):
    • Looks at both real images and fake images
    • Tries to tell them apart
    • Goal: "Spot the fakes! Don't be fooled!"

    🔄 THE COMPETITION:
    1️⃣ Generator creates a fake image
    2️⃣ Discriminator judges: "Real or Fake?"
    3️⃣ If caught: Generator learns from mistakes and improves
    4️⃣ Discriminator also gets better at detecting fakes
    5️⃣ They keep competing until Generator is SO GOOD, even the expert Discriminator can't tell!

    🏆 THE RESULT:
    After thousands of rounds, the Generator becomes an AMAZING artist!
    It can create super realistic images that never existed before!

    💡 REAL-WORLD ANALOGY:
    Think of it like a student artist (Generator) trying to forge a painting,
    while an art expert (Discriminator) tries to detect forgeries.
    The student keeps improving until they're as good as a master!

    ✨ Why GANs are MAGICAL:
    • They learn to create realistic images WITHOUT being told exactly how
    • The competition makes both networks better and better
    • No human has to teach them what "realistic" means - they figure it out!
    """

    ax2.text(0.5, 0.5, explanation, transform=ax2.transAxes,
            fontsize=10.5, verticalalignment='center', ha='center',
            family='monospace', bbox=dict(boxstyle='round,pad=1',
            facecolor='lightyellow', edgecolor='black', linewidth=2))

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/gan_simple_concept.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated gan_simple_concept.png")


def plot_vae_concept():
    """Simple VAE concept: compression and generation"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Top: VAE architecture
    ax = axes[0]
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, '🗜️ VAE: The Compression Artist', ha='center',
           fontsize=14, fontweight='bold')

    # Input image
    input_box = FancyBboxPatch((1, 3.5), 1.5, 2, boxstyle="round,pad=0.1",
                              facecolor='white', edgecolor=COLORS['blue'],
                              linewidth=3)
    ax.add_patch(input_box)
    ax.text(1.75, 4.5, '🖼️', fontsize=30, ha='center', va='center')
    ax.text(1.75, 2.8, 'Input\nImage', ha='center', fontsize=9,
           fontweight='bold')

    # Arrow to encoder
    ax.arrow(2.7, 4.5, 1, 0, head_width=0.3, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Encoder (compression)
    encoder_box = FancyBboxPatch((4, 3.5), 1.8, 2, boxstyle="round,pad=0.2",
                                facecolor=COLORS['orange'], edgecolor='black',
                                linewidth=3, alpha=0.7)
    ax.add_patch(encoder_box)
    ax.text(4.9, 5, '📦 ENCODER', ha='center', fontsize=11,
           fontweight='bold', color='white')
    ax.text(4.9, 4.2, '"Compress"', ha='center', fontsize=9, color='white')
    ax.text(4.9, 3.8, 'Squeeze into\nsmall code', ha='center', fontsize=8,
           color='white')

    # Arrow to latent space
    ax.arrow(6, 4.5, 0.8, 0, head_width=0.3, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Latent space (compressed representation)
    latent_circle = Circle((7.5, 4.5), 0.8, facecolor=COLORS['purple'],
                          edgecolor='black', linewidth=3, alpha=0.7)
    ax.add_patch(latent_circle)
    ax.text(7.5, 4.9, '💫', fontsize=20, ha='center', va='center')
    ax.text(7.5, 4.2, 'Secret\nCode', ha='center', fontsize=8,
           color='white', fontweight='bold')
    ax.text(7.5, 2.5, '(Tiny compressed version)', ha='center', fontsize=8,
           style='italic')

    # Arrow to decoder
    ax.arrow(8.5, 4.5, 0.8, 0, head_width=0.3, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Decoder (decompression)
    decoder_box = FancyBboxPatch((9.5, 3.5), 1.8, 2, boxstyle="round,pad=0.2",
                                facecolor=COLORS['green'], edgecolor='black',
                                linewidth=3, alpha=0.7)
    ax.add_patch(decoder_box)
    ax.text(10.4, 5, '📂 DECODER', ha='center', fontsize=11,
           fontweight='bold', color='white')
    ax.text(10.4, 4.2, '"Decompress"', ha='center', fontsize=9, color='white')
    ax.text(10.4, 3.8, 'Recreate the\nimage', ha='center', fontsize=8,
           color='white')

    # Arrow to output
    ax.arrow(11.5, 4.5, 1, 0, head_width=0.3, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Output image
    output_box = FancyBboxPatch((12.7, 3.5), 1.5, 2, boxstyle="round,pad=0.1",
                               facecolor='white', edgecolor=COLORS['green'],
                               linewidth=3)
    ax.add_patch(output_box)
    ax.text(13.45, 4.5, '🖼️', fontsize=30, ha='center', va='center')
    ax.text(13.45, 2.8, 'Recreated\nImage', ha='center', fontsize=9,
           fontweight='bold')
    ax.text(13.45, 2.3, '✨', fontsize=15, ha='center')

    # Bottom: Explanation
    ax = axes[1]
    ax.axis('off')

    explanation = """
    🎯 VAE IN SIMPLE TERMS (Variational Autoencoder):

    💭 THINK OF IT LIKE ZIP FILES:
    When you zip a large file, it gets compressed into something smaller.
    VAEs do the same thing with images, but in a SMART way!

    📦 HOW IT WORKS:

    1️⃣ ENCODER (The Compressor):
       • Takes a big image (like a photo of a cat)
       • Squeezes it into a tiny "secret code" (just a few numbers!)
       • Like compressing a photo to save space on your phone

    2️⃣ LATENT SPACE (The Secret Code):
       • The compressed representation
       • Contains the ESSENCE of the image (not every pixel, just the important info)
       • Like a recipe card instead of the full meal

    3️⃣ DECODER (The Decompressor):
       • Takes the secret code
       • Recreates the image from memory
       • Like unzipping a file, but it GENERATES the image

    🎨 THE MAGIC TRICK:
    Once trained, you can:
    ✨ Create NEW images by giving it NEW random codes!
    ✨ Mix codes to blend images (cat + dog = ???)
    ✨ Change codes slightly to modify images

    💡 REAL-WORLD ANALOGY:
    Imagine describing a face with just a few words: "round face, blue eyes, blonde hair, smiling"
    From just that SHORT description, someone could draw a complete face!
    That's what VAE's secret code does!

    🆚 VAE vs GAN:
    • VAE: More stable, creates smooth variations, great for compression
    • GAN: More realistic, but training is trickier (needs the competition)

    ✅ USES:
    • Generating new faces, artwork, designs
    • Image compression (better than JPEG!)
    • Anomaly detection (if compression fails, something's weird!)
    • Creating variations of existing images
    """

    ax.text(0.5, 0.5, explanation, transform=ax.transAxes,
           fontsize=10.5, verticalalignment='center', ha='center',
           family='monospace', bbox=dict(boxstyle='round,pad=1',
           facecolor='lightblue', edgecolor='black', linewidth=2))

    plt.suptitle('VAE: Learning to Compress and Create 🗜️✨',
                fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/vae_simple_concept.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated vae_simple_concept.png")


def plot_generative_applications():
    """Show real-world applications of generative models"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    applications = [
        ("🎨 AI Art Generation", "Midjourney, DALL-E,\nStable Diffusion",
         COLORS['purple'], "Create amazing\nart from text!"),
        ("🎬 Deepfakes & Videos", "Face swapping,\nde-aging actors",
         COLORS['red'], "Realistic video\nmanipulation"),
        ("👤 Face Generation", "This Person Does\nNot Exist",
         COLORS['blue'], "Create fake but\nrealistic faces"),
        ("🎵 Music Generation", "AI composers,\nstyle transfer",
         COLORS['green'], "Create new songs\nin any style!"),
        ("💊 Drug Discovery", "Generate new\nmolecule designs",
         COLORS['orange'], "Design new\nmedicines faster"),
        ("🎮 Game Content", "Generate textures,\nlevels, characters",
         COLORS['teal'], "Create infinite\ngame content!"),
    ]

    for ax, (title, desc, color, benefit) in zip(axes, applications):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Main box
        box = FancyBboxPatch((0.5, 2.5), 9, 5, boxstyle="round,pad=0.3",
                            facecolor=color, edgecolor='black',
                            linewidth=3, alpha=0.6)
        ax.add_patch(box)

        # Title with emoji
        ax.text(5, 6.5, title, ha='center', va='center', fontsize=13,
               fontweight='bold', color='white')

        # Description
        ax.text(5, 5, desc, ha='center', va='center', fontsize=10,
               color='white', style='italic')

        # Benefit box
        benefit_box = FancyBboxPatch((1.5, 3), 7, 1, boxstyle="round,pad=0.2",
                                    facecolor='white', edgecolor='black',
                                    linewidth=2, alpha=0.9)
        ax.add_patch(benefit_box)
        ax.text(5, 3.5, benefit, ha='center', va='center', fontsize=9,
               fontweight='bold', color=color)

        # Sparkles
        ax.text(1, 7, '✨', fontsize=15)
        ax.text(9, 7, '✨', fontsize=15)

    plt.suptitle('Generative Models: Creating the Future! 🚀',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/generative_applications.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated generative_applications.png")


def plot_ethical_considerations():
    """Visualize ethical considerations of generative models"""
    fig = plt.figure(figsize=(16, 10))

    ax = plt.subplot(1, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Title
    ax.text(7, 13.5, '⚖️ Ethics: The Responsible Use of AI Generation',
           ha='center', fontsize=15, fontweight='bold')

    # Good uses (left side)
    good_y = 11.5
    ax.text(3.5, good_y, '✅ POSITIVE USES', ha='center', fontsize=13,
           fontweight='bold', color=COLORS['green'],
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen',
                    edgecolor='black', linewidth=2))

    good_uses = [
        ("🎨 Creative Expression", "Artists using AI\nas a tool"),
        ("🏥 Medical Research", "Generating synthetic\nmedical data"),
        ("🎓 Education", "Creating learning\nmaterials"),
        ("♿ Accessibility", "Helping people with\ndisabilities"),
        ("🔬 Scientific Discovery", "Simulating molecules,\nproteins"),
    ]

    y_pos = good_y - 1.5
    for emoji_title, desc in good_uses:
        box = FancyBboxPatch((0.5, y_pos-0.5), 6, 0.9, boxstyle="round,pad=0.1",
                            facecolor=COLORS['green'], edgecolor='black',
                            linewidth=2, alpha=0.3)
        ax.add_patch(box)
        ax.text(1, y_pos, emoji_title, ha='left', va='center', fontsize=10,
               fontweight='bold')
        ax.text(5.5, y_pos, desc, ha='right', va='center', fontsize=9,
               style='italic')
        y_pos -= 1.2

    # Concerns (right side)
    concern_y = 11.5
    ax.text(10.5, concern_y, '⚠️ CONCERNS & RISKS', ha='center', fontsize=13,
           fontweight='bold', color=COLORS['red'],
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral',
                    edgecolor='black', linewidth=2))

    concerns = [
        ("🎭 Deepfakes", "Fake videos spreading\nmisinformation"),
        ("🎨 Copyright Issues", "Who owns AI-generated\nart?"),
        ("💼 Job Displacement", "Will AI replace\nhuman artists?"),
        ("🔒 Privacy Violations", "Generating fake images\nof real people"),
        ("📰 Fake News", "Creating fake but\nrealistic content"),
    ]

    y_pos = concern_y - 1.5
    for emoji_title, desc in concerns:
        box = FancyBboxPatch((7.5, y_pos-0.5), 6, 0.9, boxstyle="round,pad=0.1",
                            facecolor=COLORS['red'], edgecolor='black',
                            linewidth=2, alpha=0.3)
        ax.add_patch(box)
        ax.text(8, y_pos, emoji_title, ha='left', va='center', fontsize=10,
               fontweight='bold')
        ax.text(12.5, y_pos, desc, ha='right', va='center', fontsize=9,
               style='italic')
        y_pos -= 1.2

    # Bottom: Best practices
    best_practices = """
    🛡️ HOW TO USE GENERATIVE AI RESPONSIBLY:

    1️⃣ BE TRANSPARENT: Always disclose when content is AI-generated
       Example: Label AI art as "Created with AI" or "AI-assisted"

    2️⃣ RESPECT CONSENT: Don't create fake images/videos of real people without permission
       Example: Don't deepfake someone's face onto inappropriate content

    3️⃣ VERIFY SOURCES: Don't believe everything you see online
       Example: Check if that viral video might be AI-generated

    4️⃣ CREDIT PROPERLY: Acknowledge both the AI tool AND human creativity
       Example: "Concept by [Artist], generated with [AI Tool]"

    5️⃣ USE FOR GOOD: Think about the impact of what you create
       Example: Use AI to help people, not to deceive or harm them

    💭 REMEMBER: Just because we CAN create something doesn't mean we SHOULD!

    🌟 THE GOAL: Use AI as a tool to enhance human creativity and solve problems,
                 not to deceive, harm, or replace human connection.
    """

    ax.text(7, 2, best_practices, ha='center', va='center', fontsize=10,
           family='monospace', bbox=dict(boxstyle='round,pad=0.8',
           facecolor='lightyellow', edgecolor='black', linewidth=2))

    plt.tight_layout()

    output_dir = create_output_dir()
    plt.savefig(f"{output_dir}/generative_ethics.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated generative_ethics.png")


def main():
    """Generate all beginner-friendly Generative Models figures"""
    print("="*60)
    print("Generating Beginner-Friendly Generative Models Visualizations")
    print("="*60)

    plot_generative_vs_discriminative()
    plot_gan_concept()
    plot_vae_concept()
    plot_generative_applications()
    plot_ethical_considerations()

    print("="*60)
    print("✅ Generative Models figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
