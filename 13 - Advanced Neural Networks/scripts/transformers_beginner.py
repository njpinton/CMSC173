#!/usr/bin/env python3
"""
Transformers - Beginner-Friendly Visualizations
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

Simple, intuitive visualizations for understanding Transformers.
Focus on concepts, not complex math!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, Wedge, Arc
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


def plot_attention_mechanism(output_dir: str):
    """Simple explanation of attention - focusing on important words"""
    fig = plt.figure(figsize=(16, 10))

    plt.suptitle('Attention Mechanism: Focusing on What Matters 🔍',
                fontsize=16, fontweight='bold', y=0.98)

    # Top: Example without attention
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 6)
    ax1.axis('off')

    ax1.text(7, 5.5, '❌ OLD WAY: Reading Every Word Equally', ha='center',
            fontsize=13, fontweight='bold', color=COLORS['red'])

    sentence = "The cat sat on the mat"
    words = sentence.split()
    x_start = 2
    spacing = 1.5

    for i, word in enumerate(words):
        x = x_start + i * spacing
        # All words get equal attention (same size/color)
        box = FancyBboxPatch((x-0.4, 3.5), 0.8, 0.8, boxstyle="round,pad=0.1",
                            facecolor=COLORS['blue'], edgecolor='black',
                            linewidth=2, alpha=0.5)
        ax1.add_patch(box)
        ax1.text(x, 3.9, word, ha='center', va='center', fontsize=9,
                color='white', fontweight='bold')

    ax1.text(7, 2.5, 'Problem: Computer treats ALL words as equally important! 😕',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral',
                     edgecolor='black', linewidth=2))

    ax1.text(7, 1.3, 'Question: "What sat on the mat?" → Hard to know "cat" is the answer!',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    # Bottom: Example with attention
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 6)
    ax2.axis('off')

    ax2.text(7, 5.5, '✅ NEW WAY: Attention! Focus on Important Words', ha='center',
            fontsize=13, fontweight='bold', color=COLORS['green'])

    # Show attention weights with different sizes and colors
    attention_weights = [0.1, 0.8, 0.3, 0.1, 0.1, 0.4]  # "cat" and "mat" are important
    max_weight = max(attention_weights)

    for i, (word, weight) in enumerate(zip(words, attention_weights)):
        x = x_start + i * spacing
        size = 0.4 + (weight / max_weight) * 0.8  # Scale box size
        alpha = 0.3 + (weight / max_weight) * 0.7  # Scale opacity

        # Draw box with size proportional to attention
        box = FancyBboxPatch((x-size/2, 3.5-size/2), size, size,
                            boxstyle="round,pad=0.1",
                            facecolor=COLORS['orange'], edgecolor='black',
                            linewidth=3 if weight > 0.5 else 2, alpha=alpha)
        ax2.add_patch(box)
        ax2.text(x, 3.5, word, ha='center', va='center',
                fontsize=8 + int(weight * 6), color='white', fontweight='bold')

        # Show attention score
        ax2.text(x, 2.7, f'{int(weight*100)}%', ha='center', fontsize=8,
                style='italic')

    # Spotlight effect on "cat"
    spotlight = Circle((x_start + 1 * spacing, 3.5), 0.8,
                      facecolor='none', edgecolor=COLORS['yellow'],
                      linewidth=4, linestyle='--')
    ax2.add_patch(spotlight)
    ax2.text(x_start + 1 * spacing, 4.6, '🔦', fontsize=20, ha='center')

    ax2.text(7, 1.8, '✨ Attention lets the model FOCUS on important words!',
            ha='center', fontsize=11, style='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen',
                     edgecolor='black', linewidth=2))

    ax2.text(7, 0.6, 'Question: "What sat on the mat?" → Attention highlights "cat"! 🎯',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    # Add explanation box
    explanation = """
    💡 ATTENTION IN SIMPLE TERMS:

    When you read "The cat sat on the mat", you naturally focus more on "cat" and "mat"
    because they're the important nouns. Words like "the" and "on" are less important.

    ATTENTION teaches AI to do the same thing - focus on what matters! 🎯
    """

    fig.text(0.5, 0.02, explanation, ha='center', fontsize=10,
            family='monospace', bbox=dict(boxstyle='round,pad=0.6',
            facecolor='lightblue', edgecolor='black', linewidth=2))

    plt.tight_layout()

    plt.savefig(f"{output_dir}/attention_simple_concept.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated attention_simple_concept.png")


def plot_transformer_architecture(output_dir: str):
    """Simple transformer concept - how ChatGPT works"""
    fig = plt.figure(figsize=(16, 10))

    plt.suptitle('Transformers: How ChatGPT Understands Language 🤖💬',
                fontsize=16, fontweight='bold', y=0.98)

    # Main diagram
    ax = plt.subplot(2, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Input
    input_text = "How are you?"
    ax.text(7, 9.5, f'💬 Input: "{input_text}"', ha='center', fontsize=12,
           fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue',
                    edgecolor='black', linewidth=2))

    # Step 1: Words to numbers
    words = input_text.replace("?", "").split()
    y = 8
    ax.text(2, y+0.5, '1️⃣ Turn words into numbers:', ha='left', fontsize=11,
           fontweight='bold')
    for i, word in enumerate(words):
        x = 3 + i * 1.5
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                            facecolor=COLORS['blue'], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, word, ha='center', va='center', fontsize=9, color='white')
        ax.text(x, y-0.7, f'#{i+1}', ha='center', fontsize=8, style='italic')

    # Arrow down
    ax.arrow(7, 7.2, 0, -0.5, head_width=0.4, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Step 2: Self-Attention layers
    y = 6
    attention_box = FancyBboxPatch((3, y-0.8), 8, 1.6, boxstyle="round,pad=0.2",
                                  facecolor=COLORS['purple'], edgecolor='black',
                                  linewidth=3, alpha=0.7)
    ax.add_patch(attention_box)
    ax.text(7, y+0.4, '2️⃣ SELF-ATTENTION LAYERS', ha='center', fontsize=11,
           fontweight='bold', color='white')
    ax.text(7, y-0.2, '"Which words are related?"', ha='center', fontsize=9,
           color='white', style='italic')

    # Show connections between words
    word_positions = [(4, y), (5.5, y), (7, y)]
    for i, (x1, y1) in enumerate(word_positions):
        for j, (x2, y2) in enumerate(word_positions):
            if i != j:
                # Draw connection with varying thickness
                ax.plot([x1, x2], [y1, y2], 'w-', linewidth=1, alpha=0.5)

    # Arrow down
    ax.arrow(7, 5, 0, -0.5, head_width=0.4, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Step 3: Feed-Forward layers
    y = 3.8
    ff_box = FancyBboxPatch((3.5, y-0.6), 7, 1.2, boxstyle="round,pad=0.2",
                           facecolor=COLORS['orange'], edgecolor='black',
                           linewidth=3, alpha=0.7)
    ax.add_patch(ff_box)
    ax.text(7, y+0.2, '3️⃣ PROCESSING LAYERS', ha='center', fontsize=11,
           fontweight='bold', color='white')
    ax.text(7, y-0.3, '"Understanding the meaning"', ha='center', fontsize=9,
           color='white', style='italic')

    # Arrow down
    ax.arrow(7, 3, 0, -0.5, head_width=0.4, head_length=0.2,
            fc='black', ec='black', linewidth=2)

    # Step 4: Output
    y = 2
    output_box = FancyBboxPatch((4, y-0.5), 6, 1, boxstyle="round,pad=0.2",
                               facecolor=COLORS['green'], edgecolor='black',
                               linewidth=3, alpha=0.7)
    ax.add_patch(output_box)
    ax.text(7, y, '4️⃣ Generate Response: "I\'m doing great!"', ha='center',
           fontsize=11, fontweight='bold', color='white')

    # Add sparkles
    ax.text(10.5, y, '✨', fontsize=20)
    ax.text(3.5, y, '✨', fontsize=20)

    # Bottom explanation
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')

    explanation = """
    🎯 TRANSFORMER IN SIMPLE TERMS:

    💭 OLD APPROACH (RNNs): Read sentence word-by-word, left to right
       • Like reading a book one word at a time
       • Slow and forgets earlier words
       • Example: By the time you read "dog", you forgot the "big brown" at the start

    ✨ TRANSFORMER APPROACH: Look at ALL words at once!
       • Like looking at the whole sentence together
       • Uses ATTENTION to see which words are related
       • Fast and remembers everything

    🔍 THE MAGIC - SELF-ATTENTION:
       When processing the word "it", the transformer looks at the whole sentence:
       "The cat sat on the mat because it was tired"
       → Attention figures out "it" refers to "cat", not "mat"!

    🧠 HOW CHATGPT USES THIS:
       1️⃣ You type: "Translate 'hello' to Spanish"
       2️⃣ Transformer reads ALL words simultaneously
       3️⃣ Attention connects "translate" with "hello" and "Spanish"
       4️⃣ It understands: [ACTION: translate] [FROM: hello] [TO: Spanish]
       5️⃣ Generates: "Hola"

    🚀 WHY TRANSFORMERS CHANGED AI:
       ✅ Can be trained on MASSIVE amounts of text (entire internet!)
       ✅ Understands context better (knows "bank" = money vs river)
       ✅ Much faster than old methods (processes sentences in parallel)
       ✅ Can handle very long text (remember things from paragraphs ago)

    💡 KEY INSIGHT:
       Transformers are like having a super-smart friend who can:
       • Read an entire book in seconds
       • Remember every detail
       • Understand how everything connects
       • Explain it back to you in simple terms
    """

    ax2.text(0.5, 0.5, explanation, transform=ax2.transAxes,
            fontsize=10.5, verticalalignment='center', ha='center',
            family='monospace', bbox=dict(boxstyle='round,pad=1',
            facecolor='lightyellow', edgecolor='black', linewidth=2))

    plt.tight_layout()

    plt.savefig(f"{output_dir}/transformer_simple_architecture.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated transformer_simple_architecture.png")


def plot_transformer_applications(output_dir: str):
    """Show real-world transformer applications"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    applications = [
        ("🗣️ Language Translation", "Google Translate\nDeepL",
         COLORS['blue'], "Translate between\n100+ languages!"),
        ("💬 Chatbots", "ChatGPT, Claude,\nGemini", COLORS['green'],
         "Have natural\nconversations!"),
        ("📝 Writing Assistants", "Grammarly, Notion AI,\nWriting tools",
         COLORS['purple'], "Help write better\nemails & docs!"),
        ("💻 Code Generation", "GitHub Copilot,\nChatGPT coding",
         COLORS['orange'], "Write code from\ndescriptions!"),
        ("📖 Text Summarization", "Summarize articles,\npapers, books",
         COLORS['teal'], "TLDR: Get the\nmain points fast!"),
        ("❓ Question Answering", "Search engines,\nVirtual assistants",
         COLORS['pink'], "Answer questions\nfrom documents!"),
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

        # Stars
        ax.text(1.2, 7.5, '⭐', fontsize=12)
        ax.text(8.8, 7.5, '⭐', fontsize=12)

    plt.suptitle('Transformers in Daily Life: You Use These Every Day! 🌟',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/transformer_applications.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated transformer_applications.png")


def plot_vision_transformers(output_dir: str):
    """Simple vision transformer concept"""
    fig = plt.figure(figsize=(16, 10))

    plt.suptitle('Vision Transformers: Teaching Language Models to See! 👁️',
                fontsize=16, fontweight='bold', y=0.98)

    # Main diagram
    ax = plt.subplot(2, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, '🎯 THE BIG IDEA: Treat images like sentences!',
           ha='center', fontsize=13, fontweight='bold')

    # Original approach
    y = 6
    ax.text(2, y, '📝 Text:', ha='left', fontsize=11, fontweight='bold')
    words_boxes = ["The", "cat", "sat"]
    x_start = 3.5
    for i, word in enumerate(words_boxes):
        x = x_start + i * 1.2
        box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                            facecolor=COLORS['blue'], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, word, ha='center', va='center', fontsize=9, color='white')
    ax.text(8.5, y, '← Words are like "tokens"', ha='left', fontsize=9,
           style='italic')

    # Vision transformer approach
    y = 4.5
    ax.text(2, y, '🖼️ Image:', ha='left', fontsize=11, fontweight='bold')

    # Draw a simple image divided into patches
    img_size = 1.8
    img_box = Rectangle((3.3, y-0.9), img_size, img_size,
                        facecolor='lightblue', edgecolor='black', linewidth=3)
    ax.add_patch(img_box)

    # Divide into patches
    patch_size = img_size / 3
    for i in range(3):
        for j in range(3):
            x = 3.3 + j * patch_size
            y_coord = y - 0.9 + i * patch_size
            patch = Rectangle((x, y_coord), patch_size, patch_size,
                            facecolor='none', edgecolor='black', linewidth=1.5)
            ax.add_patch(patch)
            # Add pattern to show it's divided
            ax.text(x + patch_size/2, y_coord + patch_size/2,
                   f'{i*3+j+1}', ha='center', va='center', fontsize=8,
                   fontweight='bold', color=COLORS['purple'])

    ax.text(5.5, y, '→', ha='center', fontsize=20)

    # Show patches as tokens
    patch_labels = ["P1", "P2", "P3", "..."]
    x_start = 6.5
    for i, label in enumerate(patch_labels):
        x = x_start + i * 0.9
        box = FancyBboxPatch((x-0.3, y-0.3), 0.6, 0.6, boxstyle="round,pad=0.05",
                            facecolor=COLORS['orange'], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
               color='white', fontweight='bold')
    ax.text(10.5, y, '← Image "patches" like words!', ha='left', fontsize=9,
           style='italic')

    # Processing
    y = 2.8
    ax.arrow(7, y+0.5, 0, -0.3, head_width=0.4, head_length=0.15,
            fc='black', ec='black', linewidth=2)

    transformer_box = FancyBboxPatch((4.5, y-0.8), 5, 1, boxstyle="round,pad=0.2",
                                    facecolor=COLORS['purple'], edgecolor='black',
                                    linewidth=3, alpha=0.7)
    ax.add_patch(transformer_box)
    ax.text(7, y-0.3, '🤖 Same Transformer Architecture!', ha='center',
           fontsize=11, fontweight='bold', color='white')
    ax.text(7, y-0.65, '(Attention between patches)', ha='center', fontsize=9,
           color='white', style='italic')

    # Output
    y = 1
    ax.arrow(7, y+0.5, 0, -0.3, head_width=0.4, head_length=0.15,
            fc='black', ec='black', linewidth=2)

    output_box = FancyBboxPatch((5, y-0.5), 4, 0.7, boxstyle="round,pad=0.15",
                               facecolor=COLORS['green'], edgecolor='black',
                               linewidth=3, alpha=0.7)
    ax.add_patch(output_box)
    ax.text(7, y-0.15, 'Result: "This is a cat!" 🐱', ha='center',
           fontsize=11, fontweight='bold', color='white')

    # Bottom explanation
    ax2 = plt.subplot(2, 1, 2)
    ax2.axis('off')

    explanation = """
    🎨 VISION TRANSFORMERS (ViT) IN SIMPLE TERMS:

    🤔 THE CLEVER TRICK:
    Instead of inventing a NEW architecture for images, we realized:
    "Why not use the SAME transformer that works great for text?"

    📐 HOW IT WORKS:
    1️⃣ Cut image into small squares (patches) - like cutting a pizza into slices
    2️⃣ Flatten each patch into a list of numbers (treat it like a "word")
    3️⃣ Feed these "image words" into a regular transformer
    4️⃣ Let attention figure out which patches are related!

    💡 EXAMPLE:
    If you have a photo of a cat:
    • Patch 1: Contains part of the ear
    • Patch 2: Contains the eye
    • Patch 3: Contains whiskers
    → Attention learns: "These patches together = cat face!"

    🆚 ViT vs CNN:
    • CNN: Looks at local neighborhoods (like looking through a magnifying glass)
    • ViT: Looks at the whole image at once (like seeing the big picture)
    • ViT: Can find connections between far-apart parts of the image!

    🌟 WHY IT'S AWESOME:
    ✅ Simpler architecture than CNNs (one model for everything!)
    ✅ Can handle huge images (just use more patches)
    ✅ Learns better from LOTS of data
    ✅ Same code works for images, text, video, audio!

    🔮 THE FUTURE:
    Multi-modal models like GPT-4, Claude use this idea:
    → They can understand BOTH text AND images together!
    → Ask: "What's in this photo?" and show a picture
    → AI can answer because it treats images and text the same way!
    """

    ax2.text(0.5, 0.5, explanation, transform=ax2.transAxes,
            fontsize=10.5, verticalalignment='center', ha='center',
            family='monospace', bbox=dict(boxstyle='round,pad=1',
            facecolor='lightgreen', edgecolor='black', linewidth=2))

    plt.tight_layout()

    plt.savefig(f"{output_dir}/vision_transformer_concept.png", dpi=200,
               bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated vision_transformer_concept.png")


def main():
    """Generate all beginner-friendly Transformer figures"""
    output_dir = create_output_dir()

    print("="*60)
    print("Generating Beginner-Friendly Transformer Visualizations")
    print("="*60)

    plot_attention_mechanism(output_dir)
    plot_transformer_architecture(output_dir)
    plot_transformer_applications(output_dir)
    plot_vision_transformers(output_dir)

    print("="*60)
    print("✅ Transformer figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
