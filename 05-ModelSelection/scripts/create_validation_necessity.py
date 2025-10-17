"""
Create a compelling visualization showing WHY validation is necessary.
This replaces the clunky TikZ plot in the slides.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Enhanced styling
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Professional color palette
COLOR_TRAIN = '#1976D2'
COLOR_VAL = '#E53935'
COLOR_OPTIMAL = '#06A77D'

def create_validation_necessity_figure():
    """Create a compelling visualization of why we need validation."""

    fig = plt.figure(figsize=(10, 7))

    # Main plot
    ax = plt.subplot(111)

    # Generate realistic curves
    complexity = np.linspace(1, 10, 100)

    # Training error - decreases monotonically
    train_error = 5 * np.exp(-0.4 * complexity) + 0.5

    # Validation error - U-shaped
    val_error = 2.5 * np.exp(-0.3 * complexity) + 0.15 * (complexity - 4)**2 + 1.2

    # Plot curves with enhanced styling
    train_line = ax.plot(complexity, train_error, color=COLOR_TRAIN,
                         linewidth=4, label='Training Error', zorder=3)
    ax.fill_between(complexity, 0, train_error, alpha=0.15, color=COLOR_TRAIN, zorder=1)

    val_line = ax.plot(complexity, val_error, color=COLOR_VAL,
                       linewidth=4, label='Validation Error', zorder=3)
    ax.fill_between(complexity, 0, val_error, alpha=0.15, color=COLOR_VAL, zorder=1)

    # Mark the critical regions

    # 1. Underfitting region
    ax.axvspan(0.5, 2.5, alpha=0.15, color='blue', zorder=0)
    ax.text(1.5, 4.8, 'Underfitting\nRegion', ha='center', fontsize=11,
           fontweight='bold', color='darkblue',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                    edgecolor='darkblue', linewidth=2, alpha=0.8))

    # 2. Optimal point
    optimal_idx = np.argmin(val_error)
    optimal_x = complexity[optimal_idx]
    optimal_y_train = train_error[optimal_idx]
    optimal_y_val = val_error[optimal_idx]

    ax.axvline(optimal_x, color=COLOR_OPTIMAL, linestyle='--',
              linewidth=3, alpha=0.7, zorder=2)
    ax.plot(optimal_x, optimal_y_val, 'o', color=COLOR_OPTIMAL,
           markersize=16, markeredgecolor='white', markeredgewidth=2.5, zorder=5)

    # Annotation for optimal point
    ax.annotate('Optimal\nComplexity',
               xy=(optimal_x, optimal_y_val),
               xytext=(optimal_x - 1.5, optimal_y_val - 0.8),
               fontsize=12, fontweight='bold', color=COLOR_OPTIMAL,
               bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen',
                        edgecolor=COLOR_OPTIMAL, linewidth=2, alpha=0.9),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_OPTIMAL))

    # 3. Overfitting region
    ax.axvspan(7.5, 10.5, alpha=0.15, color='red', zorder=0)
    ax.text(8.7, 4.8, 'Overfitting\nRegion', ha='center', fontsize=11,
           fontweight='bold', color='darkred',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral',
                    edgecolor='darkred', linewidth=2, alpha=0.8))

    # 4. Show the gap between training and validation
    gap_x = 8.5
    gap_idx = np.argmin(np.abs(complexity - gap_x))
    gap_train = train_error[gap_idx]
    gap_val = val_error[gap_idx]

    # Draw arrow showing the gap
    ax.annotate('', xy=(gap_x + 0.3, gap_train), xytext=(gap_x + 0.3, gap_val),
               arrowprops=dict(arrowstyle='<->', lw=2.5, color='black'))
    ax.text(gap_x + 0.8, (gap_train + gap_val) / 2,
           'Overfitting\nGap!', fontsize=10, fontweight='bold',
           va='center', color='darkred',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    # 5. Add warning box for "Training Error Only"
    ax.text(0.98, 0.97,
           "⚠ Without validation:\nWe'd pick complex model\n(lowest training error)",
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFE4E1',
                    edgecolor='darkred', linewidth=2.5, alpha=0.95),
           color='darkred', fontweight='bold')

    # 6. Add success box for "With validation"
    ax.text(0.02, 0.97,
           "✓ With validation:\nWe find optimal\ncomplexity!",
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#E8F5E9',
                    edgecolor=COLOR_OPTIMAL, linewidth=2.5, alpha=0.95),
           color=COLOR_OPTIMAL, fontweight='bold')

    # Styling
    ax.set_xlabel('Model Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error', fontsize=14, fontweight='bold')
    ax.set_title('Why Validation Error is Essential', fontsize=16,
                fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='upper center', frameon=True, shadow=True,
             ncol=2, bbox_to_anchor=(0.5, -0.08))
    ax.set_xlim([0.5, 10.5])
    ax.set_ylim([0, 5.5])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.tight_layout()
    plt.savefig('../figures/validation_necessity.png', dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: validation_necessity.png")


if __name__ == "__main__":
    print("Creating enhanced validation necessity visualization...")
    create_validation_necessity_figure()
    print("Enhanced figure created successfully!")
