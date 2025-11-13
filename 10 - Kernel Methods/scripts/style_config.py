#!/usr/bin/env python3
"""
Shared Professional Styling Configuration
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton
"""

import matplotlib.pyplot as plt

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
    import os
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
