#!/usr/bin/env python3
"""
Generate All Figures for Cross Validation Presentation
CMSC 173 - Machine Learning

This script runs all figure generation scripts and creates all visualizations
needed for the cross validation presentation.
"""

import os
import sys

def run_script(script_name):
    """Run a Python script and handle errors."""
    try:
        print(f"\n{'='*50}")
        print(f"Running {script_name}...")
        print(f"{'='*50}")

        # Import and run the script
        if script_name == "validation_methods.py":
            import validation_methods
        elif script_name == "hyperparameter_search.py":
            import hyperparameter_search
        elif script_name == "real_world_examples.py":
            import real_world_examples

        print(f"✓ {script_name} completed successfully!")

    except Exception as e:
        print(f"✗ Error running {script_name}: {str(e)}")
        return False

    return True

def main():
    """Generate all figures for the cross validation presentation."""
    print("Cross Validation Figure Generation")
    print("CMSC 173 - Machine Learning")
    print("=" * 60)

    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # List of scripts to run
    scripts = [
        "validation_methods.py",
        "hyperparameter_search.py",
        "real_world_examples.py"
    ]

    success_count = 0
    total_scripts = len(scripts)

    for script in scripts:
        if os.path.exists(script):
            if run_script(script):
                success_count += 1
        else:
            print(f"Warning: {script} not found, skipping...")

    print(f"\n{'='*60}")
    print(f"Figure Generation Complete!")
    print(f"Successfully generated: {success_count}/{total_scripts} script sets")
    print(f"{'='*60}")

    # Check figures directory
    figures_dir = "../figures"
    if os.path.exists(figures_dir):
        figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        print(f"Total figures generated: {len(figures)}")
        for fig in sorted(figures):
            print(f"  ✓ {fig}")
    else:
        print("Warning: Figures directory not found!")

if __name__ == "__main__":
    main()