#!/usr/bin/env python3
"""
Generate All Figures for Exploratory Data Analysis Presentation
CMSC 173 - Machine Learning

This script runs all figure generation scripts and creates all visualizations
needed for the EDA presentation.
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
        if script_name == "core_methods.py":
            import core_methods
        elif script_name == "advanced_techniques.py":
            import advanced_techniques
        elif script_name == "real_world_examples.py":
            import real_world_examples

        print(f"✓ {script_name} completed successfully!")

    except Exception as e:
        print(f"✗ Error running {script_name}: {str(e)}")
        return False

    return True

def main():
    """Generate all figures for the EDA presentation."""
    print("Exploratory Data Analysis Figure Generation")
    print("CMSC 173 - Machine Learning")
    print("=" * 60)

    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Create figures directory if it doesn't exist
    figures_dir = "../figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"Created figures directory: {figures_dir}")

    # List of scripts to run
    scripts = [
        "core_methods.py",
        "advanced_techniques.py",
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
    print(f"EDA Figure Generation Complete!")
    print(f"Successfully generated: {success_count}/{total_scripts} script sets")
    print(f"{'='*60}")

    # Check figures directory
    if os.path.exists(figures_dir):
        figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        print(f"Total figures generated: {len(figures)}")
        for fig in sorted(figures):
            print(f"  ✓ {fig}")
    else:
        print("Warning: Figures directory not found!")

    print(f"\n{'='*60}")
    print("Figure descriptions:")
    print("• 01_data_types_overview.png - Data structure and types analysis")
    print("• 02_univariate_numerical.png - Numerical variable distributions")
    print("• 03_univariate_categorical.png - Categorical variable frequencies")
    print("• 04_correlation_analysis.png - Feature correlation matrix")
    print("• 05_missing_data_analysis.png - Missing data patterns")
    print("• 06_outlier_detection.png - Outlier identification methods")
    print("• 07_feature_engineering.png - Feature creation examples")
    print("• 08_normalization_comparison.png - Data scaling methods")
    print("• 09_feature_selection.png - Feature importance analysis")
    print("• 10_bivariate_analysis.png - Feature relationship exploration")
    print("• 11_target_analysis.png - Target variable patterns")
    print("• 12_business_insights.png - Actionable business findings")
    print("• 13_ml_pipeline_demo.png - EDA to ML workflow integration")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()