#!/usr/bin/env python3
"""
Master Script to Generate All Figures for Clustering
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates all visualization figures for the Clustering module.
"""

import core_concepts
import advanced_methods
import evaluation_metrics
import real_world_applications


def main():
    """Generate all figures for Clustering module"""
    print("\n" + "="*70)
    print(" "*15 + "CLUSTERING MODULE - FIGURE GENERATION")
    print(" "*20 + "CMSC 173 - Machine Learning")
    print(" "*12 + "University of the Philippines - Cebu")
    print("="*70 + "\n")

    print("📊 Generating comprehensive visualization suite...")
    print("   Target: 18-20 figures at 150-200 DPI")
    print("   Estimated time: 2-3 minutes\n")

    # Generate all figure sets
    print("="*70)
    print("[1/4] CORE CONCEPTS")
    print("="*70)
    core_concepts.main()

    print("\n" + "="*70)
    print("[2/4] ADVANCED METHODS")
    print("="*70)
    advanced_methods.main()

    print("\n" + "="*70)
    print("[3/4] EVALUATION METRICS")
    print("="*70)
    evaluation_metrics.main()

    print("\n" + "="*70)
    print("[4/4] REAL-WORLD APPLICATIONS")
    print("="*70)
    real_world_applications.main()

    # Summary
    print("\n" + "="*70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output directory: ../figures/")
    print("\n📊 Figure Summary:")
    print("   • Core Concepts:           6 figures")
    print("   • Advanced Methods:        6 figures")
    print("   • Evaluation Metrics:      5 figures")
    print("   • Real-World Applications: 4 figures")
    print("   " + "-"*40)
    print("   TOTAL:                    21 figures")
    print("\n✨ Ready for LaTeX slides and Jupyter notebook!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
