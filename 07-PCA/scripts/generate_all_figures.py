#!/usr/bin/env python3
"""
Master Script: Generate All PCA Figures
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates all figures for the PCA module.
"""

import core_concepts
import advanced_methods
import evaluation_metrics
import real_world_applications

def main():
    print("=" * 60)
    print("Generating All Figures for Principal Component Analysis")
    print("CMSC 173 - Machine Learning")
    print("=" * 60)

    print("\n[1/4] Core Concepts...")
    core_concepts.main()

    print("\n[2/4] Advanced Methods...")
    advanced_methods.main()

    print("\n[3/4] Evaluation Metrics...")
    evaluation_metrics.main()

    print("\n[4/4] Real-World Applications...")
    real_world_applications.main()

    print("\n" + "=" * 60)
    print("✅ All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
