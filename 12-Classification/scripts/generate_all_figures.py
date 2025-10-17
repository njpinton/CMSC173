#!/usr/bin/env python3
"""
Master Script: Generate All Classification Figures
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

This script generates all figures for the Classification module.
"""

import naive_bayes_figures
import knn_figures
import decision_tree_figures

def main():
    print("=" * 60)
    print("Generating All Figures for Classification Methods")
    print("CMSC 173 - Machine Learning")
    print("=" * 60)

    print("\n[1/3] Naïve Bayes...")
    naive_bayes_figures.main()

    print("\n[2/3] K-Nearest Neighbors...")
    knn_figures.main()

    print("\n[3/3] Decision Trees...")
    decision_tree_figures.main()

    print("\n" + "=" * 60)
    print("✅ All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
