#!/usr/bin/env python3
"""
Master Script: Generate All Advanced Neural Network Figures
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton

Beginner-friendly visualizations for advanced NN topics.
"""

import cnn_beginner
import generative_models_beginner
import transformers_beginner
import diffusion_beginner

def main():
    print("=" * 70)
    print("Generating All Figures for Advanced Neural Networks")
    print("Beginner-Friendly Version - Focus on Intuition!")
    print("CMSC 173 - Machine Learning")
    print("=" * 70)

    print("\n[1/4] Convolutional Neural Networks...")
    cnn_beginner.main()

    print("\n[2/4] Generative Models (GANs, VAEs)...")
    generative_models_beginner.main()

    print("\n[3/4] Transformers & Attention...")
    transformers_beginner.main()

    print("\n[4/4] Diffusion Models...")
    diffusion_beginner.main()

    print("\n" + "=" * 70)
    print("✅ All 17 figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
