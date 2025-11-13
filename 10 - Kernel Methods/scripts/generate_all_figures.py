#!/usr/bin/env python3
"""
Master script to generate all figures for Kernel Methods
CMSC 173 - Machine Learning
University of the Philippines - Cebu
Instructor: Noel Jeffrey Pinton
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Generate all kernel methods figures"""
    try:
        import linear_svm
        import kernel_methods
        import regression_kernels
        import multiclass_kernels

        print("=" * 60)
        print("Generating All Kernel Methods Figures")
        print("CMSC 173 - Machine Learning")
        print("=" * 60)

        print("\n[1/4] Linear SVM Figures...")
        linear_svm.main()

        print("\n[2/4] Kernel Methods Figures...")
        kernel_methods.plot_kernel_trick_transformation()
        print("✓ Generated kernel_trick_transformation.png")
        kernel_methods.plot_different_kernels()
        print("✓ Generated different_kernels.png")
        kernel_methods.plot_rbf_kernel_parameters()
        print("✓ Generated rbf_kernel_parameters.png")
        kernel_methods.plot_kernel_functions()
        print("✓ Generated kernel_functions.png")

        print("\n[3/4] Regression Kernels Figures...")
        regression_kernels.plot_svr_demonstration()
        print("✓ Generated svr_demonstration.png")
        regression_kernels.plot_epsilon_parameter_effect()
        print("✓ Generated epsilon_parameter_effect.png")
        regression_kernels.plot_kernel_ridge_vs_svr()
        print("✓ Generated kernel_ridge_vs_svr.png")
        regression_kernels.plot_regularization_comparison()
        print("✓ Generated regularization_comparison.png")

        print("\n[4/4] Multiclass Kernels Figures...")
        multiclass_kernels.plot_multiclass_strategies()
        print("✓ Generated multiclass_strategies.png")
        multiclass_kernels.plot_ovr_detailed()
        print("✓ Generated ovr_detailed.png")
        multiclass_kernels.plot_kernel_multiclass_comparison()
        print("✓ Generated kernel_multiclass_comparison.png")
        multiclass_kernels.plot_multiclass_confidence()
        print("✓ Generated multiclass_confidence.png")

        print("\n" + "=" * 60)
        print("✅ All figures generated successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("Make sure all required packages are installed:")
        print("pip install numpy matplotlib scikit-learn")
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()