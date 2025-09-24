#!/usr/bin/env python3
"""
Master script to generate all figures for Kernel Methods presentation.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import linear_svm
    import kernel_methods
    import regression_kernels
    import multiclass_kernels

    print("=" * 60)
    print("GENERATING ALL KERNEL METHODS FIGURES")
    print("=" * 60)

    print("\n1. Generating Linear SVM figures...")
    linear_svm.plot_linear_svm_margins()
    linear_svm.plot_svm_optimization()
    linear_svm.plot_hard_vs_soft_margin()
    print("✓ Linear SVM figures complete")

    print("\n2. Generating Kernel Methods figures...")
    kernel_methods.plot_kernel_trick_transformation()
    kernel_methods.plot_different_kernels()
    kernel_methods.plot_rbf_kernel_parameters()
    kernel_methods.plot_kernel_functions()
    print("✓ Kernel Methods figures complete")

    print("\n3. Generating Regression Kernels figures...")
    regression_kernels.plot_svr_demonstration()
    regression_kernels.plot_epsilon_parameter_effect()
    regression_kernels.plot_kernel_ridge_vs_svr()
    regression_kernels.plot_regularization_comparison()
    print("✓ Regression Kernels figures complete")

    print("\n4. Generating Multiclass Kernels figures...")
    multiclass_kernels.plot_multiclass_strategies()
    multiclass_kernels.plot_ovr_detailed()
    multiclass_kernels.plot_kernel_multiclass_comparison()
    multiclass_kernels.plot_multiclass_confidence()
    print("✓ Multiclass Kernels figures complete")

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("Check the '../figures/' directory for all PNG files.")
    print("=" * 60)

except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required packages are installed:")
    print("pip install numpy matplotlib scikit-learn seaborn")
except Exception as e:
    print(f"Error generating figures: {e}")
    import traceback
    traceback.print_exc()