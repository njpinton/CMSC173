# 📊 Kernel Methods

**CMSC 173 - Machine Learning**  
**University of the Philippines - Cebu**  
**Department of Computer Science**  
**Instructor:** Noel Jeffrey Pinton

A comprehensive educational package covering Support Vector Machines, kernel functions, and their applications in classification and regression. This module provides theoretical foundations, practical implementations, and state-of-the-art techniques for kernel-based machine learning.

---

## 🎯 Learning Objectives

After completing this module, students will be able to:

1. **Understand** the mathematical foundations of Support Vector Machines and the maximum margin principle
2. **Implement** kernel methods from scratch using Python and scikit-learn
3. **Apply** the kernel trick to solve non-linear classification and regression problems
4. **Evaluate** kernel-based models using appropriate metrics and validation techniques
5. **Analyze** the computational complexity and convergence properties of SVM optimization
6. **Compare** different kernel functions and select appropriate kernels for specific problems

---

## 📁 Repository Structure

```
05-KernelMethods/
├── figures/                    # 15 visualization PNGs (200 DPI)
│   ├── linear_svm_margins.png
│   ├── svm_optimization.png
│   ├── hard_vs_soft_margin.png
│   ├── kernel_trick_transformation.png
│   ├── different_kernels.png
│   ├── rbf_kernel_parameters.png
│   ├── kernel_functions.png
│   ├── svr_demonstration.png
│   ├── epsilon_parameter_effect.png
│   ├── kernel_ridge_vs_svr.png
│   ├── regularization_comparison.png
│   ├── multiclass_strategies.png
│   ├── ovr_detailed.png
│   ├── kernel_multiclass_comparison.png
│   └── multiclass_confidence.png
│
├── notebook/                   # Interactive Jupyter workshop
│   └── kernel_methods_tutorial.ipynb (3.0 MB with outputs)
│
├── scripts/                    # Python figure generation
│   ├── style_config.py
│   ├── linear_svm.py
│   ├── kernel_methods.py
│   ├── regression_kernels.py
│   ├── multiclass_kernels.py
│   ├── tikz_replacements.py
│   └── generate_all_figures.py
│
├── slides/                     # LaTeX Beamer presentation
│   ├── kernel_methods_slides.tex (37 slides)
│   ├── kernel_methods_slides.pdf
│   └── kernel_methods_slides.log
│
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NumPy, Matplotlib, Scikit-learn
- Jupyter (for notebook)
- LaTeX (for slides)

### Generate Figures

```bash
cd scripts
python3 generate_all_figures.py
```

### Build Slides

```bash
cd slides
pdflatex kernel_methods_slides.tex
pdflatex kernel_methods_slides.tex  # Second pass for TOC
```

### Run Notebook

```bash
cd notebook
jupyter notebook kernel_methods_tutorial.ipynb
```

---

## 📊 Key Topics

1. **Linear SVM**: Maximum margin, hard/soft margin, optimization
2. **Kernel Methods**: Kernel trick, Mercer's theorem, common kernels
3. **Non-Linear Classification**: RBF, polynomial, sigmoid kernels
4. **Kernel Regression**: SVR, kernel ridge regression, parameter tuning
5. **Multiclass**: One-vs-Rest, One-vs-One strategies

---

## 📖 Resources

- **Bishop (2006)**: *Pattern Recognition and Machine Learning*, Ch. 7
- **Murphy (2022)**: *Probabilistic Machine Learning*, Ch. 17
- **Scikit-learn**: [SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)

---

**Last Updated:** October 3, 2025  
**Version:** 1.0 (Enhanced)  
**Status:** ✅ Production Ready
