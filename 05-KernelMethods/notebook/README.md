# Kernel Methods Jupyter Notebook

This folder contains an interactive Jupyter notebook that accompanies the kernel methods slides for CMSC 173.

## Contents

- `kernel_methods_tutorial.ipynb` - Comprehensive notebook with hands-on examples covering all major kernel methods concepts

## Notebook Sections

1. **Introduction & Motivation** - Visual comparison of linear vs non-linear data
2. **Support Vector Machines** - Linear SVM with margin visualization
3. **Hard vs Soft Margin** - Effect of regularization parameter C
4. **The Kernel Trick** - Comparison of different kernel functions
5. **RBF Kernel Parameter Tuning** - Grid visualization of C and γ effects
6. **Multi-class Classification** - OvR, OvO, and native strategies
7. **Support Vector Regression** - SVR with different kernels
8. **Epsilon Parameter in SVR** - ε-tube visualization
9. **Kernel Ridge Regression** - Comparison with SVR
10. **Hyperparameter Optimization** - Validation curves and grid search

## Prerequisites

```bash
pip install numpy matplotlib scikit-learn seaborn jupyter
```

## Usage

```bash
cd notebook/
jupyter notebook kernel_methods_tutorial.ipynb
```

## Features

- **Interactive visualizations** with matplotlib
- **Minimal output** - focused on key insights
- **Short introductions** for each section
- **Hands-on examples** using scikit-learn
- **Parameter exploration** with visual feedback
- **Real-world datasets** for demonstration

Perfect for students wanting to experiment with kernel methods concepts while following the lecture slides.