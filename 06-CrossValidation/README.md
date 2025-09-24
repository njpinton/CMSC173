# Kernel Methods - CMSC 173 Machine Learning

A comprehensive presentation on Kernel Methods covering Support Vector Machines, kernel functions, regression, and multi-class classification.

## 📁 Repository Structure

```
05-KernelMethods/
├── figures/              # Generated visualization plots (15 PNG files)
├── scripts/              # Python scripts for figure generation
├── slides/               # LaTeX presentation and compiled PDF
└── README.md            # This file
```

## 🎯 Presentation Topics

### Core Concepts
- **Introduction & Motivation** - Why kernel methods are essential for non-linear problems
- **Support Vector Machines** - Maximum margin classification principles
- **Large-Margin Classifiers** - Geometric interpretation and optimization
- **Quadratic Optimization** - Primal and dual formulations with KKT conditions

### Advanced Topics
- **Nonlinear SVM using Kernels** - The kernel trick and feature mapping
- **Mercer's Theorem** - Valid kernel conditions and construction rules
- **Multiple Kernel Learning** - Combining kernels for enhanced performance
- **Multi-class Classification** - One-vs-Rest, One-vs-One strategies

### Regression Applications
- **Support Vector Regression** - ε-insensitive loss and sparse solutions
- **Kernel Ridge Regression** - Regularized kernel-based regression
- **Parameter Tuning** - Guidelines for C, γ, and ε selection

## 🔧 Building the Presentation

### Prerequisites
- LaTeX distribution (TeX Live 2025 or similar)
- Python 3.x with required packages:
  ```bash
  pip install numpy matplotlib scikit-learn seaborn
  ```

### Quick Start
1. **Generate all figures:**
   ```bash
   cd scripts/
   python3 generate_all_figures.py
   ```

2. **Compile the presentation:**
   ```bash
   cd slides/
   pdflatex kernel_methods_slides.tex
   ```

### Generated Figures (15 total)
- `linear_svm_margins.png` - SVM decision boundaries and support vectors
- `svm_optimization.png` - Optimization objective visualization
- `hard_vs_soft_margin.png` - Comparison of margin types
- `kernel_trick_transformation.png` - 2D to 3D transformation demo
- `different_kernels.png` - Linear, polynomial, RBF, sigmoid comparison
- `rbf_kernel_parameters.png` - Effect of C and γ parameters
- `kernel_functions.png` - Mathematical visualization of kernel functions
- `multiclass_strategies.png` - OvR, OvO, direct multiclass comparison
- `ovr_detailed.png` - Detailed One-vs-Rest decomposition
- `kernel_multiclass_comparison.png` - Kernel performance on multiclass data
- `multiclass_confidence.png` - Prediction confidence visualization
- `svr_demonstration.png` - SVR with different kernels
- `epsilon_parameter_effect.png` - Impact of ε parameter in SVR
- `kernel_ridge_vs_svr.png` - Comparison of regression methods
- `regularization_comparison.png` - Regularization strength effects

## 📊 Presentation Features

### Mathematical Rigor
- Complete SVM optimization formulations (primal and dual)
- Kernel trick mathematical derivation
- Mercer's theorem conditions and examples
- Worked examples with step-by-step calculations

### Visual Learning
- High-quality matplotlib figures with scikit-learn examples
- Interactive visualizations of decision boundaries
- Parameter effect demonstrations
- Real-world dataset applications

### Practical Implementation
- Hyperparameter tuning guidelines
- Performance comparison tables
- Software library recommendations
- Best practices and common pitfalls

### Professional Presentation
- Metropolis Beamer theme with Wolverine color scheme
- Consistent formatting and alignment
- Alertboxes highlighting key concepts
- 40 slides with balanced content density

## 🚀 Usage

### For Students
- Review mathematical foundations before diving into implementation
- Use figures to understand geometric interpretations
- Follow practical guidelines for real-world applications

### For Instructors
- Modular presentation structure allows topic selection
- High-quality figures suitable for lecture slides
- Comprehensive coverage from basics to advanced topics

### For Practitioners
- Parameter selection guidelines based on data characteristics
- Performance comparison insights
- Implementation best practices

## 🔍 Key Learning Outcomes

After this presentation, learners will understand:

1. **Theoretical Foundations**
   - Maximum margin principle and its importance
   - Kernel trick for implicit high-dimensional mapping
   - Optimization theory behind SVM formulations

2. **Practical Applications**
   - When to use different kernel functions
   - How to tune hyperparameters effectively
   - Multi-class extension strategies

3. **Implementation Skills**
   - Proper data preprocessing for kernel methods
   - Model selection and validation techniques
   - Performance evaluation and debugging

## 📝 Notes

- The presentation assumes familiarity with basic machine learning concepts
- Mathematical notation follows standard ML textbook conventions
- All figures are generated reproducibly using provided Python scripts
- LaTeX source is well-documented for easy customization

## 🔧 Troubleshooting

### Common Issues
- **Missing figures:** Run `python3 generate_all_figures.py` in scripts/ directory
- **LaTeX compilation errors:** Ensure all packages are installed (`pgfplots`, `beamer`, `metropolis`)
- **Font warnings:** Use XeLaTeX or LuaLaTeX for optimal typography (optional)

### Dependencies
- **Python packages:** numpy, matplotlib, scikit-learn, seaborn
- **LaTeX packages:** beamer, metropolis, pgfplots, tikz, amsmath

For additional support or questions, refer to the course materials or contact the instructor.