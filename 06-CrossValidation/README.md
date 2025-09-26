# Cross Validation & Hyperparameter Tuning - CMSC 173 Machine Learning

A comprehensive presentation on Cross Validation and Hyperparameter Tuning covering validation methods, search strategies, and best practices.

## 📁 Repository Structure

```
06-CrossValidation/
├── figures/              # Generated visualization plots (13 PNG files)
├── notebooks/            # Interactive Jupyter workshop
├── scripts/              # Python scripts for figure generation
├── slides/               # LaTeX presentation and compiled PDF
└── README.md            # This file
```

## 🎯 Presentation Topics

### Core Validation Methods
- **How to Validate Models** - The fundamental problem and three-way data split
- **Holdout Validation** - Simple train/validation/test splitting
- **K-Fold Cross-Validation** - Systematic rotation of validation sets
- **Leave-One-Out Cross-Validation** - Maximum data utilization approach
- **Stratified K-Fold** - Maintaining class distributions in splits

### Advanced Validation Techniques
- **Time Series Cross-Validation** - Preserving temporal dependencies
- **Group K-Fold** - Handling clustered data
- **Monte Carlo Cross-Validation** - Flexible random sampling
- **Nested Cross-Validation** - Unbiased model selection and evaluation

### Hyperparameter Search Methods
- **Grid Search** - Exhaustive parameter space exploration
- **Random Search** - Efficient random parameter sampling
- **Bayesian Optimization** - Smart search using Gaussian processes
- **Optuna/TPE** - Tree-structured Parzen estimator approach

### Practical Applications
- **Learning Curves** - Diagnosing bias vs variance
- **Validation Curves** - Single parameter optimization
- **Bias-Variance Tradeoff** - Understanding model complexity
- **Best Practices** - Common pitfalls and how to avoid them

## 🚀 Interactive Workshop

### Jupyter Notebook Companion
**File**: `notebooks/cross_validation_workshop.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/njpinton/CMSC173/blob/main/06-CrossValidation/notebooks/cross_validation_workshop.ipynb)

**Features:**
- **Hands-on activities** with real datasets (breast cancer dataset)
- **Step-by-step implementation** of validation methods
- **Interactive comparisons** of grid search vs random search
- **Diagnostic tools** (learning curves, validation curves)
- **Common pitfalls** demonstration with data leakage examples
- **Student activity** - Build your own model selection pipeline
- **Complete solutions** included

**Learning Objectives:**
1. Implement different cross-validation methods
2. Compare hyperparameter search strategies
3. Interpret learning and validation curves
4. Apply best practices to avoid common pitfalls
5. Build complete model selection pipelines

**Time Requirement:** ~45 minutes (including 15-minute student activity)

## 🔧 Building the Presentation

### Prerequisites
- LaTeX distribution (TeX Live 2025 or similar)
- Python 3.x with required packages:
  ```bash
  pip install numpy matplotlib scikit-learn seaborn scipy
  ```

### Quick Start
1. **Generate all figures:**
   ```bash
   cd scripts/
   python3 generate_all_figures.py
   ```

   Or generate them individually:
   ```bash
   python3 validation_methods.py
   python3 hyperparameter_search.py
   python3 real_world_examples.py
   ```

2. **Compile the presentation:**
   ```bash
   cd slides/
   pdflatex cross_validation_slides.tex
   pdflatex cross_validation_slides.tex  # Run twice for cross-references
   ```

### Generated Figures (13 total)
- `holdout_validation.png` - Three-way data split visualization
- `kfold_validation.png` - 5-fold cross-validation demonstration
- `model_comparison_cv.png` - Cross-validated model performance comparison
- `loocv_validation.png` - Leave-one-out cross-validation illustration
- `stratified_kfold_comparison.png` - Regular vs stratified k-fold comparison
- `grid_search_visualization.png` - Grid search parameter space and heatmap
- `random_vs_grid_search.png` - Comparison of search strategies
- `optuna_optimization_trace.png` - Bayesian optimization trajectory
- `search_efficiency_comparison.png` - Performance of different search methods
- `learning_curves_cv.png` - Learning curves for different algorithms
- `validation_curves.png` - Hyperparameter validation curves
- `bias_variance_tradeoff.png` - Visual demonstration of bias-variance decomposition
- `hyperparameter_surface.png` - 3D hyperparameter performance landscape

## 📊 Presentation Features

### Mathematical Rigor
- Complete formulations of k-fold CV error estimation
- Bias-variance decomposition mathematics
- Bayesian optimization theory (Gaussian processes, acquisition functions)
- Statistical significance testing for model comparison

### Visual Learning
- High-quality matplotlib figures with scikit-learn examples
- Interactive parameter exploration visualizations
- Step-by-step algorithm illustrations
- Real-world performance comparisons

### Practical Implementation
- Hyperparameter tuning guidelines for different scenarios
- Cross-validation best practices and common pitfalls
- Computational efficiency considerations
- Statistical testing for model comparison

### Professional Presentation
- Metropolis Beamer theme with Wolverine color scheme
- Consistent formatting and mathematical notation
- Alertboxes highlighting key concepts and insights
- 47 slides with balanced content density

## 🚀 Usage

### For Students
- Understand the fundamental validation problem and why it matters
- Learn when to use different cross-validation methods
- Master hyperparameter search strategies
- Avoid common validation pitfalls

### For Instructors
- Modular presentation structure allows topic selection
- High-quality figures suitable for lecture slides
- Comprehensive coverage from basics to advanced topics
- Mathematical rigor with practical insights

### For Practitioners
- Guidelines for choosing appropriate validation methods
- Hyperparameter optimization strategies for different scenarios
- Performance comparison methodologies
- Best practices for avoiding data leakage and selection bias

## 🔍 Key Learning Outcomes

After this presentation, learners will understand:

1. **Validation Fundamentals**
   - Why proper validation is crucial for model selection
   - The bias-variance tradeoff in validation methods
   - How to choose appropriate validation strategies

2. **Cross-Validation Methods**
   - When to use holdout vs k-fold vs LOOCV
   - How to handle special data types (time series, groups, imbalanced)
   - Statistical considerations for robust evaluation

3. **Hyperparameter Optimization**
   - Trade-offs between grid search, random search, and Bayesian optimization
   - How to set up proper search spaces
   - Computational efficiency considerations

4. **Best Practices**
   - Common pitfalls: data leakage, selection bias, inappropriate CV methods
   - Statistical significance testing for model comparison
   - Proper experimental design for machine learning

## 📝 Implementation Notes

### Python Scripts
- **`validation_methods.py`**: Creates visualizations for different CV methods
- **`hyperparameter_search.py`**: Demonstrates search strategies and efficiency
- **`real_world_examples.py`**: Shows practical applications with real datasets
- **`generate_all_figures.py`**: Convenience script to generate all figures

### LaTeX Presentation
- **Theme**: Matches existing course materials (Metropolis + Wolverine)
- **Figures**: All paths relative to slides directory (`../figures/`)
- **Math**: Extensive use of amsmath for proper equation formatting
- **Structure**: Clear section organization with detailed table of contents

## 🔧 Troubleshooting

### Common Issues
- **Missing figures:** Run `python3 generate_all_figures.py` in scripts/ directory
- **LaTeX compilation errors:** Ensure all packages are installed (beamer, metropolis, pgfplots, tikz, amsmath)
- **Python import errors:** Install required packages with pip
- **Font warnings:** Use XeLaTeX or LuaLaTeX for optimal typography (optional)

### Dependencies
- **Python packages:** numpy, matplotlib, scikit-learn, seaborn, scipy
- **LaTeX packages:** beamer, metropolis, pgfplots, tikz, amsmath, algorithms
- **System requirements:** Python 3.6+, LaTeX distribution with modern packages

## 📚 Additional Resources

### Further Reading
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- James, G., et al. (2013). An Introduction to Statistical Learning
- Raschka, S. & Mirjalili, V. (2019). Python Machine Learning

### Online Resources
- [scikit-learn Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Hyperparameter Optimization Methods](https://optuna.readthedocs.io/)
- [Statistical Learning Theory](https://web.stanford.edu/class/cs229t/)

### Tools and Libraries
- **Optuna**: Advanced hyperparameter optimization
- **Hyperopt**: Distributed hyperparameter optimization
- **Ray Tune**: Scalable hyperparameter tuning
- **Weights & Biases**: Experiment tracking and hyperparameter sweeps

---

For additional support or questions, refer to the course materials or contact the instructor.