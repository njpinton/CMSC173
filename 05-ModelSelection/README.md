# 📊 Model Selection and Evaluation

A comprehensive educational package for understanding model selection, evaluation, and the bias-variance tradeoff in machine learning.

## 🎯 Learning Objectives

By the end of this module, you will be able to:

- **Understand the bias-variance decomposition** and its role in model selection
- **Apply proper validation techniques** including train-test splitting and cross-validation
- **Implement regularization methods** (Ridge, Lasso, Elastic Net) to prevent overfitting
- **Choose and interpret appropriate evaluation metrics** for classification and regression
- **Diagnose and fix underfitting and overfitting** in machine learning models
- **Follow best practices** for reliable model evaluation

## 📁 Repository Structure

```
09-ModelSelection/
├── figures/              # Generated visualizations (16 PNG files)
│   ├── bias_variance_tradeoff.png
│   ├── underfitting_overfitting.png
│   ├── learning_curves.png
│   ├── model_complexity_curve.png
│   ├── train_test_split.png
│   ├── cross_validation_schemes.png
│   ├── validation_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── metrics_comparison.png
│   ├── regularization_effect.png
│   ├── regularization_paths.png
│   ├── l1_vs_l2_geometry.png
│   ├── regularization_comparison.png
│   └── sparsity_comparison.png
├── notebooks/            # Interactive Jupyter workshop
│   └── model_selection_workshop.ipynb
├── scripts/              # Python scripts for figure generation
│   ├── core_methods.py
│   ├── regularization_methods.py
│   ├── validation_methods.py
│   └── generate_all_figures.py
├── slides/               # LaTeX Beamer presentation
│   ├── model_selection_slides.tex
│   └── model_selection_slides.pdf
└── README.md            # This file
```

## 📊 Key Topics Covered

### 1. Bias-Variance Decomposition

Understanding the fundamental tradeoff in machine learning:

- **Bias**: Error from wrong assumptions in the learning algorithm
- **Variance**: Error from sensitivity to training set variations
- **Irreducible Error**: Noise in the data
- **Total Error** = Bias² + Variance + Irreducible Error

**Key Insight**: As model complexity increases, bias decreases but variance increases. The optimal model balances both.

### 2. Model Validation and Evaluation

Proper techniques for assessing model performance:

- **Train-Validation-Test Split**: Standard 3-way split strategy
- **Cross-Validation**: K-Fold, Stratified K-Fold, Leave-One-Out
- **Learning Curves**: Diagnosing underfitting and overfitting
- **Validation Curves**: Hyperparameter tuning visualization

**Golden Rule**: Never evaluate final performance on training data!

### 3. Regularization

Techniques to prevent overfitting:

- **Ridge (L2) Regularization**:
  - Adds penalty: λ × ||w||²₂
  - Shrinks coefficients toward zero
  - Keeps all features
  - Has closed-form solution

- **Lasso (L1) Regularization**:
  - Adds penalty: λ × ||w||₁
  - Can set coefficients exactly to zero
  - Performs automatic feature selection
  - Creates sparse models

- **Elastic Net**:
  - Combines L1 and L2 penalties
  - Best of both worlds
  - Handles correlated features well

### 4. Evaluation Metrics

#### Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: TP / (TP + FP)
- **Recall (Sensitivity)**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions

#### Regression Metrics

- **MSE (Mean Squared Error)**: Average squared difference
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² (Coefficient of Determination)**: Proportion of variance explained

## 🚀 Quick Start

### Prerequisites

```bash
# Required Python packages
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### Installation

```bash
# Install dependencies
pip install numpy matplotlib seaborn scikit-learn jupyter

# Or using conda
conda install numpy matplotlib seaborn scikit-learn jupyter
```

### Generate All Figures

```bash
cd scripts
python generate_all_figures.py
```

Expected output: 16 PNG files in the `figures/` directory.

### Build LaTeX Presentation

```bash
cd slides
pdflatex model_selection_slides.tex
pdflatex model_selection_slides.tex  # Run twice for TOC
```

Expected output: `model_selection_slides.pdf` (45 slides)

### Run Jupyter Workshop

```bash
cd notebooks
jupyter notebook model_selection_workshop.ipynb
```

Or open directly in Google Colab using the badge in the notebook.

## 📚 Workshop Structure

The Jupyter notebook is designed for a **45-60 minute** hands-on session:

### Part 1: Understanding Bias-Variance (10 min)
- Visualize underfitting, good fit, and overfitting
- Explore model complexity curves
- Learn to diagnose fitting issues

### Part 2: Model Complexity and Validation Curves (8 min)
- Systematic evaluation of model complexity
- Understanding training vs test error

### Part 3: Cross-Validation (8 min)
- Implement K-Fold cross-validation
- Robust performance estimation
- Visualize fold-wise results

### Part 4: Regularization (12 min)
- Apply Ridge, Lasso, and Elastic Net
- Compare regularization methods
- Visualize coefficient sparsity

### Part 5: Classification Metrics (7 min)
- Confusion matrices
- ROC curves and AUC
- Precision-Recall tradeoffs

### Student Activity (15 min)
- Independent model selection challenge
- Practice all learned techniques
- Solutions provided for self-checking

## 🎓 Presentation Highlights

The 45-slide presentation covers:

1. **Introduction** (3 slides)
   - The model selection problem
   - Pipeline overview
   - Train-validation-test splitting

2. **Bias-Variance Decomposition** (5 slides)
   - Mathematical formulation
   - Visual explanations
   - Practical implications

3. **Model Validation** (6 slides)
   - Why validation is critical
   - Cross-validation schemes
   - Learning and validation curves

4. **Evaluation Metrics** (5 slides)
   - Classification metrics
   - Regression metrics
   - ROC and precision-recall curves

5. **Regularization** (7 slides)
   - Ridge, Lasso, Elastic Net
   - Geometric interpretation
   - Regularization paths
   - Comparison and use cases

6. **Best Practices** (5 slides)
   - Dos and don'ts
   - Data leakage prevention
   - Hyperparameter tuning strategies
   - Nested cross-validation

7. **Summary** (2 slides)
   - Key takeaways
   - Additional resources

## 🔧 Troubleshooting

### Common Issues

**Issue**: LaTeX compilation errors
```bash
# Solution: Install missing packages
tlmgr install beamertheme-metropolis pgfplots
```

**Issue**: Python import errors
```bash
# Solution: Install missing packages
pip install --upgrade scikit-learn matplotlib seaborn
```

**Issue**: Figures not displaying in Jupyter
```bash
# Solution: Enable inline plotting
%matplotlib inline
```

**Issue**: Overfull boxes in LaTeX
```bash
# Check the log file
grep -i "overfull" model_selection_slides.log
# Current status: 7 overfull boxes (acceptable, <30 target)
```

### Performance Tips

- **For large datasets**: Use stratified sampling for cross-validation
- **For many features**: Start with Lasso to identify important features
- **For correlated features**: Use Elastic Net or Ridge
- **For fast iteration**: Use validation curves before full grid search

## 📖 Additional Resources

### Textbooks

- Hastie, Tibshirani, Friedman - *The Elements of Statistical Learning* (Chapter 7)
- Bishop - *Pattern Recognition and Machine Learning* (Chapter 1.5)
- James et al. - *An Introduction to Statistical Learning* (Chapters 5 & 6)
- Goodfellow et al. - *Deep Learning* (Chapter 5.2)

### Online Resources

- [Scikit-learn User Guide: Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [Cross-Validation Tutorial](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Regularization Techniques](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

### Related Papers

- Hastie et al. (2009) - "The Elements of Statistical Learning"
- Kohavi (1995) - "A Study of Cross-Validation and Bootstrap"
- Tibshirani (1996) - "Regression Shrinkage and Selection via the Lasso"

## 🎯 Learning Outcomes Assessment

After completing this module, you should be able to:

- [ ] Explain the bias-variance tradeoff mathematically and intuitively
- [ ] Implement proper train-validation-test splitting
- [ ] Use K-Fold cross-validation for model selection
- [ ] Apply Ridge, Lasso, and Elastic Net regularization
- [ ] Choose appropriate evaluation metrics for different problems
- [ ] Diagnose overfitting and underfitting from learning curves
- [ ] Prevent data leakage in preprocessing pipelines
- [ ] Select optimal hyperparameters using validation curves
- [ ] Interpret confusion matrices and ROC curves
- [ ] Follow best practices for reliable model evaluation

## 🤝 Contributing

This educational package was created for CMSC 173 Machine Learning. If you find errors or have suggestions for improvement:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This material is provided for educational purposes as part of CMSC 173 Machine Learning course.

## 🙏 Acknowledgments

- Dataset sources: scikit-learn built-in datasets
- Visualization inspiration: scikit-learn documentation
- Mathematical formulations: "The Elements of Statistical Learning"

---

**Course**: CMSC 173 - Machine Learning
**Topic**: Model Selection and Evaluation
**Format**: Lecture slides + Interactive workshop
**Duration**: 90 minutes (45 min lecture + 45 min workshop)

For questions or issues, please refer to the course materials or consult the instructor.
