# Classification Methods

**CMSC 173 - Machine Learning**
**University of the Philippines - Cebu**
**Department of Computer Science**
**Instructor:** Noel Jeffrey Pinton

Comprehensive educational package on Classification Methods covering three fundamental algorithms: Naive Bayes, K-Nearest Neighbors, and Decision Trees. This module provides publication-quality materials for understanding supervised learning and classification techniques in machine learning.

---

## Learning Objectives

After completing this module, students will be able to:

1. **Understand** the fundamental concepts of classification and supervised learning
2. **Implement** Naive Bayes, K-Nearest Neighbors, and Decision Trees from scratch
3. **Apply** appropriate classification methods to real-world datasets
4. **Evaluate** classification performance using multiple metrics (accuracy, precision, recall, F1-score)
5. **Compare** different classification methods and select the best for a given problem
6. **Interpret** model predictions and explain decision-making processes

---

## Repository Structure

```
12-Classification/
├── figures/                    # 15 visualization PNGs (150-200 DPI)
│   ├── nb_bayes_theorem.png
│   ├── nb_naive_assumption.png
│   ├── nb_gaussian_decision.png
│   ├── nb_iris_example.png
│   ├── knn_concept.png
│   ├── knn_distance_metrics.png
│   ├── knn_decision_boundaries.png
│   ├── knn_bias_variance.png
│   ├── knn_iris_example.png
│   ├── dt_tree_structure.png
│   ├── dt_splitting_criteria.png
│   ├── dt_iris_example.png
│   ├── dt_decision_boundaries.png
│   ├── dt_overfitting.png
│   └── dt_feature_importance.png
│
├── notebooks/                  # Interactive Jupyter workshop
│   └── workshop.ipynb          # 75-90 min hands-on session (10 sections)
│
├── scripts/                    # Python figure generation (optional)
│   └── generate_figures.py    # Script to generate all visualizations
│
├── slides/                     # LaTeX Beamer presentation
│   ├── slides.tex              # 54 slides covering all methods
│   ├── slides.pdf              # Compiled presentation
│   └── slides.log              # Compilation log
│
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Key Topics Covered

### 1. Introduction & Motivation
- What is classification?
- Classification vs regression
- Real-world applications (medical, finance, text, image)
- Supervised learning paradigm

### 2. Naive Bayes Classification
- **Foundations:**
  - Bayes' theorem and conditional probability
  - The "naive" independence assumption
  - Prior, likelihood, posterior probabilities
- **Variants:**
  - Gaussian Naive Bayes (continuous features)
  - Multinomial Naive Bayes (count data)
  - Bernoulli Naive Bayes (binary features)
- **Implementation:**
  - Parameter estimation from training data
  - Log probabilities to avoid underflow
  - Laplace smoothing for zero frequencies
- **Practical Considerations:**
  - When to use Naive Bayes
  - Text classification applications
  - Advantages and limitations

### 3. K-Nearest Neighbors (KNN)
- **Core Concepts:**
  - Instance-based learning
  - Distance metrics (Euclidean, Manhattan, Minkowski)
  - Lazy learning (no training phase)
- **Algorithm:**
  - Finding k nearest neighbors
  - Majority voting for classification
  - Choosing optimal k value
- **Key Considerations:**
  - Importance of feature scaling
  - Curse of dimensionality
  - Bias-variance tradeoff with k
  - Computational complexity
- **Implementation:**
  - From-scratch implementation
  - Cross-validation for k selection
  - Decision boundary visualization

### 4. Decision Trees
- **Foundations:**
  - Tree structure (root, internal nodes, leaves)
  - CART algorithm (Classification And Regression Trees)
  - Greedy recursive splitting
- **Splitting Criteria:**
  - Gini impurity
  - Entropy and information gain
  - Comparing split quality
- **Tree Building:**
  - Feature selection at each node
  - Threshold selection for continuous features
  - Stopping criteria
- **Overfitting & Pruning:**
  - Pre-pruning (early stopping)
  - Post-pruning (cost-complexity)
  - Hyperparameter tuning (max_depth, min_samples_split)
- **Feature Importance:**
  - Computing importance scores
  - Interpreting results
- **Ensemble Methods Preview:**
  - Random Forest
  - Gradient Boosting
  - AdaBoost

### 5. Method Comparison & Selection
- Decision boundary characteristics
- Performance comparison on multiple datasets
- When to use each method
- Computational efficiency
- Interpretability tradeoffs

### 6. Evaluation & Validation
- **Metrics:**
  - Accuracy, precision, recall, F1-score
  - Confusion matrix
  - ROC curves and AUC (binary classification)
- **Validation Strategies:**
  - Train/test split
  - Cross-validation
  - Stratified sampling for imbalanced data
- **Hyperparameter Tuning:**
  - Grid search
  - Random search
  - Cross-validation for model selection

### 7. Best Practices
- Data preprocessing (scaling, encoding, missing values)
- Feature engineering
- Handling class imbalance
- Common pitfalls and solutions
- Production deployment considerations

---

## Quick Start

### Prerequisites

**Required:**
- Python 3.8+
- NumPy 1.20+
- Matplotlib 3.3+
- Scikit-learn 0.24+
- Pandas 1.2+
- Seaborn 0.11+

**Optional:**
- Jupyter 1.0+ (for interactive notebooks)
- LaTeX (full distribution for slides)

### Installation

```bash
# Navigate to module directory
cd 12-Classification

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, matplotlib, sklearn, pandas; print('Setup complete!')"
```

### Run Interactive Workshop

```bash
# Launch Jupyter notebook
cd notebooks
jupyter notebook workshop.ipynb

# Or use JupyterLab
jupyter lab workshop.ipynb

# Workshop duration: 75-90 minutes
# Sections: 10 (Setup, Motivation, Naive Bayes, KNN, Decision Trees,
#              Comparison, Advanced Topics, Challenge, Solutions, Summary)
```

### Compile LaTeX Presentation

```bash
# Navigate to slides directory
cd slides

# Compile with pdflatex (requires full LaTeX distribution)
pdflatex slides.tex
pdflatex slides.tex  # Run twice for proper references

# Output: slides.pdf (54 slides)
```

### Generate Figures (Optional)

```bash
# If figures need to be regenerated
cd scripts
python generate_figures.py

# Figures will be saved in ../figures/
# Expected output: 15 PNG files (150-200 DPI)
```

---

## Workshop Structure

The interactive Jupyter notebook workshop is designed for a 75-90 minute hands-on session:

| Section | Topic | Duration | Activities |
|---------|-------|----------|------------|
| **1** | Setup & Imports | 5 min | Environment verification, library imports |
| **2** | Part 1 - Motivation | 8 min | Classification overview, Iris dataset exploration |
| **3** | Part 2 - Naive Bayes | 15 min | Bayes theorem, from-scratch implementation, visualization |
| **4** | Part 3 - KNN | 15 min | Distance metrics, algorithm implementation, k selection |
| **5** | Part 4 - Decision Trees | 15 min | CART algorithm, tree visualization, feature importance |
| **6** | Part 5 - Comparison | 10 min | Performance metrics, confusion matrices, method comparison |
| **7** | Part 6 - Advanced Topics | 7 min | Text classification, Random Forest preview |
| **8** | Student Challenge | 15 min | Hands-on Wine dataset analysis (7 tasks) |
| **9** | Solutions | 5 min | Detailed solutions with explanations |
| **10** | Summary & Next Steps | 5 min | Key takeaways, resources, future directions |

**Total:** 75-90 minutes (adjustable based on student pace)

---

## Presentation Highlights

The LaTeX Beamer presentation (`slides.pdf`) covers:

| Section | Slides | Key Content |
|---------|--------|-------------|
| **Introduction & Motivation** | 1-4 | Definition, applications, comparison with regression |
| **Naive Bayes** | 5-17 | Bayes theorem, independence assumption, variants, examples, pros/cons |
| **K-Nearest Neighbors** | 18-31 | Algorithm, distance metrics, choosing k, curse of dimensionality, examples |
| **Decision Trees** | 32-47 | CART algorithm, splitting criteria, pruning, feature importance, ensembles |
| **Comparison & Selection** | 48-50 | Decision boundaries, comparison table, when to use each method |
| **Best Practices** | 51-52 | Preprocessing, hyperparameter tuning, common pitfalls |
| **Summary** | 53-54 | Key takeaways, further reading, Q&A |

**Statistics:**
- **Total slides:** 54
- **LaTeX source:** ~1500 lines
- **Figures referenced:** 15
- **Code examples:** 3 complete algorithms (pseudocode)
- **Worked examples:** 3 (one per method)

---

## Datasets Used

### 1. Iris Dataset
- **Samples:** 150
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Use:** Primary dataset for all method demonstrations

### 2. Wine Dataset
- **Samples:** 178
- **Features:** 13 (chemical properties)
- **Classes:** 3 (cultivars)
- **Use:** Student challenge dataset

### 3. Breast Cancer Dataset (Optional)
- **Samples:** 569
- **Features:** 30 (computed from digitized images)
- **Classes:** 2 (malignant, benign)
- **Use:** Bonus challenge, ROC curve demonstration

---

## Figures Overview

All figures are publication-quality PNG images (150-200 DPI):

### Naive Bayes Figures
1. **nb_bayes_theorem.png** - Visual explanation of Bayes' theorem components
2. **nb_naive_assumption.png** - Graphical model showing independence assumption
3. **nb_gaussian_decision.png** - Decision boundaries for Gaussian Naive Bayes
4. **nb_iris_example.png** - Naive Bayes classification results on Iris dataset

### K-Nearest Neighbors Figures
5. **knn_concept.png** - Illustration of KNN classification principle
6. **knn_distance_metrics.png** - Comparison of distance metrics (Euclidean, Manhattan)
7. **knn_decision_boundaries.png** - Decision boundaries for different k values
8. **knn_bias_variance.png** - Bias-variance tradeoff as function of k
9. **knn_iris_example.png** - KNN classification results on Iris dataset

### Decision Tree Figures
10. **dt_tree_structure.png** - Example decision tree structure
11. **dt_splitting_criteria.png** - Comparison of Gini vs Entropy
12. **dt_iris_example.png** - Decision tree classification on Iris dataset
13. **dt_decision_boundaries.png** - Decision boundaries for different depths
14. **dt_overfitting.png** - Visualization of overfitting with tree depth
15. **dt_feature_importance.png** - Feature importance bar chart

---

## Comparison Summary

### Quick Reference Table

| Criterion | Naive Bayes | KNN | Decision Trees |
|-----------|-------------|-----|----------------|
| **Type** | Probabilistic | Instance-based | Rule-based |
| **Training Time** | Fast (O(nd)) | None | Medium (O(nd log n)) |
| **Prediction Time** | Fast (O(Cd)) | Slow (O(nd)) | Fast (O(log n)) |
| **Memory** | Low | High | Medium |
| **Interpretability** | Medium | Low | High |
| **Assumptions** | Independence | Locality | None |
| **Scaling Required** | No | Yes (critical) | No |
| **Handles Categorical** | Yes | Needs encoding | Yes |
| **Overfitting Risk** | Low | Medium | High |
| **Best For** | Text, high-dim | Pattern recognition | Interpretability |

### When to Use Each Method

**Naive Bayes:**
- ✅ Text classification (spam filtering, document categorization)
- ✅ Real-time prediction (fast training and inference)
- ✅ High-dimensional data
- ✅ Small training sets
- ❌ Features highly correlated

**K-Nearest Neighbors:**
- ✅ Small to medium datasets (n < 10,000)
- ✅ Low dimensions (d < 20)
- ✅ Non-linear patterns
- ✅ No training time budget
- ❌ Large datasets or high dimensions

**Decision Trees:**
- ✅ Interpretability crucial
- ✅ Mixed feature types
- ✅ Feature interactions important
- ✅ Building ensemble models
- ❌ Need best single-model accuracy (use ensembles instead)

---

## Common Pitfalls & Solutions

### 1. Not Scaling Features (KNN)
**Problem:** Features with large ranges dominate distance calculations

**Solution:** Always use StandardScaler or MinMaxScaler before KNN
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Overfitting Decision Trees
**Problem:** Deep trees memorize training data

**Solution:** Use pruning parameters
```python
DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10
)
```

### 3. Ignoring Class Imbalance
**Problem:** Majority class dominates predictions

**Solution:** Use class weights or resampling
```python
# Class weights
clf = GaussianNB(class_weight='balanced')

# Or use SMOTE for oversampling
from imblearn.over_sampling import SMOTE
```

### 4. Using Test Data for Tuning
**Problem:** Optimistically biased test accuracy

**Solution:** Use cross-validation
```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, param_grid, cv=5)
```

### 5. Naive Bayes Zero Probabilities
**Problem:** Unseen feature values cause zero probability

**Solution:** Use Laplace smoothing
```python
# Already implemented in sklearn
MultinomialNB(alpha=1.0)  # Laplace smoothing
```

---

## Hyperparameter Tuning Guide

### Naive Bayes
```python
# For Gaussian NB
params = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  # Variance smoothing
}

# For Multinomial NB
params = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]  # Laplace smoothing
}
```

### K-Nearest Neighbors
```python
params = {
    'n_neighbors': [1, 3, 5, 7, 9, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # For Minkowski
}
```

### Decision Trees
```python
params = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 10, 20, 50],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}
```

**Example Usage:**
```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

---

## Troubleshooting

### LaTeX Compilation Issues

**Problem:** `! LaTeX Error: File 'beamer.cls' not found`
```bash
# Solution: Install full LaTeX distribution
# macOS: brew install --cask mactex
# Ubuntu: sudo apt-get install texlive-full
# Windows: Install MiKTeX or TeX Live
```

**Problem:** Missing figures in PDF
```bash
# Solution: Ensure figures directory exists
cd scripts
python generate_figures.py  # Generate figures first
cd ../slides
pdflatex slides.tex
```

### Notebook Issues

**Problem:** Kernel dies during execution
```python
# Solution: Reduce dataset size or increase memory
X_subset = X[:1000]  # Use subset
```

**Problem:** ImportError for packages
```bash
# Solution: Install missing packages
pip install -r requirements.txt
```

### Model Performance Issues

**Problem:** Poor KNN performance
```python
# Solution: Scale features!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Problem:** Decision tree overfitting
```python
# Solution: Limit tree complexity
clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20
)
```

**Problem:** Naive Bayes underperforming
```python
# Solution: Check feature independence assumption
# Consider feature selection or different method
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)
```

---

## Additional Resources

### Recommended Textbooks

1. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman (2009)
   - Chapter 9: Additive Models, Trees, and Related Methods
   - Chapter 13: Prototype Methods and Nearest-Neighbors
   - Comprehensive statistical perspective

2. **"Pattern Recognition and Machine Learning"** - Christopher Bishop (2006)
   - Chapter 4: Linear Models for Classification
   - Mathematical rigor with Bayesian approach

3. **"Machine Learning"** - Tom Mitchell (1997)
   - Chapter 3: Decision Tree Learning
   - Chapter 6: Bayesian Learning
   - Classic introduction to the field

4. **"Machine Learning: A Probabilistic Perspective"** - Kevin Murphy (2012)
   - Chapter 3: Generative Models for Discrete Data
   - Chapter 16: Exemplar-Based Methods
   - Modern probabilistic treatment

### Key Research Papers

1. **Breiman et al. (1984)** - "Classification and Regression Trees (CART)"
   - Original CART algorithm paper

2. **Quinlan (1986)** - "Induction of Decision Trees"
   - ID3 algorithm and information gain

3. **Cover & Hart (1967)** - "Nearest Neighbor Pattern Classification"
   - Theoretical foundations of KNN

4. **Rish (2001)** - "An Empirical Study of the Naive Bayes Classifier"
   - Why Naive Bayes works despite violated assumptions

5. **Domingos & Pazzani (1997)** - "On the Optimality of the Simple Bayesian Classifier"
   - Conditions under which Naive Bayes is optimal

### Online Resources

1. **Scikit-learn Documentation**
   - https://scikit-learn.org/stable/supervised_learning.html
   - Comprehensive API reference and examples

2. **Kaggle Learn**
   - https://www.kaggle.com/learn
   - Free interactive tutorials

3. **StatQuest with Josh Starmer (YouTube)**
   - https://www.youtube.com/c/joshstarmer
   - Excellent visual explanations

4. **UCI Machine Learning Repository**
   - https://archive.ics.uci.edu/ml/
   - Hundreds of datasets for practice

5. **Coursera: Machine Learning by Andrew Ng**
   - Classic introduction to ML
   - Week 6-7 cover classification

### Interactive Demos

1. **Distill.pub Articles**
   - https://distill.pub/
   - Interactive machine learning visualizations

2. **Seeing Theory**
   - https://seeing-theory.brown.edu/
   - Visual introduction to probability (Bayes theorem)

3. **MLDemos**
   - http://mldemos.epfl.ch/
   - Desktop application for visualizing ML algorithms

---

## Learning Outcomes Assessment

Use this checklist to assess your mastery of the module:

### Conceptual Understanding
- [ ] I can explain what classification is and how it differs from regression
- [ ] I understand Bayes' theorem and the naive independence assumption
- [ ] I can describe how KNN makes predictions based on neighbors
- [ ] I understand how decision trees recursively split data
- [ ] I can explain the bias-variance tradeoff for each method

### Mathematical Skills
- [ ] I can compute posterior probabilities using Bayes' theorem
- [ ] I can calculate distances between points (Euclidean, Manhattan)
- [ ] I can compute Gini impurity and entropy for a dataset
- [ ] I understand information gain and how it guides tree building
- [ ] I can work through classification examples by hand

### Implementation Skills
- [ ] I can implement Naive Bayes from scratch using NumPy
- [ ] I can implement KNN from scratch using NumPy
- [ ] I know how to use scikit-learn for all three methods
- [ ] I can preprocess data appropriately (scaling, encoding)
- [ ] I can tune hyperparameters using cross-validation

### Evaluation & Analysis
- [ ] I can compute accuracy, precision, recall, and F1-score
- [ ] I can create and interpret confusion matrices
- [ ] I know when to use which evaluation metric
- [ ] I can visualize decision boundaries for 2D data
- [ ] I can compare multiple classifiers systematically

### Practical Application
- [ ] I can choose the appropriate method for a given problem
- [ ] I can apply classification to real-world datasets
- [ ] I can handle common issues (scaling, imbalance, overfitting)
- [ ] I can interpret feature importance in decision trees
- [ ] I can explain model predictions to stakeholders

### Best Practices
- [ ] I know when feature scaling is required
- [ ] I can avoid data leakage in train/test splits
- [ ] I understand the importance of cross-validation
- [ ] I can recognize and fix overfitting
- [ ] I follow proper ML workflow (preprocess → train → validate → test)

**Target:** Check off at least 80% (24/30) for module completion

---

## Module Statistics

- **Total figures:** 15 (PNG, 150-200 DPI)
- **Presentation slides:** 54
- **Workshop duration:** 75-90 minutes
- **Code sections:** 10 main sections + challenge
- **Learning objectives:** 6 primary outcomes
- **Classification methods:** 3 (Naive Bayes, KNN, Decision Trees)
- **Datasets:** 3 (Iris, Wine, Breast Cancer)
- **Implementation examples:** 3 (from-scratch implementations)
- **Evaluation metrics:** 5 (accuracy, precision, recall, F1, confusion matrix)
- **Hyperparameters covered:** 10+ across all methods
- **Assessment items:** 30 checkboxes

---

## Contact & Support

**Instructor:** Noel Jeffrey Pinton
**Course:** CMSC 173 - Machine Learning
**Institution:** University of the Philippines - Cebu
**Department:** Computer Science

**For questions or issues:**
1. Consult the troubleshooting section above
2. Review the workshop notebook solutions
3. Check scikit-learn documentation
4. Contact instructor during office hours

---

## License & Attribution

This educational module was created for CMSC 173 - Machine Learning at the University of the Philippines - Cebu.

**Usage:**
- Educational use is encouraged
- Attribution required when sharing or adapting materials
- Not for commercial use without permission

**Citation:**
```
Pinton, N.J. (2025). Classification Methods - CMSC 173 Machine Learning Module.
Department of Computer Science, University of the Philippines - Cebu.
```

---

## Version History

**Current Version:** 2.0 (2025)
- Comprehensive 54-slide presentation with 15 figures
- Interactive 10-section workshop notebook (75-90 min)
- Three complete from-scratch implementations
- Extensive comparison and evaluation sections
- Real-world applications and case studies
- Detailed troubleshooting and resources

---

## Quick Reference

### Classification in 5 Steps

1. **Load and explore** your data
2. **Preprocess:** Scale (for KNN), encode categorical, handle missing values
3. **Split:** Train/test or cross-validation
4. **Train:** Choose method and tune hyperparameters
5. **Evaluate:** Use multiple metrics, not just accuracy

### Common Code Patterns

```python
# Pattern 1: Basic classification pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

# Pattern 2: Hyperparameter tuning with cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Pattern 3: Comparing multiple classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5)
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"{name}: {score:.3f}")

# Pattern 4: Visualizing decision boundaries (2D)
import matplotlib.pyplot as plt
import numpy as np

h = 0.02  # Step size
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.show()
```

---

**Happy Learning!**

*"The best way to understand classification is to implement it yourself, experiment with different methods, and apply it to real problems."*
