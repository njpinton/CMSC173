# 📊 Principal Component Analysis

**CMSC 173 - Machine Learning**
**University of the Philippines - Cebu**
**Department of Computer Science**
**Instructor:** Noel Jeffrey Pinton

Comprehensive educational package on Principal Component Analysis (PCA) covering mathematical foundations, implementation, and real-world applications. This module provides publication-quality materials for understanding dimensionality reduction and feature extraction techniques in machine learning.

---

## 🎯 Learning Objectives

After completing this module, students will be able to:

1. **Understand** the mathematical foundations of PCA including covariance matrices and eigenvalue decomposition
2. **Implement** PCA from scratch using eigenvalue decomposition and compare with industry-standard implementations
3. **Apply** PCA for dimensionality reduction, data visualization, and feature extraction on real-world datasets
4. **Evaluate** and interpret principal components using variance explained, scree plots, and reconstruction error
5. **Compare** PCA with other dimensionality reduction techniques including t-SNE, Kernel PCA, and manifold learning methods
6. **Analyze** computational complexity and apply advanced PCA variants for large-scale and nonlinear problems

---

## 📁 Repository Structure

```
11-PCA/
├── figures/                    # 20 visualization PNGs (150-200 DPI)
│   ├── dimensionality_reduction_concept.png
│   ├── curse_of_dimensionality.png
│   ├── variance_visualization.png
│   ├── covariance_eigenvectors.png
│   ├── component_correlation.png
│   ├── pca_reconstruction.png
│   ├── reconstruction_error.png
│   ├── scree_plot.png
│   ├── standardization_importance.png
│   ├── svd_decomposition.png
│   ├── kernel_pca_concept.png
│   ├── sparse_pca.png
│   ├── incremental_pca.png
│   ├── computational_complexity.png
│   ├── digits_visualization.png
│   ├── image_compression.png
│   ├── noise_filtering.png
│   ├── eigenfaces_application.png
│   ├── data_visualization_application.png
│   └── feature_engineering_application.png
│
├── notebooks/                  # Interactive Jupyter workshop
│   └── workshop.ipynb         # 60-75 min hands-on session (9 sections)
│
├── scripts/                    # Python figure generation
│   ├── core_concepts.py       # Basic PCA visualizations
│   ├── advanced_methods.py    # Kernel PCA, Incremental PCA
│   ├── evaluation_metrics.py  # Validation and component selection
│   ├── real_world_applications.py  # Applied examples
│   └── generate_all_figures.py     # Master script
│
├── slides/                     # LaTeX Beamer presentation
│   ├── slides.tex             # 50 slides (2938 lines)
│   ├── slides.pdf             # Compiled presentation (8.0MB)
│   └── slides.log             # Compilation log (33 overfull boxes)
│
└── README.md                   # This file
```

---

## 📊 Key Topics Covered

### 1. Introduction & Motivation
- What is Principal Component Analysis?
- The curse of dimensionality
- Real-world applications and use cases
- When to use (and not use) PCA
- Historical context and development

### 2. Mathematical Foundations
- Covariance matrices and variance
- Eigenvalues and eigenvectors
- Linear algebra fundamentals
- Orthogonal transformations
- Variance maximization principle
- Mathematical derivation of PCA

### 3. The PCA Algorithm
- Step-by-step algorithm walkthrough
- Data standardization and centering
- Covariance matrix computation
- Eigenvalue decomposition
- Component selection strategies
- Data projection and transformation
- Reconstruction and inverse transform

### 4. PCA Variants & Extensions
- **Kernel PCA:** Nonlinear dimensionality reduction
- **Incremental PCA:** Batch processing for large datasets
- **Sparse PCA:** Interpretable sparse loadings
- **Randomized PCA:** Fast approximation for big data
- **Probabilistic PCA:** Bayesian framework

### 5. Evaluation & Component Selection
- Explained variance ratio
- Scree plots and elbow method
- Reconstruction error analysis
- Kaiser criterion
- Cross-validation approaches
- Choosing optimal number of components

### 6. Applications & Case Studies
- Image compression and reconstruction
- Eigenfaces for face recognition
- Data visualization (2D/3D projections)
- Noise filtering and denoising
- Feature engineering and preprocessing
- Exploratory data analysis

### 7. Best Practices & Common Pitfalls
- Importance of standardization
- Avoiding data leakage with train/test splits
- Interpreting principal components
- Computational considerations
- When PCA fails
- Alternative approaches

---

## 🚀 Quick Start

### Prerequisites

**Required:**
- Python 3.8+
- NumPy 1.20+
- Matplotlib 3.3+
- Scikit-learn 0.24+
- Pandas 1.2+
- Seaborn 0.11+

**Optional (for advanced features):**
- SciPy 1.6+ (for advanced linear algebra)
- Jupyter 1.0+ (for interactive notebooks)

### Installation

```bash
# Navigate to module directory
cd 11-PCA

# Install dependencies
pip install numpy matplotlib scikit-learn pandas seaborn scipy jupyter

# Verify installation
python -c "import numpy, matplotlib, sklearn; print('Setup complete!')"
```

### Generate All Figures

```bash
# Run the master figure generation script
cd scripts
python generate_all_figures.py

# Figures will be saved in ../figures/
# Expected output: 20 PNG files (150-200 DPI)
# Total generation time: ~2-3 minutes
```

### Compile LaTeX Presentation

```bash
# Navigate to slides directory
cd slides

# Compile with pdflatex (requires full LaTeX distribution)
pdflatex slides.tex
pdflatex slides.tex  # Run twice for proper references

# Output: slides.pdf (8.0MB, 50 slides)
# Note: 33 overfull boxes (within acceptable limits)
```

### Run Interactive Workshop

```bash
# Launch Jupyter notebook
cd notebooks
jupyter notebook workshop.ipynb

# Or use JupyterLab
jupyter lab workshop.ipynb

# Workshop duration: 60-75 minutes
# Sections: 9 (Setup, Motivation, Core Concepts, Implementation,
#              Evaluation, Advanced Topics, Challenge, Solutions, Summary)
```

---

## 📖 Workshop Structure

The interactive Jupyter notebook workshop is designed for a 60-75 minute hands-on session:

| Section | Topic | Duration | Activities |
|---------|-------|----------|------------|
| **1** | Setup & Imports | 5 min | Environment verification, library imports |
| **2** | Part 1 - Motivation | 8 min | Curse of dimensionality demo, handwritten digits example |
| **3** | Part 2 - Core Concepts | 12 min | Mathematical foundations, 2D PCA visualization, variance explained |
| **4** | Part 3 - Implementation | 15 min | PCA from scratch, sklearn comparison, reconstruction demo |
| **5** | Part 4 - Evaluation | 10 min | Performance metrics, comparison with t-SNE |
| **6** | Part 5 - Advanced Topics | 10 min | Kernel PCA, computational complexity, Incremental PCA |
| **7** | Student Challenge | 15 min | Hands-on Iris dataset analysis (6 tasks) |
| **8** | Solutions | 5 min | Detailed solutions with explanations |
| **9** | Summary & Next Steps | 5 min | Key takeaways, practical guidelines, resources |

**Total:** 60-75 minutes (adjustable based on student pace)

---

## 🎓 Presentation Highlights

The LaTeX Beamer presentation (`slides.pdf`) covers:

| Section | Slides | Key Content |
|---------|--------|-------------|
| **Introduction & Motivation** | 1-15 | Definition, applications, curse of dimensionality, real-world examples |
| **Mathematical Foundations** | 16-32 | Covariance, eigenvalues, linear algebra, derivation, worked examples |
| **The PCA Algorithm** | 33-42 | Step-by-step algorithm, implementation pseudocode, visualization |
| **PCA Variants** | 43-48 | Kernel PCA, Incremental PCA, Sparse PCA, Randomized PCA |
| **Evaluation & Component Selection** | 49-57 | Variance explained, scree plots, reconstruction error, selection criteria |
| **Applications** | 58-68 | Image compression, eigenfaces, visualization, noise filtering, feature engineering |
| **Best Practices & Pitfalls** | 69-77 | Standardization, common mistakes, computational tips, when not to use PCA |
| **Summary** | 78-50 | Key takeaways, comparison table, resources, Q&A |

**Statistics:**
- **Total slides:** 50
- **LaTeX source:** 2938 lines
- **PDF size:** 8.0 MB
- **Figures embedded:** 20
- **Code examples:** 12+ algorithms/implementations
- **Overfull boxes:** 33 (typography warnings, non-critical)

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Figure Generation Issues

**Problem:** `ModuleNotFoundError: No module named 'sklearn'`
```bash
# Solution: Install scikit-learn
pip install scikit-learn
```

**Problem:** Figures look pixelated or low quality
```bash
# Solution: Adjust DPI in script
# In generate_all_figures.py, increase DPI parameter:
plt.savefig('figure.png', dpi=200, bbox_inches='tight')
```

**Problem:** Script hangs during execution
```bash
# Solution: Disable interactive backend
# Add at start of script:
import matplotlib
matplotlib.use('Agg')
```

#### LaTeX Compilation Issues

**Problem:** `! LaTeX Error: File 'beamer.cls' not found`
```bash
# Solution: Install full LaTeX distribution
# macOS: brew install --cask mactex
# Ubuntu: sudo apt-get install texlive-full
# Windows: Install MiKTeX or TeX Live
```

**Problem:** Overfull hbox warnings
```text
# Note: 33 overfull boxes are expected and acceptable
# They indicate minor text overflow, not errors
# PDF compiles successfully with these warnings
```

**Problem:** Missing figures in PDF
```bash
# Solution: Ensure figures/ directory exists and contains all PNGs
cd scripts
python generate_all_figures.py  # Generate figures first
cd ../slides
pdflatex slides.tex  # Then compile
```

#### Notebook Issues

**Problem:** Kernel dies during execution
```bash
# Solution: Increase memory or use subset of data
# In notebook cells, reduce dataset size:
X_subset = X[:1000]  # Use first 1000 samples
```

**Problem:** Plots not displaying inline
```python
# Solution: Add magic command at start of notebook
%matplotlib inline
```

**Problem:** Slow t-SNE execution
```python
# Solution: Use smaller subset for t-SNE demo
X_tsne = tsne.fit_transform(X_test[:500])  # Already implemented
```

#### General Python Issues

**Problem:** `numpy.linalg.LinAlgError: Eigenvalues did not converge`
```python
# Solution: Add small regularization to covariance matrix
cov_matrix = np.cov(X.T) + 1e-10 * np.eye(X.shape[1])
```

**Problem:** Different results from sklearn PCA
```python
# Solution: Check signs of eigenvectors (arbitrary)
# PCA components are unique up to sign flip
# Use absolute values for comparison
```

---

## 📚 Additional Resources

### Recommended Textbooks

1. **"Pattern Recognition and Machine Learning"** - Christopher Bishop (2006)
   - Chapter 12: Continuous Latent Variables (PCA coverage)
   - Mathematical rigor with probabilistic interpretation

2. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman (2009)
   - Chapter 14.5: Principal Components, Curves and Surfaces
   - Statistical perspective with practical examples

3. **"Principal Component Analysis"** - I.T. Jolliffe (2002)
   - Comprehensive book dedicated entirely to PCA
   - Theory and applications

4. **"Machine Learning: A Probabilistic Perspective"** - Kevin Murphy (2012)
   - Chapter 12: Latent Linear Models (includes PCA, PPCA, Factor Analysis)

### Key Research Papers

1. **Pearson, K. (1901)** - "On Lines and Planes of Closest Fit to Systems of Points in Space"
   - Original PCA paper, philosophical foundations

2. **Hotelling, H. (1933)** - "Analysis of a Complex of Statistical Variables into Principal Components"
   - Mathematical formalization of PCA

3. **Jolliffe, I.T. & Cadima, J. (2016)** - "Principal Component Analysis: A Review and Recent Developments"
   - Modern comprehensive review paper

4. **Schölkopf, B., Smola, A., & Müller, K.R. (1998)** - "Nonlinear Component Analysis as a Kernel Eigenvalue Problem"
   - Kernel PCA introduction

5. **Halko, N., Martinsson, P.G., & Tropp, J.A. (2011)** - "Finding Structure with Randomness"
   - Randomized algorithms for large-scale PCA

### Online Resources

1. **Scikit-learn Documentation**
   - https://scikit-learn.org/stable/modules/decomposition.html
   - Comprehensive API reference and examples

2. **StatQuest with Josh Starmer (YouTube)**
   - "PCA Step-by-Step" video series
   - Excellent intuitive explanations

3. **Coursera: Machine Learning Specialization**
   - Andrew Ng's course (Week 8: Dimensionality Reduction)
   - Free audit option available

4. **Victor Lavrenko's YouTube Channel**
   - "PCA" lecture series
   - Clear mathematical explanations

5. **3Blue1Brown - "Eigenvectors and eigenvalues"**
   - Visual intuition for linear algebra concepts
   - Beautiful animations

### Interactive Demos

1. **Principal Component Analysis Explained Visually**
   - http://setosa.io/ev/principal-component-analysis/
   - Interactive browser-based demo

2. **PCA Visualization (by Victor Powell)**
   - Interactive 2D/3D PCA explorer
   - Adjust parameters in real-time

### Code Repositories

1. **Scikit-learn Source Code**
   - https://github.com/scikit-learn/scikit-learn
   - `sklearn/decomposition/pca.py` - Production-quality implementation

2. **PCA Tutorial Notebooks**
   - https://github.com/jakevdp/PythonDataScienceHandbook
   - Chapter 5: Machine Learning (includes PCA)

---

## ✅ Learning Outcomes Assessment

Use this checklist to assess your mastery of the module content:

### Conceptual Understanding
- [ ] I can explain what PCA is and why it's useful
- [ ] I understand the curse of dimensionality and how PCA addresses it
- [ ] I can describe the mathematical foundations (covariance, eigenvalues)
- [ ] I understand the difference between PCA and other dimensionality reduction methods
- [ ] I can explain when to use PCA and when not to use it

### Mathematical Skills
- [ ] I can compute covariance matrices by hand for small datasets
- [ ] I understand eigenvalue decomposition conceptually
- [ ] I can interpret eigenvectors as principal components
- [ ] I understand the variance maximization objective
- [ ] I can work through the PCA derivation step-by-step

### Implementation Skills
- [ ] I can implement PCA from scratch using NumPy
- [ ] I know how to use scikit-learn's PCA implementation
- [ ] I can standardize data appropriately before PCA
- [ ] I can apply PCA to real-world datasets
- [ ] I can reconstruct data from principal components

### Evaluation & Analysis
- [ ] I can create and interpret scree plots
- [ ] I understand explained variance ratio
- [ ] I can calculate reconstruction error
- [ ] I know how to choose the optimal number of components
- [ ] I can create biplots to visualize loadings

### Advanced Topics
- [ ] I understand Kernel PCA for nonlinear patterns
- [ ] I know when to use Incremental PCA for large datasets
- [ ] I can compare PCA with t-SNE and other methods
- [ ] I understand computational complexity trade-offs
- [ ] I'm familiar with Sparse PCA and Probabilistic PCA

### Practical Application
- [ ] I can apply PCA for data visualization
- [ ] I can use PCA for feature engineering/preprocessing
- [ ] I can implement eigenfaces for face recognition
- [ ] I can use PCA for image compression
- [ ] I can troubleshoot common PCA issues

### Communication Skills
- [ ] I can explain PCA to non-technical audiences
- [ ] I can write clear documentation for PCA code
- [ ] I can present PCA results effectively
- [ ] I can justify PCA choices in technical reports
- [ ] I can teach PCA concepts to peers

**Target:** Check off at least 80% (28/35) for module completion

---

## 🎯 Module Statistics

- **Total figures:** 20 (PNG, 150-200 DPI)
- **Presentation slides:** 50 (8.0 MB PDF)
- **Workshop duration:** 60-75 minutes
- **Code sections:** 9 main sections
- **Learning objectives:** 6 primary outcomes
- **Python scripts:** 5 (core_concepts, advanced_methods, evaluation_metrics, real_world_applications, generate_all_figures)
- **LaTeX source:** 2938 lines
- **Overfull boxes:** 33 (acceptable)
- **Topics covered:** 7 major areas
- **Hands-on challenges:** 6 student tasks
- **Assessment items:** 35 checkboxes

---

## 📞 Contact & Support

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

## 📄 License & Attribution

This educational module was created for CMSC 173 - Machine Learning at the University of the Philippines - Cebu.

**Usage:**
- Educational use is encouraged
- Attribution required when sharing or adapting materials
- Not for commercial use without permission

**Citation:**
```
Pinton, N.J. (2025). Principal Component Analysis - CMSC 173 Machine Learning Module.
Department of Computer Science, University of the Philippines - Cebu.
```

---

## 🔄 Version History

**Current Version:** 2.0 (2025)
- Comprehensive 50-slide presentation with 20 figures
- Interactive 9-section workshop notebook (60-75 min)
- Complete implementation from scratch
- Advanced topics: Kernel PCA, Incremental PCA, Sparse PCA
- Real-world applications and case studies
- Extensive troubleshooting and resources

---

## 🌟 Quick Reference

### PCA in 5 Steps

1. **Standardize** your data: `X_scaled = StandardScaler().fit_transform(X)`
2. **Create PCA object:** `pca = PCA(n_components=k)`
3. **Fit to training data:** `pca.fit(X_train_scaled)`
4. **Transform data:** `X_reduced = pca.transform(X_scaled)`
5. **Evaluate:** Check `pca.explained_variance_ratio_`

### Common Patterns

```python
# Pattern 1: Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(StandardScaler().fit_transform(X))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)

# Pattern 2: Retain 95% variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced to {pca.n_components_} components")

# Pattern 3: Image reconstruction
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(images)
X_reconstructed = pca.inverse_transform(X_reduced)

# Pattern 4: Find optimal k
explained_var = []
for k in range(1, X.shape[1]+1):
    pca = PCA(n_components=k)
    pca.fit(X_scaled)
    explained_var.append(pca.explained_variance_ratio_.sum())
plt.plot(range(1, len(explained_var)+1), explained_var)
```

---

**Happy Learning! 🚀**

*"The best way to understand PCA is to implement it yourself, visualize it extensively, and apply it to real problems."*
