# 📊 Clustering

**CMSC 173 - Machine Learning**
**University of the Philippines - Cebu**
**Department of Computer Science**
**Instructor:** Noel Jeffrey Pinton

Comprehensive educational package on clustering algorithms including partitional (K-Means, GMM) and hierarchical methods. This module provides publication-quality materials for understanding and implementing unsupervised learning techniques.

---

## 🎯 Learning Objectives

After completing this module, students will be able to:

1. **Understand** the fundamentals of unsupervised learning and clustering
2. **Implement** K-Means and Hierarchical clustering algorithms from scratch
3. **Apply** Gaussian Mixture Models for soft clustering
4. **Evaluate** clustering quality using internal and external validation metrics
5. **Analyze** convergence properties and computational complexity
6. **Compare** different clustering methods and select appropriate algorithms

---

## 📁 Repository Structure

```
10-Clustering/
├── figures/                    # 21 visualization PNGs (150-200 DPI)
│   ├── clustering_motivation.png
│   ├── kmeans_iterations.png
│   ├── voronoi_diagram.png
│   ├── elbow_method.png
│   ├── silhouette_analysis.png
│   ├── distance_metrics.png
│   ├── kmeans_initialization_comparison.png
│   ├── gmm_soft_clustering.png
│   ├── hierarchical_dendrogram.png
│   ├── linkage_methods_comparison.png
│   ├── clustering_3d.png
│   ├── agglomerative_steps.png
│   ├── internal_validation_metrics.png
│   ├── external_validation_metrics.png
│   ├── cluster_quality_heatmap.png
│   ├── optimal_k_comparison.png
│   ├── clustering_confusion_matrix.png
│   ├── customer_segmentation.png
│   ├── image_color_quantization.png
│   ├── iris_species_clustering.png
│   └── document_clustering.png
│
├── notebooks/                  # Interactive Jupyter workshop
│   └── workshop.ipynb         # 60-75 min hands-on session (1.26MB)
│
├── scripts/                    # Python figure generation
│   ├── core_concepts.py       # Basic clustering visualizations
│   ├── advanced_methods.py    # GMM, hierarchical, 3D plots
│   ├── evaluation_metrics.py  # Validation and metrics
│   ├── real_world_applications.py  # Applied examples
│   └── generate_all_figures.py     # Master script
│
├── slides/                     # LaTeX Beamer presentation
│   ├── clustering_slides.tex  # 50 slides
│   ├── clustering_slides.pdf  # Compiled presentation (5.3MB)
│   └── clustering_slides.log  # Compilation log
│
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 📊 Key Topics Covered

### 1. Introduction & Motivation
- What is clustering?
- Supervised vs unsupervised learning
- Real-world applications
- Types of clustering (partitional vs hierarchical)

### 2. Distance Metrics
- Euclidean distance
- Manhattan distance
- Cosine similarity
- Metric properties and axioms

### 3. K-Means Clustering
- Lloyd's algorithm
- Objective function (WCSS minimization)
- K-Means++ initialization
- Voronoi tessellation
- Choosing K (elbow method, silhouette analysis)

### 4. Gaussian Mixture Models
- Soft clustering approach
- Probabilistic model
- EM algorithm
- Comparison with K-Means

### 5. Hierarchical Clustering
- Agglomerative vs divisive methods
- Linkage criteria (single, complete, average, Ward)
- Dendrograms and interpretation
- Complexity analysis

### 6. Cluster Validation
- **Internal metrics:** Silhouette, Davies-Bouldin, Calinski-Harabasz
- **External metrics:** ARI, NMI
- Model selection strategies

### 7. Real-World Applications
- Customer segmentation
- Image color quantization
- Species classification (Iris dataset)
- Document clustering

---

## 🚀 Quick Start

### Prerequisites

**Required:**
- Python 3.8+
- NumPy 1.20+
- Matplotlib 3.3+
- Scikit-learn 0.24+
- Pandas 1.2+

**Optional (for advanced features):**
- PyTorch 1.8+ (for deep learning examples)
- scikit-image 0.18+ (for image processing)

### Installation

```bash
# Navigate to module directory
cd 10-Clustering

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda create -n cmsc173-clustering python=3.9
conda activate cmsc173-clustering
pip install -r requirements.txt
```

### Generate All Figures

```bash
cd scripts
python3 generate_all_figures.py
```

**Expected output:**
```
======================================================================
               CLUSTERING MODULE - FIGURE GENERATION
                    CMSC 173 - Machine Learning
            University of the Philippines - Cebu
======================================================================

📊 Generating comprehensive visualization suite...
   Target: 18-20 figures at 150-200 DPI
   Estimated time: 2-3 minutes

...

✅ ALL FIGURES GENERATED SUCCESSFULLY!
======================================================================

📁 Output directory: ../figures/

📊 Figure Summary:
   • Core Concepts:           6 figures
   • Advanced Methods:        6 figures
   • Evaluation Metrics:      5 figures
   • Real-World Applications: 4 figures
   ----------------------------------------
   TOTAL:                    21 figures

✨ Ready for LaTeX slides and Jupyter notebook!
```

### Build LaTeX Presentation

```bash
cd slides
pdflatex clustering_slides.tex
pdflatex clustering_slides.tex  # Second compilation for TOC
```

**Verify compilation:**
```bash
# Check for errors
grep -i "error" clustering_slides.log | wc -l  # Should be 0

# Check overfull boxes
grep "Overfull" clustering_slides.log | wc -l  # Should be <20

# Verify PDF generated
ls -lh clustering_slides.pdf  # Should show ~5.3MB
```

### Run Jupyter Workshop

```bash
cd notebooks
jupyter notebook workshop.ipynb
```

**Or use JupyterLab:**
```bash
jupyter lab workshop.ipynb
```

---

## 📚 Workshop Structure

### Overview
- **Duration:** 60-75 minutes
- **Format:** Interactive coding + theory
- **Difficulty:** Intermediate to Advanced
- **File Size:** 1.26MB (with outputs)

### Detailed Breakdown

| Section | Time | Description |
|---------|------|-------------|
| Setup & Imports | 5 min | Environment configuration |
| Part 1: Motivation | 8 min | Clustering fundamentals |
| Part 2: K-Means | 15 min | Implementation from scratch |
| Part 3: Hierarchical | 10 min | Dendrogram and linkage methods |
| Part 4: Evaluation | 10 min | Validation metrics |
| Part 5: GMM | 10 min | Soft clustering |
| Student Challenge | 15 min | Iris dataset clustering |
| Solutions | 8 min | Solution walkthrough |
| Summary | 5 min | Key takeaways |

---

## 🎓 Presentation Highlights

### Slide Distribution (50 slides total)

| Section | Slides | Focus |
|---------|--------|-------|
| Title & Outline | 2 | Course information |
| Introduction | 4 | Motivation, applications |
| Distance Metrics | 3 | Foundations |
| K-Means | 9 | Algorithm, initialization, choosing K |
| GMM | 5 | Soft clustering, EM algorithm |
| Hierarchical | 6 | Agglomerative, linkage, dendrograms |
| Validation | 6 | Internal and external metrics |
| Applications | 5 | Real-world examples |
| Best Practices | 4 | Guidelines, pitfalls |
| Summary | 4 | Takeaways, resources |

### Key Features
- ✅ Metropolis theme with Wolverine colors
- ✅ Professional matplotlib figures (150-200 DPI)
- ✅ Mathematical rigor with derivations
- ✅ Algorithm pseudocode
- ✅ 14 overfull boxes (all <150pt)
- ✅ File size: 5.3MB

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Figures not generated**
```bash
# Check if output directory exists
ls figures/

# If not, scripts will create it automatically
cd scripts && python3 generate_all_figures.py
```

**Issue 2: LaTeX compilation errors**
```bash
# Check log file
grep -i "error" slides/clustering_slides.log

# Common fix: Missing packages
sudo tlmgr install metropolis
sudo tlmgr install algorithm
sudo tlmgr install pgfplots
```

**Issue 3: Notebook kernel crashes**
```python
# Reduce memory usage - in notebook:
%reset -f

# Or restart kernel:
# Kernel > Restart & Clear Output
```

**Issue 4: Import errors**
```bash
# Verify installations
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "import sklearn; print(sklearn.__version__)"

# Reinstall if needed
pip install --upgrade numpy scikit-learn matplotlib
```

**Issue 5: Notebook not showing outputs**
```bash
# Re-execute notebook
cd notebooks
jupyter nbconvert --to notebook --execute --inplace workshop.ipynb

# Verify file size increased
ls -lh workshop.ipynb  # Should be >1MB
```

---

## 📖 Additional Resources

### Textbooks
- 📕 **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Chapter 9]
- 📗 **Murphy, K. P.** (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. [Chapter 21]
- 📘 **Hastie, T., et al.** (2009). *The Elements of Statistical Learning*. Springer. [Chapter 14]

### Research Papers
- 📄 **Arthur, D. & Vassilvitskii, S.** (2007). "K-Means++: The Advantages of Careful Seeding". *SODA*.
- 📄 **Dempster, A., Laird, N., & Rubin, D.** (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm". *JRSS Series B*.
- 📄 **Rousseeuw, P. J.** (1987). "Silhouettes: A Graphical Aid to the Interpretation of Cluster Analysis". *J. Comp. App. Math*.

### Online Resources
- 💻 **scikit-learn:** [Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- 🎥 **StatQuest:** [K-Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
- 📝 **Distill.pub:** [Visualizing K-Means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
- 🎓 **Coursera:** Machine Learning Specialization (Unsupervised Learning)

### Tools & Libraries
- **Python:** scikit-learn, scipy, sklearn.cluster
- **R:** stats::kmeans, stats::hclust, cluster package
- **Julia:** Clustering.jl
- **MATLAB:** kmeans, linkage, clusterdata

---

## 🎯 Learning Outcomes Assessment

After completing this module, you should be able to:

### Conceptual Understanding
- [ ] Explain the difference between supervised and unsupervised learning
- [ ] Describe when to use clustering vs other ML techniques
- [ ] Identify appropriate clustering methods for different data types
- [ ] Compare partitional vs hierarchical approaches

### Technical Skills
- [ ] Implement K-Means algorithm from scratch
- [ ] Apply scikit-learn clustering methods to real data
- [ ] Compute and interpret distance metrics
- [ ] Select optimal K using multiple methods
- [ ] Interpret dendrograms and linkage patterns

### Advanced Topics
- [ ] Understand EM algorithm for GMM
- [ ] Analyze convergence properties of clustering algorithms
- [ ] Derive computational complexity (time and space)
- [ ] Apply validation metrics appropriately
- [ ] Handle real-world challenges (scaling, outliers, etc.)

---

## 💡 Key Algorithms Summary

### K-Means
```
Time Complexity:  O(nKdT) where T = iterations
Space Complexity: O(n + K)
Pros: Fast, simple, scalable
Cons: Sensitive to initialization, assumes spherical clusters
```

### Hierarchical (Agglomerative)
```
Time Complexity:  O(n² log n) with efficient data structures
Space Complexity: O(n²) for distance matrix
Pros: No K required, interpretable dendrogram
Cons: Expensive for large n, can't undo merges
```

### Gaussian Mixture Model
```
Time Complexity:  O(nKd) per EM iteration
Space Complexity: O(nK + Kd²) for full covariance
Pros: Soft assignments, flexible shapes
Cons: More parameters, can overfit
```

---

## 📧 Contact

**Instructor:** Noel Jeffrey Pinton
**Department:** Computer Science
**University:** University of the Philippines - Cebu
**Course:** CMSC 173 - Machine Learning

---

## 📜 License

This educational material is provided for CMSC 173 students at UP Cebu.

---

## 🙏 Acknowledgments

- University of the Philippines - Cebu, Department of Computer Science
- CMSC 173 Machine Learning course students
- scikit-learn development team for excellent documentation
- Research community for foundational clustering algorithms

---

## 📊 Module Statistics

- **Total Figures:** 21 PNG files (150-200 DPI)
- **Total Slides:** 50 (5.3MB PDF)
- **Notebook Size:** 1.26MB (with outputs)
- **Python Scripts:** 5 files (~1,500 lines of code)
- **Documentation:** Comprehensive README (this file)
- **Estimated Completion Time:** 5-7 hours (full module creation)
- **Student Workshop Time:** 60-75 minutes

---

**Last Updated:** October 5, 2025
**Version:** 1.0
**Status:** ✅ Production Ready
