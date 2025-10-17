# CMSC 173: Machine Learning

**Instructor:** Noel Jeffrey Pinton
**Institution:** University of the Philippines - Cebu
**Department:** Department of Computer Science

Welcome! This repository contains comprehensive educational materials for **CMSC 173: Machine Learning** including professional LaTeX slides, Python implementations, interactive notebooks, and hands-on workshops.

## 📚 Course Modules

### Foundational Topics
- **00-IntroToMachineLearning** - Course overview, ML taxonomy, and applications
- **01-LinearRegression** - Least squares, gradient descent, model evaluation
- **02-LogisticRegression** - Binary/multi-class classification, decision boundaries
- **03-Regularization** - L1/L2 regularization, Ridge, Lasso, Elastic Net
- **04-ExploratoryDataAnalysis** - Data preprocessing, visualization, feature engineering

### Advanced Methods
- **05-KernelMethods** - Kernel trick, SVM, kernel functions
- **06-CrossValidation** - Model selection, k-fold CV, hyperparameter tuning
- **07-NeuralNetworks** - Feedforward networks, backpropagation, optimization
- **08-ParameterEstimation** - MLE, MAP, Bayesian estimation
- **09-ModelSelection** - Bias-variance tradeoff, evaluation metrics
- **10-Clustering** - K-means, hierarchical clustering, DBSCAN

### Specialized Topics
- **11-PCA** - Dimensionality reduction, eigenfaces, feature extraction
- **12-Classification** - KNN, Decision Trees, Naive Bayes
- **13-AdvancedNeuralNetworks** - CNNs, Transformers, GANs, Diffusion Models

## 🎯 Module Structure

Each module follows a consistent, high-quality structure:

- **slides/** - Professional Beamer LaTeX slides (PDF + source)
- **figures/** - High-resolution visualizations (matplotlib, 200 DPI)
- **scripts/** - Python code for generating figures and demonstrations
- **notebooks/** - Interactive Jupyter workshops with exercises
- **README.md** - Comprehensive module documentation
- **requirements.txt** - Python dependencies for reproducibility

## 💻 Getting Started

### Prerequisites
```bash
# Python 3.8+
python3 -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
```

### Using a Module
```bash
cd 11-PCA/  # Example module
pip install -r requirements.txt

# Run interactive workshop
jupyter notebook notebooks/workshop.ipynb

# Generate figures
python scripts/generate_all_figures.py

# View slides
open slides/slides.pdf
```

## 📖 Course Materials

- **CourseOutline.pdf** - Complete course syllabus and schedule
- **create_ml_educational_package_prompt_v2.txt** - Module creation guidelines (v2.1)
- Professional slides with consistent theme and branding
- Executable notebooks with hands-on exercises
- Comprehensive figure generation scripts

## 🎓 Learning Approach

This course emphasizes:
- **Theoretical foundations** with mathematical rigor
- **Practical implementation** using NumPy, scikit-learn, PyTorch
- **Real-world applications** and use cases
- **Hands-on workshops** with guided exercises
- **Professional presentation** suitable for research university students

## 🛠️ Tools and Technologies

- **LaTeX/Beamer** - Slide generation
- **Python** - Implementation language
- **Jupyter** - Interactive notebooks
- **scikit-learn** - Classical ML algorithms
- **PyTorch** - Deep learning frameworks
- **matplotlib/seaborn** - Data visualization

## 📝 Notes for Students

- All materials are self-contained and follow consistent quality standards
- Notebooks include both conceptual explanations and executable code
- Slides use professional LaTeX with UPC branding
- Figures are publication-quality (200 DPI, consistent styling)
- Virtual environments recommended for dependency management

## 🔄 Updates

This repository is actively maintained throughout the semester. Recent additions:
- Four new modules (Intro ML, PCA, Classification, Advanced Neural Networks)
- Enhanced visual design with thematic images and quotes
- Beginner-friendly workshops for advanced topics
- Updated to v2.1 educational package guidelines

## 🤝 Contributing

Feedback and suggestions are welcome! If you find issues or have ideas for improvement:
- Open an issue on GitHub
- Contact the instructor
- Propose improvements to existing materials

---

**Version:** 2.1 (October 2025)
**Status:** Production-Ready
University of the Philippines - Cebu | Department of Computer Science

