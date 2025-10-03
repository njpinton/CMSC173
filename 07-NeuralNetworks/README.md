# 📊 Artificial Neural Networks

**CMSC 173 - Machine Learning**
**University of the Philippines - Cebu**
**Department of Computer Science**
**Instructor:** Noel Jeffrey Pinton

A comprehensive, publication-quality educational package covering artificial neural networks, including architecture, activation functions, forward propagation, backpropagation, and regularization techniques.

---

## 🎯 Learning Objectives

After completing this module, students will be able to:

1. **Understand** neural network architecture and fundamental components
2. **Implement** forward and backward propagation algorithms from scratch
3. **Apply** different activation functions and analyze their properties
4. **Evaluate** regularization techniques to prevent overfitting
5. **Analyze** gradient flow and convergence properties
6. **Compare** different neural network configurations and hyperparameters

---

## 📁 Repository Structure

```
07-NeuralNetworks/
├── figures/                    # 13 visualization PNGs (300 DPI)
│   ├── perceptron_structure.png
│   ├── activation_functions.png
│   ├── activation_derivatives.png
│   ├── multilayer_network.png
│   ├── forward_propagation_flow.png
│   ├── backpropagation_flow.png
│   ├── computational_graph.png
│   ├── gradient_descent_visualization.png
│   ├── overfitting_regularization_demo.png
│   ├── dropout_visualization.png
│   ├── l1_vs_l2_regularization.png
│   ├── regularization_comparison.png
│   └── training_curves_regularization.png
│
├── notebooks/                  # Interactive Jupyter workshop
│   └── neural_networks_workshop.ipynb  # 60-75 min hands-on session (681KB with outputs)
│
├── scripts/                    # Python figure generation
│   ├── neural_network_basics.py        # Basic visualizations (4 figures)
│   ├── forward_backprop_demo.py        # Forward/backprop demos (4 figures)
│   ├── regularization_techniques.py    # Regularization plots (5 figures)
│   └── generate_all_figures.py         # Master script
│
├── slides/                     # LaTeX Beamer presentation
│   ├── neural_networks_slides.tex      # 53 slides source
│   ├── neural_networks_slides.pdf      # Compiled presentation (2.8MB)
│   └── neural_networks_slides.log      # Compilation log
│
└── README.md                   # This file
```

---

## 📊 Key Topics Covered

### 1. Neural Network Fundamentals
- Perceptron structure and learning rule
- Multi-layer feedforward networks
- Neuron activation and signal propagation
- Network architecture design principles

### 2. Activation Functions
- Sigmoid, tanh, ReLU, Leaky ReLU
- Mathematical properties and derivatives
- Vanishing/exploding gradient problems
- Selection guidelines for different tasks

### 3. Training Algorithms
- Forward propagation mechanics
- Backpropagation algorithm
- Gradient descent optimization
- Computational graph perspective

### 4. Regularization Techniques
- L1 and L2 weight penalties
- Dropout regularization
- Early stopping strategies
- Overfitting prevention

---

## 🚀 Quick Start

### Prerequisites

**Required:**
- Python 3.8+
- NumPy 1.20+
- Matplotlib 3.3+
- Scikit-learn 0.24+
- Jupyter 1.0+

**Optional:**
- Seaborn (for enhanced visualizations)
- Pandas (for data handling)

### Installation

```bash
# Navigate to directory
cd 07-NeuralNetworks

# Install dependencies
pip install numpy matplotlib seaborn scikit-learn jupyter

# Or use a specific version
pip install numpy==1.21.0 matplotlib==3.4.2 scikit-learn==0.24.2
```

### Generate All Figures

```bash
cd scripts
python3 generate_all_figures.py
```

Expected output:
```
============================================================
Generating All Figures for Neural Networks
CMSC 173 - Machine Learning
============================================================

[1/3] Basic Neural Network Concepts...
✓ Generated perceptron_structure.png
✓ Generated activation_functions.png
✓ Generated activation_derivatives.png
✓ Generated multilayer_network.png

[2/3] Forward and Backward Propagation...
✓ Generated forward_propagation_flow.png
✓ Generated backpropagation_flow.png
✓ Generated computational_graph.png
✓ Generated gradient_descent_visualization.png

[3/3] Regularization Techniques...
✓ Generated overfitting_regularization_demo.png
✓ Generated dropout_visualization.png
✓ Generated l1_vs_l2_regularization.png
✓ Generated regularization_comparison.png
✓ Generated training_curves_regularization.png

============================================================
✅ All figures generated successfully!
============================================================
```

### Build LaTeX Presentation

```bash
cd slides
pdflatex neural_networks_slides.tex
pdflatex neural_networks_slides.tex  # Second compilation for TOC
```

**Verify compilation:**
```bash
# Check for errors
grep -i "error" neural_networks_slides.log | wc -l  # Should be 0

# Check overfull boxes
grep "Overfull" neural_networks_slides.log | wc -l  # ~18 (acceptable)

# Verify PDF generated
ls -lh neural_networks_slides.pdf  # Should be ~2.8MB
```

### Run Jupyter Workshop

```bash
cd notebooks
jupyter notebook neural_networks_workshop.ipynb
```

**Or use JupyterLab:**
```bash
jupyter lab neural_networks_workshop.ipynb
```

---

## 📚 Workshop Structure

### Overview
- **Duration:** 60-75 minutes
- **Format:** Interactive coding + theory
- **Difficulty:** Intermediate

### Detailed Breakdown

| Section | Time | Description |
|---------|------|-------------|
| Setup & Imports | 5 min | Environment configuration |
| Part 1: Perceptrons | 10 min | Single neuron fundamentals |
| Part 2: Activation Functions | 12 min | Non-linearity exploration |
| Part 3: Multi-layer Networks | 15 min | Building deeper architectures |
| Part 4: Backpropagation | 15 min | Training algorithm implementation |
| Part 5: Regularization | 10 min | Overfitting prevention |
| Student Challenge | 15 min | Hands-on comparison task |
| Summary | 5 min | Key takeaways |

---

## 🎓 Presentation Highlights

### Slide Distribution (53 total)

| Section | Slides | Focus |
|---------|--------|-------|
| Title & Outline | 2 | Course information |
| Introduction | 5 | Motivation, history, applications |
| Perceptron | 6 | Single neuron architecture |
| Activation Functions | 8 | Non-linearity and derivatives |
| Multi-layer Networks | 7 | Deep architecture |
| Forward Propagation | 6 | Forward pass mechanics |
| Backpropagation | 10 | Gradient computation |
| Regularization | 7 | Overfitting prevention |
| Summary | 2 | Takeaways, next steps |

### Key Features
- ✅ Metropolis theme with Wolverine colors
- ✅ Professional matplotlib figures (300 DPI)
- ✅ Mathematical rigor with derivations
- ✅ Algorithm pseudocode
- ✅ 18 overfull boxes (all <45pt, acceptable)
- ✅ Zero compilation errors

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Figures not generated**
```bash
# Check if output directory exists
ls figures/

# If not, create it
mkdir -p figures

# Re-run generation
cd scripts && python3 generate_all_figures.py
```

**Issue 2: LaTeX compilation errors**
```bash
# Check log file
grep -i "error" slides/neural_networks_slides.log

# Common fix: Missing packages
sudo tlmgr install metropolis beamertheme-metropolis
sudo tlmgr install algorithm algorithmic
```

**Issue 3: Notebook kernel crashes**
```python
# Reduce memory usage
# In notebook, clear variables:
%reset -f

# Or restart kernel:
# Kernel > Restart & Clear Output
```

**Issue 4: Import errors**
```bash
# Verify installations
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "import matplotlib; print(matplotlib.__version__)"

# Reinstall if needed
pip install --upgrade numpy matplotlib scikit-learn
```

---

## 📖 Additional Resources

### Textbooks
- 📕 **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. [Chapters 6-8]
- 📗 **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. [Chapter 5]
- 📘 **Murphy, K. P.** (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. [Chapter 13]

### Research Papers
- 📄 Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323(6088), 533-536.
- 📄 Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). "Dropout: A simple way to prevent neural networks from overfitting." *JMLR*, 15(1), 1929-1958.
- 📄 LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." *Nature*, 521(7553), 436-444.

### Online Resources
- 💻 **GitHub:** [Neural Networks from Scratch](https://github.com/nnfs)
- 🎥 **Videos:** [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- 📝 **Blog Posts:** [Andrej Karpathy's Blog](http://karpathy.github.io/)
- 🎓 **Courses:** [Stanford CS231n](http://cs231n.stanford.edu/)

### Tools & Libraries
- **Scikit-learn:** [Documentation](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- **PyTorch:** [Tutorials](https://pytorch.org/tutorials/)
- **TensorFlow:** [Guide](https://www.tensorflow.org/guide/keras/sequential_model)

---

## 🎯 Learning Outcomes Assessment

After completing this module, you should be able to:

### Conceptual Understanding
- [ ] Explain how neurons process and transmit signals
- [ ] Describe the role of activation functions
- [ ] Identify appropriate network architectures
- [ ] Compare forward and backward propagation

### Technical Skills
- [ ] Implement a feedforward network from scratch
- [ ] Code the backpropagation algorithm
- [ ] Apply regularization techniques effectively
- [ ] Evaluate model performance using metrics

### Advanced Topics
- [ ] Analyze gradient flow and vanishing gradients
- [ ] Derive backpropagation equations
- [ ] Optimize hyperparameters systematically
- [ ] Debug training issues and convergence problems

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
- Classic neural network research by Rumelhart, Hinton, Williams, and LeCun

---

**Last Updated:** October 2, 2025
**Version:** 2.0
**Status:** ✅ Production Ready

---

## 📊 Quality Metrics

**Technical Requirements:**
- ✅ LaTeX compiles: 0 errors
- ✅ Overfull boxes: 18 (all <45pt, acceptable)
- ✅ Python scripts: All execute without errors
- ✅ Figures: 13 generated at 300 DPI
- ✅ Notebook: 681KB with outputs (>200KB requirement)
- ✅ PDF: 2.8MB, 53 slides

**Visual Quality:**
- ✅ Consistent color palette across all figures
- ✅ Professional styling (thick lines, shadows, annotations)
- ✅ High-resolution figures (300 DPI > 150-200 DPI target)
- ✅ LaTeX typography and mathematical notation
- ✅ Readable on projection screens

**Educational Quality:**
- ✅ Clear learning objectives defined
- ✅ Progressive complexity (simple → advanced)
- ✅ Mathematical derivations included
- ✅ Real implementations demonstrated
- ✅ Student activities with solutions
- ✅ Assessment opportunities provided

---

**🎉 Ready to explore the fascinating world of neural networks!**

*This package provides a complete foundation for understanding artificial neural networks through theory, visualization, and hands-on implementation suitable for high-caliber computer science students.*
