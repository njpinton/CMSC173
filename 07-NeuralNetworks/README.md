# Neural Networks Educational Package
**CMSC 173 - Machine Learning**

A comprehensive educational package covering artificial neural networks, including architecture, activation functions, forward propagation, backpropagation, and regularization techniques.

## 📚 Package Contents

### 🎯 Learning Objectives
Students will learn to:
- Understand neural network architecture and components
- Implement forward and backward propagation algorithms
- Compare different activation functions and their properties
- Apply regularization techniques to prevent overfitting
- Build and train neural networks from scratch
- Evaluate model performance using appropriate metrics

### 📂 Directory Structure
```
07-NeuralNetworks/
├── README.md                           # This file
├── slides/
│   ├── neural_networks_slides.tex      # LaTeX Beamer presentation source
│   └── neural_networks_slides.pdf      # Compiled presentation (46 slides)
├── scripts/
│   ├── neural_network_basics.py        # Basic neural network visualizations
│   ├── forward_backprop_demo.py        # Forward/backward propagation demos
│   ├── regularization_techniques.py    # Regularization method visualizations
│   └── generate_all_figures.py         # Master script to generate all figures
├── notebooks/
│   └── neural_networks_workshop.ipynb  # Interactive student workshop
└── figures/                            # Generated visualization outputs
    ├── perceptron_structure.png
    ├── activation_functions.png
    ├── activation_derivatives.png
    ├── multilayer_network.png
    ├── forward_propagation_flow.png
    ├── backpropagation_flow.png
    ├── computational_graph.png
    ├── gradient_descent_visualization.png
    ├── overfitting_demo.png
    ├── dropout_visualization.png
    ├── regularization_comparison.png
    ├── training_curves_regularization.png
    └── l1_vs_l2_comparison.png
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib seaborn scikit-learn jupyter
```

### Generate All Figures
```bash
cd scripts/
python generate_all_figures.py
```

### View Presentation
```bash
cd slides/
# LaTeX compilation (requires pdflatex)
pdflatex neural_networks_slides.tex
# Or view the pre-compiled PDF
open neural_networks_slides.pdf
```

### Run Interactive Workshop
```bash
cd notebooks/
jupyter notebook neural_networks_workshop.ipynb
```

## 📊 Figures and Visualizations

### Basic Neural Network Concepts (4 figures)
1. **Perceptron Structure** - Detailed diagram of a single perceptron with weights, bias, and activation
2. **Activation Functions** - Comparison of sigmoid, tanh, ReLU, and Leaky ReLU
3. **Activation Derivatives** - Gradients of activation functions for backpropagation
4. **Multi-layer Network** - Architecture of a feedforward neural network

### Forward and Backward Propagation (4 figures)
5. **Forward Propagation Flow** - Step-by-step forward pass visualization
6. **Backpropagation Flow** - Gradient flow during backward pass
7. **Computational Graph** - Graph representation for automatic differentiation
8. **Gradient Descent** - Optimization landscape and parameter updates

### Regularization Techniques (5 figures)
9. **Overfitting Demonstration** - Training vs validation curves
10. **Dropout Visualization** - How dropout prevents overfitting
11. **Regularization Comparison** - L1 vs L2 regularization effects
12. **Training Curves with Regularization** - Impact on learning dynamics
13. **L1 vs L2 Parameter Effects** - Weight distribution comparisons

## 🎓 Educational Components

### 1. LaTeX Beamer Presentation (46 slides)
- **Introduction to Neural Networks** (7 slides)
- **Neural Network Architecture** (8 slides)
- **Activation Functions** (8 slides)
- **Forward Propagation** (6 slides)
- **Backpropagation Algorithm** (8 slides)
- **Regularization Techniques** (7 slides)
- **Summary and Next Steps** (2 slides)

**Features:**
- Metropolis theme with Wolverine color scheme
- Mathematical rigor with proper notation
- Code examples and practical insights
- Professional alertboxes for key concepts

### 2. Interactive Jupyter Notebook
- **Part 1:** Understanding activation functions with implementations
- **Part 2:** Building neural networks from scratch
- **Part 3:** Training on real datasets
- **Part 4:** 15-minute student activity comparing activation functions
- **Part 5:** Regularization techniques (dropout, L2)
- **Part 6:** Comprehensive model evaluation

**Student Activity:**
- Compare sigmoid, tanh, and ReLU activations
- Train networks with different configurations
- Analyze performance differences
- Complete solutions provided

### 3. Python Scripts for Figure Generation
All scripts follow consistent styling and include:
- Comprehensive documentation
- Error handling
- Modular design
- Professional visualizations
- Educational annotations

## 🔧 Technical Details

### Neural Network Implementation
- **From-scratch implementation** using only NumPy
- **Forward propagation** with matrix operations
- **Backpropagation** using chain rule
- **Multiple activation functions** (sigmoid, tanh, ReLU)
- **Regularization techniques** (dropout, L2)
- **Gradient descent optimization**

### Dataset Examples
- **Binary classification** with synthetic data
- **Complex patterns** (concentric circles)
- **Feature standardization** demonstrated
- **Train/validation splits** for proper evaluation

### Key Algorithms Covered
1. **Perceptron learning rule**
2. **Feedforward neural networks**
3. **Backpropagation algorithm**
4. **Gradient descent optimization**
5. **Dropout regularization**
6. **L1/L2 weight penalties**

## 📈 Learning Outcomes Assessment

### Knowledge Checks
- Activation function properties and derivatives
- Forward propagation matrix operations
- Backpropagation gradient calculations
- Regularization effects on training dynamics

### Practical Skills
- Implement neural networks from scratch
- Choose appropriate activation functions
- Apply regularization techniques
- Evaluate model performance
- Debug training issues

### Student Activity Solutions
Complete solutions provided for:
- Activation function comparison
- Architecture impact analysis
- Regularization effectiveness
- Performance metric interpretation

## 🎯 Course Integration

### Prerequisites
- Linear algebra (matrix operations)
- Calculus (derivatives, chain rule)
- Python programming basics
- Basic machine learning concepts

### Time Allocation
- **Lecture:** 90 minutes (using slides)
- **Workshop:** 60 minutes (Jupyter notebook)
- **Student Activity:** 15 minutes (built into workshop)
- **Discussion:** 15 minutes (after activity)

### Assessment Opportunities
- Quiz on activation function properties
- Programming assignment: implement backpropagation
- Project: compare regularization methods
- Presentation: neural network architectures

## 🔗 Extensions and Advanced Topics

### Suggested Follow-ups
1. **Convolutional Neural Networks** for image processing
2. **Recurrent Neural Networks** for sequences
3. **Optimization algorithms** (Adam, RMSprop)
4. **Batch normalization** techniques
5. **Deep learning frameworks** (TensorFlow, PyTorch)

### Research Connections
- Universal approximation theorem
- Lottery ticket hypothesis
- Neural network interpretability
- Adversarial examples
- Transfer learning

## 📚 References and Further Reading

### Textbooks
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

### Key Papers
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*.
- Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*.

### Online Resources
- CS231n: Convolutional Neural Networks (Stanford)
- Deep Learning Specialization (Coursera)
- Neural Networks and Deep Learning (3Blue1Brown)

## 🛠️ Troubleshooting

### Common Issues

#### 1. **LaTeX Compilation**

**Problem:** Overfull box warnings during compilation
```bash
Overfull \vbox (32.53pt too high) detected at line 70
```

**Solution:** This is acceptable! The package has ~18 overfull box warnings which are cosmetic only and don't affect PDF readability. According to course standards:
- ✅ **Acceptable:** <30 minor warnings (<30pt each)
- ⚠️ **Review:** 30-50pt warnings - check PDF for readability issues
- ❌ **Critical:** >50pt warnings - content may be cut off

**How to check:**
```bash
cd slides/
pdflatex neural_networks_slides.tex
grep -i "overfull" neural_networks_slides.log | wc -l
# View actual PDF to confirm readability
open neural_networks_slides.pdf
```

**Problem:** Missing LaTeX packages
```bash
! LaTeX Error: File `metropolis.sty' not found
```

**Solution:** Install required packages
```bash
# For TeX Live
tlmgr install metropolis beamertheme-metropolis

# For MacTeX
sudo tlmgr update --self
sudo tlmgr install metropolis
```

#### 2. **Python Script Issues**

**Problem:** Import errors when running scripts
```bash
ModuleNotFoundError: No module named 'numpy'
```

**Solution:** Install required packages
```bash
pip install numpy matplotlib seaborn scikit-learn
# Or use requirements if available
pip install -r requirements.txt
```

**Problem:** Figure generation fails
```bash
PermissionError: [Errno 13] Permission denied: '../figures/'
```

**Solution:** Check directory permissions
```bash
cd scripts/
mkdir -p ../figures
chmod 755 ../figures
python3 generate_all_figures.py
```

#### 3. **Jupyter Notebook Issues**

**Problem:** Kernel not starting or crashes
```bash
Dead kernel
```

**Solution:** Restart kernel and reinstall if needed
```bash
jupyter kernelspec list
python3 -m ipykernel install --user --name=cmsc173
# Restart Jupyter and select the cmsc173 kernel
```

**Problem:** Notebook cells don't execute
```bash
# Clear all outputs and restart kernel
# In Jupyter: Kernel → Restart & Clear Output
# Then run cells sequentially
```

#### 4. **Missing Figures**

**Problem:** LaTeX compilation shows missing figure warnings

**Solution:** Regenerate all figures
```bash
cd scripts/
python3 generate_all_figures.py
# Verify figures exist
ls -la ../figures/*.png
```

### Performance Notes

- **LaTeX Compilation:** ~10-15 seconds for full build
- **Figure Generation:** ~3-5 seconds for all 13 figures
- **Notebook Execution:** ~2-3 minutes for complete run

### Quality Metrics

**Current Status (✅ Meets Standards):**
- ✅ LaTeX compiles successfully (0 errors)
- ✅ Overfull boxes: 18 (<30 acceptable threshold)
- ✅ All Python scripts execute without errors
- ✅ All 13 figures generated correctly
- ✅ Jupyter notebook structure complete
- ✅ PDF readable and professional (2.8MB, 46 slides)

### Contact Information
For questions about this educational package:
- Course: CMSC 173 - Machine Learning
- Topics: Neural Networks, Deep Learning
- Components: Slides, Notebooks, Visualizations

---

**🎉 Ready to explore the fascinating world of neural networks!**

*This package provides a complete foundation for understanding artificial neural networks through theory, visualization, and hands-on implementation.*