#!/usr/bin/env python3
"""
Update Module 12 HTML with comprehensive Neural Networks content from LaTeX.
"""

import json
from pathlib import Path

# Comprehensive slide data extracted from neural_networks_slides.tex
SLIDES_DATA = [
    {
        "title": "What Are Neural Networks?",
        "content": """<h3>Artificial Neural Networks: Computing systems inspired by biological neural networks</h3>
<h4>Biological Inspiration:</h4>
<ul>
    <li><strong>Neurons:</strong> Basic processing units</li>
    <li><strong>Synapses:</strong> Weighted connections</li>
    <li><strong>Learning:</strong> Adapting connection strengths</li>
    <li><strong>Parallel processing:</strong> Massive connectivity</li>
</ul>
<h4>Artificial Counterpart:</h4>
<ul>
    <li><strong>Perceptrons:</strong> Mathematical neurons</li>
    <li><strong>Weights:</strong> Learnable parameters</li>
    <li><strong>Training:</strong> Gradient-based optimization</li>
    <li><strong>Layers:</strong> Organized processing units</li>
</ul>
<div class="highlight">
    <strong>Key Insight:</strong> Neural networks can learn complex non-linear mappings from data by adjusting weights through training.
</div>"""
    },
    {
        "title": "Why Neural Networks?",
        "content": """<h4>Motivation: Limitations of Linear Models</h4>
<h5>Linear Models:</h5>
<ul>
    <li>Limited to linear decision boundaries</li>
    <li>Cannot solve XOR problem</li>
    <li>Restricted representational power</li>
    <li>Simple but insufficient for complex data</li>
</ul>
<h5>Neural Networks:</h5>
<ul>
    <li>Non-linear decision boundaries</li>
    <li>Universal approximation capability</li>
    <li>Hierarchical feature learning</li>
    <li>Scalable to complex problems</li>
</ul>
<div class="definition">
    <strong>Universal Approximation Theorem:</strong> A neural network with a single hidden layer can approximate any continuous function to arbitrary accuracy (given sufficient neurons).
</div>
<h4>Key Advantages:</h4>
<ul>
    <li>Automatic feature extraction</li>
    <li>End-to-end learning</li>
    <li>Flexible architectures</li>
</ul>"""
    },
    {
        "title": "The Perceptron: Building Block",
        "content": """<h4>Mathematical Model:</h4>
<ul>
    <li><strong>Linear Combination:</strong> z = Σ(w<sub>i</sub> · x<sub>i</sub>) + b = w<sup>T</sup>x + b</li>
    <li><strong>Activation:</strong> y = σ(z) where σ is an activation function</li>
</ul>
<h4>Components:</h4>
<ul>
    <li>Inputs (x₁, x₂, ..., x<sub>n</sub>)</li>
    <li>Weights (w₁, w₂, ..., w<sub>n</sub>)</li>
    <li>Bias term (b)</li>
    <li>Summation function</li>
    <li>Activation function (σ)</li>
    <li>Output (y)</li>
</ul>
<div class="highlight">
    <strong>Foundation:</strong> The perceptron is the fundamental building block of all neural networks.
</div>"""
    },
    {
        "title": "Neural Network Components",
        "content": """<h4>Single Processing Unit:</h4>
<ul>
    <li>Multiple inputs with associated weights</li>
    <li>Bias term</li>
    <li>Summation of weighted inputs</li>
    <li>Activation function</li>
    <li>Single output</li>
</ul>
<h4>Multi-Layer Perceptron:</h4>
<ul>
    <li>Input layer</li>
    <li>Hidden layers (one or more)</li>
    <li>Output layer</li>
    <li>Fully connected layers</li>
    <li>Each connection represents a learnable weight</li>
</ul>
<h4>Key Concepts:</h4>
<ul>
    <li><strong>Processing Unit:</strong> z = Σ(w<sub>i</sub>·x<sub>i</sub>) + w₀, then y = σ(z)</li>
    <li><strong>Network:</strong> Multiple units arranged in layers with feedforward connections</li>
</ul>"""
    },
    {
        "title": "Perceptron: Mathematical Formulation",
        "content": """<h4>Complete Mathematical Description:</h4>
<p>z = Σ(w<sub>i</sub>·x<sub>i</sub>) + b = w<sup>T</sup>x + b</p>
<p>y = σ(z) = σ(w<sup>T</sup>x + b)</p>
<p><strong>where:</strong></p>
<ul>
    <li>x = [x₁, x₂, ..., x<sub>n</sub>]<sup>T</sup>: input vector</li>
    <li>w = [w₁, w₂, ..., w<sub>n</sub>]<sup>T</sup>: weight vector</li>
    <li>b: bias term</li>
    <li>σ: activation function</li>
</ul>
<h4>Step Function (Original):</h4>
<p>σ(z) = 1 if z ≥ 0, 0 if z < 0</p>
<p><strong>Problem:</strong> Not differentiable</p>
<h4>Sigmoid Function (Modern):</h4>
<p>σ(z) = 1 / (1 + e<sup>-z</sup>)</p>
<p><strong>Advantage:</strong> Smooth and differentiable</p>"""
    },
    {
        "title": "Perceptron Learning Algorithm",
        "content": """<h4>Goal: Learn weights w and bias b</h4>
<p>Minimize prediction error through optimization</p>
<h4>Original Perceptron Rule:</h4>
<p><strong>For misclassified point (x<sub>i</sub>, y<sub>i</sub>):</strong></p>
<ul>
    <li>w<sub>j</sub> := w<sub>j</sub> + α · (y<sub>i</sub> - ŷ<sub>i</sub>) · x<sub>ij</sub></li>
    <li>b := b + α · (y<sub>i</sub> - ŷ<sub>i</sub>)</li>
</ul>
<p>where α is the learning rate</p>
<p><strong>Convergence:</strong> Guaranteed for linearly separable data</p>
<h4>Gradient Descent (Modern):</h4>
<p><strong>Define loss function:</strong> L = ½(y - ŷ)²</p>
<p><strong>Weight updates:</strong></p>
<ul>
    <li>w<sub>j</sub> := w<sub>j</sub> - α · dL/dw<sub>j</sub></li>
    <li>w<sub>j</sub> := w<sub>j</sub> - α · (y - ŷ) · σ'(z) · x<sub>j</sub></li>
    <li>b := b - α · (y - ŷ) · σ'(z)</li>
</ul>
<div class="warning">
    <strong>Limitation:</strong> Single perceptron can only learn linearly separable functions. Solution: Multi-layer networks!
</div>"""
    },
    {
        "title": "Activation Functions: Non-linearity",
        "content": """<h4>Purpose of Activation Functions:</h4>
<div class="definition">
    <strong>Activation functions introduce non-linearity</strong> into the network, enabling it to learn complex patterns.
</div>
<h4>Common Activation Functions:</h4>
<ul>
    <li>Sigmoid</li>
    <li>Hyperbolic tangent (tanh)</li>
    <li>ReLU (Rectified Linear Unit)</li>
    <li>Leaky ReLU</li>
    <li>ELU (Exponential Linear Unit)</li>
</ul>
<h4>Why Non-linearity Matters:</h4>
<ul>
    <li>Without activation: stacking layers = one linear transformation</li>
    <li>With activation: layers can learn hierarchical features</li>
    <li>Enable learning of complex decision boundaries</li>
</ul>"""
    },
    {
        "title": "Sigmoid Activation",
        "content": """<h4>Sigmoid Function:</h4>
<p>σ(x) = 1 / (1 + e<sup>-x</sup>)</p>
<h4>Properties:</h4>
<ul>
    <li><strong>Range:</strong> (0, 1)</li>
    <li><strong>Smooth and differentiable:</strong> Good for gradient descent</li>
    <li><strong>Output interpretable as probability:</strong> Useful for classification</li>
</ul>
<h4>Derivative:</h4>
<p>σ'(x) = σ(x)(1 - σ(x))</p>
<h4>Issues:</h4>
<ul>
    <li><strong>Vanishing gradients:</strong> Gradients approach 0 for large |x|</li>
    <li><strong>Not zero-centered:</strong> Output always positive</li>
    <li>Expensive computation (exponential)</li>
</ul>"""
    },
    {
        "title": "Tanh and ReLU",
        "content": """<h4>Hyperbolic Tangent (tanh):</h4>
<p>tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)</p>
<ul>
    <li><strong>Range:</strong> (-1, 1) - zero-centered!</li>
    <li><strong>Steeper gradients:</strong> Better than sigmoid</li>
    <li><strong>Derivative:</strong> tanh'(x) = 1 - tanh²(x)</li>
</ul>
<h4>ReLU (Rectified Linear Unit):</h4>
<p>ReLU(x) = max(0, x)</p>
<h4>Advantages of ReLU:</h4>
<ul>
    <li>Computationally efficient (simple thresholding)</li>
    <li>No vanishing gradient for x > 0</li>
    <li>Sparse activation (many neurons off)</li>
    <li>Most popular choice in modern networks</li>
</ul>
<p><strong>Derivative:</strong> ReLU'(x) = 1 if x > 0, 0 if x ≤ 0</p>"""
    },
    {
        "title": "Leaky ReLU and Variants",
        "content": """<h4>Leaky ReLU:</h4>
<p>LeakyReLU(x) = x if x > 0, α·x if x ≤ 0</p>
<h4>Advantages:</h4>
<ul>
    <li>Avoids "dying ReLU" problem</li>
    <li>Small gradient for negative inputs</li>
    <li>Typically α = 0.01</li>
    <li>Better gradient flow during backpropagation</li>
</ul>
<p><strong>Derivative:</strong> LeakyReLU'(x) = 1 if x > 0, α if x ≤ 0</p>
<h4>When to Use:</h4>
<ul>
    <li><strong>ReLU:</strong> Default choice for hidden layers</li>
    <li><strong>Leaky ReLU:</strong> If dying ReLU is a problem</li>
    <li><strong>Sigmoid:</strong> Output layer for binary classification</li>
    <li><strong>Tanh:</strong> For zero-centered data or regression</li>
</ul>"""
    },
    {
        "title": "Multi-Layer Network Architecture",
        "content": """<h4>Key Components:</h4>
<ul>
    <li><strong>Layers:</strong> Input → Hidden → Hidden → ... → Output</li>
    <li><strong>Connections:</strong> Each neuron connects to all neurons in next layer (fully connected)</li>
</ul>
<h4>Example Architecture:</h4>
<ul>
    <li>Input layer: 784 neurons (features)</li>
    <li>Hidden layer 1: 128 neurons</li>
    <li>Hidden layer 2: 64 neurons</li>
    <li>Output layer: 10 neurons (classes)</li>
</ul>
<h4>Weight matrices and bias vectors:</h4>
<ul>
    <li>W⁽¹⁾, b⁽¹⁾ between input and hidden1</li>
    <li>W⁽²⁾, b⁽²⁾ between hidden1 and hidden2</li>
    <li>W⁽³⁾, b⁽³⁾ between hidden2 and output</li>
</ul>
<h4>Information Flow:</h4>
<p><strong>Forward Propagation:</strong> Information flows from input to output through the network</p>"""
    },
    {
        "title": "Network Math: Forward Pass",
        "content": """<h4>For a network with L layers:</h4>
<ul>
    <li>a⁽⁰⁾ = x (input layer)</li>
    <li>z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾ for l = 1, 2, ..., L</li>
    <li>a⁽ˡ⁾ = σ⁽ˡ⁾(z⁽ˡ⁾) for l = 1, 2, ..., L</li>
    <li>ŷ = a⁽ᴸ⁾ (output layer)</li>
</ul>
<h4>where:</h4>
<ul>
    <li>W⁽ˡ⁾ ∈ ℝ⁽ⁿˡ ˣ ⁿˡ⁻¹⁾: weight matrix for layer l</li>
    <li>b⁽ˡ⁾ ∈ ℝⁿˡ: bias vector for layer l</li>
    <li>n_l: number of neurons in layer l</li>
    <li>σ⁽ˡ⁾: activation function for layer l</li>
</ul>"""
    },
    {
        "title": "Network Dimensions",
        "content": """<h4>Matrix Dimensions for Layer l:</h4>
<ul>
    <li><strong>Input:</strong> a⁽ˡ⁻¹⁾ has shape (n_(l-1), 1)</li>
    <li><strong>Weights:</strong> W⁽ˡ⁾ has shape (n_l, n_(l-1))</li>
    <li><strong>Output:</strong> a⁽ˡ⁾ has shape (n_l, 1)</li>
</ul>
<h4>Batch Processing:</h4>
<ul>
    <li><strong>Input batch:</strong> A⁽ˡ⁻¹⁾ has shape (n_(l-1), m)</li>
    <li><strong>Output batch:</strong> A⁽ˡ⁾ has shape (n_l, m)</li>
    <li>where m is the batch size</li>
</ul>
<h4>Parameter Count Example:</h4>
<p><strong>Network: 784 → 128 → 64 → 10</strong></p>
<p>Total = 784×128 + 128 + 128×64 + 64 + 64×10 + 10 = <strong>109,386 parameters</strong></p>
<h4>Memory Considerations:</h4>
<ul>
    <li>Scales with network depth</li>
    <li>Scales with layer width</li>
    <li>Scales with batch size</li>
</ul>"""
    },
    {
        "title": "Network Design Considerations",
        "content": """<h4>Depth vs Width:</h4>
<h5>Deeper Networks:</h5>
<ul>
    <li>More layers, fewer neurons per layer</li>
    <li>Better feature hierarchies</li>
    <li>Can represent more complex functions</li>
    <li><strong>Risk:</strong> Vanishing gradients</li>
</ul>
<h5>Wider Networks:</h5>
<ul>
    <li>Fewer layers, more neurons per layer</li>
    <li>More parameters at each level</li>
    <li>Easier to train</li>
    <li><strong>Risk:</strong> Overfitting</li>
</ul>
<h4>Architecture Guidelines:</h4>
<ul>
    <li><strong>Hidden Layer Size:</strong> Start with size between input and output dimensions</li>
    <li><strong>Rule of thumb:</strong> sqrt(n_input × n_output)</li>
    <li><strong>Number of Layers:</strong> Simple problems: 1-2, Complex: 3+, Very deep: Special techniques needed</li>
</ul>
<div class="highlight">
    <strong>Best Practice:</strong> Start simple and gradually increase complexity. Use validation performance to guide architecture choices.
</div>"""
    },
    {
        "title": "Forward Propagation: Information Flow",
        "content": """<h4>Forward Pass: Information flows from input to output</h4>
<h4>Step-by-step Process:</h4>
<ol>
    <li>Input layer receives features (x)</li>
    <li>Hidden layer 1 computes: z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾, then a⁽¹⁾ = σ(z⁽¹⁾)</li>
    <li>Hidden layer 2 computes: z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾, then a⁽²⁾ = σ(z⁽²⁾)</li>
    <li>Output layer produces prediction (ŷ)</li>
</ol>
<h4>Mathematical Formulation:</h4>
<ul>
    <li>z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾</li>
    <li>a⁽¹⁾ = σ(z⁽¹⁾)</li>
    <li>z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾</li>
    <li>a⁽²⁾ = σ(z⁽²⁾) = ŷ</li>
</ul>"""
    },
    {
        "title": "Forward Propagation Algorithm",
        "content": """<h4>Step-by-step Algorithm:</h4>
<p><strong>Input:</strong> x, weights {W⁽ˡ⁾}, biases {b⁽ˡ⁾}</p>
<ol>
    <li>Set a⁽⁰⁾ = x</li>
    <li>For l = 1 to L:
        <ul>
            <li>Compute pre-activation: z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾</li>
            <li>Apply activation: a⁽ˡ⁾ = σ⁽ˡ⁾(z⁽ˡ⁾)</li>
        </ul>
    </li>
    <li>Output: ŷ = a⁽ᴸ⁾</li>
</ol>
<h4>Vectorized Implementation (Batch):</h4>
<ul>
    <li>Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾ W⁽ˡ⁾ᵀ + b⁽ˡ⁾</li>
    <li>A⁽ˡ⁾ = σ⁽ˡ⁾(Z⁽ˡ⁾)</li>
    <li>where A⁽ˡ⁾ has shape (m, n_l) for m examples</li>
</ul>
<h4>Computational Complexity:</h4>
<p>O(L × N × M) where L = layers, N = max neurons, M = batch size</p>
<p><strong>Key:</strong> All intermediate values z⁽ˡ⁾, a⁽ˡ⁾ are stored for backpropagation</p>"""
    },
    {
        "title": "Backpropagation: Core Algorithm",
        "content": """<h4>Backpropagation: Efficient gradient computation via chain rule</h4>
<h4>Algorithm Overview:</h4>
<ol>
    <li><strong>Forward Pass:</strong> Compute all a⁽ˡ⁾ and z⁽ˡ⁾ (store them!)</li>
    <li><strong>Compute Output Error:</strong> δ⁽ᴸ⁾ = dL/da⁽ᴸ⁾ ⊙ σ'(z⁽ᴸ⁾)</li>
    <li><strong>For l = L-1 down to 1:</strong>
        <ul>
            <li>Propagate Error: δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)</li>
        </ul>
    </li>
    <li><strong>For l = 1 to L (compute gradients):</strong>
        <ul>
            <li>dL/dW⁽ˡ⁾ = δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ</li>
            <li>dL/db⁽ˡ⁾ = δ⁽ˡ⁾</li>
        </ul>
    </li>
</ol>
<h4>Computational Complexity:</h4>
<p><strong>Time:</strong> O(number of weights) - same as forward pass!</p>
<p><strong>Space:</strong> O(network size) - must store all activations</p>"""
    },
    {
        "title": "Backpropagation: Chain Rule",
        "content": """<h4>Core Idea: Chain Rule Application</h4>
<h4>Goal:</h4>
<p>Compute dL/dW⁽ˡ⁾ and dL/db⁽ˡ⁾ for all layers</p>
<h4>Chain Rule:</h4>
<ul>
    <li>dL/dW⁽ˡ⁾ = dL/dz⁽ˡ⁾ · dz⁽ˡ⁾/dW⁽ˡ⁾</li>
    <li>dL/db⁽ˡ⁾ = dL/dz⁽ˡ⁾ · dz⁽ˡ⁾/db⁽ˡ⁾</li>
    <li>dL/da⁽ˡ⁻¹⁾ = dL/dz⁽ˡ⁾ · dz⁽ˡ⁾/da⁽ˡ⁻¹⁾</li>
</ul>
<h4>Define Error Terms:</h4>
<p>δ⁽ˡ⁾ = dL/dz⁽ˡ⁾ (error at pre-activation)</p>
<h4>Key Equations:</h4>
<ul>
    <li>dL/dW⁽ˡ⁾ = δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ</li>
    <li>dL/db⁽ˡ⁾ = δ⁽ˡ⁾</li>
    <li>δ⁽ˡ⁻¹⁾ = (W⁽ˡ⁾)ᵀ δ⁽ˡ⁾ ⊙ σ'(z⁽ˡ⁻¹⁾)</li>
</ul>
<p>⊙ denotes element-wise multiplication</p>"""
    },
    {
        "title": "Loss Function and Gradients",
        "content": """<h4>Output Layer Error Computation:</h4>
<h5>For output layer L:</h5>
<p>δ⁽ᴸ⁾ = dL/da⁽ᴸ⁾ ⊙ σ'(z⁽ᴸ⁾)</p>
<h4>Common Case: MSE Loss + Sigmoid</h4>
<p><strong>Loss:</strong> L = ½(a⁽ᴸ⁾ - y)²</p>
<p><strong>Gradient:</strong> dL/da⁽ᴸ⁾ = a⁽ᴸ⁾ - y</p>
<p><strong>Sigmoid derivative:</strong> σ'(z) = a(1 - a)</p>
<p><strong>Combined:</strong> δ⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ a⁽ᴸ⁾ ⊙ (1 - a⁽ᴸ⁾)</p>
<h4>Hidden Layer Errors:</h4>
<p>δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ σ'(z⁽ˡ⁾)</p>
<h4>Weight and Bias Updates:</h4>
<ul>
    <li>W⁽ˡ⁾ := W⁽ˡ⁾ - α · dL/dW⁽ˡ⁾</li>
    <li>b⁽ˡ⁾ := b⁽ˡ⁾ - α · dL/db⁽ˡ⁾</li>
</ul>
<p>where α is the learning rate</p>"""
    },
    {
        "title": "Why Backpropagation Works",
        "content": """<h4>Key Advantages:</h4>
<ul>
    <li><strong>Efficiency:</strong> Time complexity = forward pass, not exponential in network depth</li>
    <li><strong>Reuses Computations:</strong> Uses chain rule to efficiently compute gradients</li>
    <li><strong>Automatic:</strong> No manual gradient derivation needed</li>
    <li><strong>Exact:</strong> Computes exact gradients (not numerical approximations)</li>
    <li><strong>General:</strong> Works for any differentiable network</li>
</ul>
<h4>Historical Impact:</h4>
<ul>
    <li><strong>Authors:</strong> Rumelhart, Hinton, Williams (1986)</li>
    <li><strong>Significance:</strong> Made deep learning practical</li>
    <li><strong>Modern:</strong> Foundation of all deep learning frameworks</li>
</ul>
<div class="highlight">
    <strong>Without backpropagation:</strong> Computing gradients for large networks would be computationally infeasible. This algorithm is crucial for training neural networks.
</div>"""
    },
    {
        "title": "Overfitting Problem",
        "content": """<h4>Overfitting: Model learns training data too well</h4>
<h4>Symptoms:</h4>
<ul>
    <li>High training accuracy, low validation accuracy</li>
    <li>Model performs well on training data but poorly on new data</li>
    <li>Complex decision boundaries that fit noise</li>
    <li>Large gap between train and validation metrics</li>
</ul>
<h4>Root Causes:</h4>
<ul>
    <li>Too many parameters relative to training data</li>
    <li>Training for too long (training loss still decreasing)</li>
    <li>Insufficient regularization</li>
    <li>Model complexity exceeds problem complexity</li>
</ul>
<h4>Solutions:</h4>
<ul>
    <li>Regularization techniques (L1, L2, Dropout)</li>
    <li>More training data</li>
    <li>Simpler model architecture</li>
    <li>Early stopping</li>
    <li>Data augmentation</li>
</ul>"""
    },
    {
        "title": "L2 Regularization (Ridge)",
        "content": """<h4>Add Penalty Term to Loss Function:</h4>
<p>L_total = L_data + λ · Σ over l of ||W⁽ˡ⁾||₂²</p>
<p>where ||W⁽ˡ⁾||₂² = Σᵢ Σⱼ (W_ij⁽ˡ⁾)²</p>
<h4>Effect of L2 Regularization:</h4>
<ul>
    <li>Shrinks weights towards zero</li>
    <li>Uniform penalty on all weights</li>
    <li>Smooth weight distributions</li>
    <li>Prevents very large weights</li>
</ul>
<h4>Gradient Modification:</h4>
<p>dL_total/dW⁽ˡ⁾ = dL_data/dW⁽ˡ⁾ + 2λ·W⁽ˡ⁾</li>
<h4>When to Use L2:</h4>
<ul>
    <li>General-purpose regularization</li>
    <li>All features potentially relevant</li>
    <li>Want smooth weight shrinkage</li>
    <li>Most common choice</li>
</ul>
<h4>Typical Values:</h4>
<p>λ = 0.0001 to 0.01</p>"""
    },
    {
        "title": "L1 Regularization (Lasso)",
        "content": """<h4>Add L1 Penalty to Loss Function:</h4>
<p>L_total = L_data + λ · Σ over l of ||W⁽ˡ⁾||₁</p>
<p>where ||W⁽ˡ⁾||₁ = Σᵢ Σⱼ |W_ij⁽ˡ⁾|</p>
<h4>Effect of L1 Regularization:</h4>
<ul>
    <li><strong>Promotes sparsity:</strong> Many weights become exactly zero</li>
    <li><strong>Feature selection:</strong> Automatic elimination of irrelevant features</li>
    <li><strong>Sparse networks:</strong> Reduced parameter count</li>
    <li><strong>Interpretability:</strong> Clear feature importance</li>
</ul>
<h4>Gradient Modification:</h4>
<p>dL_total/dW⁽ˡ⁾ = dL_data/dW⁽ˡ⁾ + λ·sign(W⁽ˡ⁾)</p>
<h4>When to Use L1:</h4>
<ul>
    <li>Feature selection needed</li>
    <li>Many irrelevant features</li>
    <li>Want sparse models</li>
    <li>Interpretability important</li>
</ul>"""
    },
    {
        "title": "Dropout Regularization",
        "content": """<h4>Dropout Technique: Randomly deactivate neurons during training</h4>
<h4>Key Idea:</h4>
<p>Randomly set neurons to zero during training to prevent co-adaptation and improve generalization.</p>
<h4>Training Phase (with Dropout):</h4>
<ul>
    <li>Randomly drop neurons with probability p (typically 0.5)</li>
    <li>Only active connections used for forward/backward pass</li>
    <li>Creates ensemble effect</li>
    <li>Different network on each iteration</li>
</ul>
<h4>Testing Phase (no Dropout):</h4>
<ul>
    <li>All neurons active</li>
    <li>Weights scaled to maintain expected activation levels</li>
    <li>No randomness during inference</li>
    <li>Deterministic predictions</li>
</ul>
<h4>Typical Dropout Rates:</h4>
<ul>
    <li>Hidden layers: 0.2 - 0.5</li>
    <li>Input layer: 0.1 - 0.2</li>
    <li>Output layer: No dropout</li>
</ul>"""
    },
    {
        "title": "Dropout Mathematics",
        "content": """<h4>Training Phase:</h4>
<ul>
    <li>r⁽ˡ⁾ ~ Bernoulli(p) (dropout mask)</li>
    <li>ã⁽ˡ⁾ = r⁽ˡ⁾ ⊙ a⁽ˡ⁾ (apply mask)</li>
    <li>z⁽ˡ⁺¹⁾ = W⁽ˡ⁺¹⁾ ã⁽ˡ⁾ + b⁽ˡ⁺¹⁾</li>
</ul>
<h4>Testing Phase:</h4>
<ul>
    <li>z⁽ˡ⁺¹⁾ = p · W⁽ˡ⁺¹⁾ a⁽ˡ⁾ + b⁽ˡ⁺¹⁾ (scale weights)</li>
</ul>
<h4>Benefits of Dropout:</h4>
<ul>
    <li><strong>Prevents Overfitting:</strong> Reduces complex co-adaptations</li>
    <li><strong>Model Averaging:</strong> Approximates ensemble of networks</li>
    <li><strong>Robust Features:</strong> Forces redundant representations</li>
    <li><strong>Easy to Implement:</strong> Simple modification to forward pass</li>
</ul>
<h4>Why Scaling Works:</h4>
<ul>
    <li>Training: Each neuron is "on" with probability p</li>
    <li>Testing: All neurons are "on"</li>
    <li>Scaling by p maintains expected activation levels</li>
</ul>"""
    },
    {
        "title": "Weight Initialization",
        "content": """<h4>Proper Initialization is Critical for Training Success</h4>
<h4>Poor Initialization Problems:</h4>
<h5>All Zeros:</h5>
<p>W_ij = 0 ⟹ No gradient flow, no learning</p>
<h5>Too Large:</h5>
<p>W_ij ~ N(0, 1) ⟹ Activation saturation, vanishing gradients</p>
<h5>Too Small:</h5>
<p>W_ij ~ N(0, 0.01) ⟹ Weak signals, slow learning</p>
<h4>Good Initialization Schemes:</h4>
<h5>Xavier/Glorot (for Sigmoid/Tanh):</h5>
<p>W_ij ~ N(0, √(2/(n_in + n_out)))</p>
<h5>He Initialization (for ReLU):</h5>
<p>W_ij ~ N(0, √(2/n_in))</p>
<h4>Bias Initialization:</h4>
<p>b_i = 0 (usually sufficient)</p>
<div class="highlight">
    <strong>Why These Work:</strong> Maintain activation variance and gradient variance across layers during initialization.
</div>"""
    },
    {
        "title": "Learning Rate Selection",
        "content": """<h4>Learning Rate α: Critical Hyperparameter</h4>
<h4>Effects of Learning Rate:</h4>
<h5>Too High:</h5>
<ul>
    <li>Overshooting: Jumps over minima</li>
    <li>Instability: Loss oscillates or explodes</li>
    <li>Divergence: Network doesn't converge</li>
    <li>Large weights: Values become unreasonable</li>
</ul>
<h5>Too Low:</h5>
<ul>
    <li>Slow convergence: Training takes forever</li>
    <li>Gets stuck: Local minima, saddle points</li>
    <li>Poor performance: Suboptimal solutions</li>
    <li>Inefficient: Wasted computation</li>
</ul>
<h4>Typical Good Range:</h4>
<p>10⁻⁴ to 10⁻¹ (depends on optimizer)</p>
<h4>Rule of Thumb:</h4>
<ul>
    <li>SGD: 0.01 - 0.1</li>
    <li>Adam: 0.001 - 0.01 (default 0.001)</li>
</ul>"""
    },
    {
        "title": "Advanced Optimizers",
        "content": """<h4>Beyond Standard Gradient Descent</h4>
<h4>SGD with Momentum:</h4>
<p>v_t = β·v_(t-1) + (1-β)·∇L</p>
<p>W := W - α·v_t</p>
<ul>
    <li>Accumulates gradients over time</li>
    <li>Smooths out noisy gradients</li>
    <li>Faster convergence</li>
</ul>
<h4>Adam (Adaptive Moments):</h4>
<p>m_t = β₁·m_(t-1) + (1-β₁)·∇L</p>
<p>v_t = β₂·v_(t-1) + (1-β₂)·(∇L)²</p>
<p>W := W - α·m_t/(√(v_t) + ε)</p>
<h4>Adam Advantages:</h4>
<ul>
    <li>Adaptive per-parameter learning rates</li>
    <li>Handles sparse gradients well</li>
    <li>Less sensitive to learning rate</li>
    <li>Default choice in most frameworks</li>
</ul>
<h4>Default Settings:</h4>
<ul>
    <li>Optimizer: Adam</li>
    <li>Learning rate: 0.001</li>
    <li>β₁: 0.9, β₂: 0.999</li>
</ul>"""
    },
    {
        "title": "Training Diagnostics",
        "content": """<h4>Monitor During Training:</h4>
<h4>Loss Curves:</h4>
<ul>
    <li><strong>Training loss:</strong> Should decrease monotonically</li>
    <li><strong>Validation loss:</strong> Should decrease, then stabilize</li>
    <li><strong>Gap:</strong> Large gap indicates overfitting</li>
</ul>
<h4>Warning Signs:</h4>
<ul>
    <li><strong>Loss increases:</strong> Learning rate too high</li>
    <li><strong>Loss plateaus early:</strong> Learning rate too low</li>
    <li><strong>Validation loss increases:</strong> Overfitting</li>
    <li><strong>Loss becomes NaN:</strong> Gradient explosion</li>
</ul>
<h4>Gradient Monitoring:</h4>
<ul>
    <li><strong>Gradient norms:</strong> Should be reasonable (10⁻⁶ to 10⁻¹)</li>
    <li><strong>Vanishing:</strong> Gradients → 0 in early layers</li>
    <li><strong>Exploding:</strong> Gradients become very large</li>
</ul>
<h4>Health Check:</h4>
<ul>
    <li>Is training loss decreasing?</li>
    <li>Are gradients reasonable?</li>
    <li>Are weights updating appropriately?</li>
    <li>Is validation performance improving?</li>
</ul>"""
    },
    {
        "title": "Common Training Problems",
        "content": """<h4>Problem: Vanishing Gradients</h4>
<p><strong>Symptoms:</strong> Early layers don't learn, gradients → 0</p>
<p><strong>Solutions:</strong></p>
<ul>
    <li>Use ReLU activations</li>
    <li>Proper weight initialization</li>
    <li>Batch normalization</li>
    <li>Residual connections</li>
</ul>
<h4>Problem: Overfitting</h4>
<p><strong>Symptoms:</strong> Training accuracy >> validation accuracy</p>
<p><strong>Solutions:</strong></p>
<ul>
    <li>Add regularization (L2, dropout)</li>
    <li>Reduce model complexity</li>
    <li>More training data</li>
    <li>Early stopping</li>
</ul>
<h4>Problem: Exploding Gradients</h4>
<p><strong>Symptoms:</strong> Loss becomes NaN, weights blow up</p>
<p><strong>Solutions:</strong></p>
<ul>
    <li>Gradient clipping</li>
    <li>Lower learning rate</li>
    <li>Better initialization</li>
</ul>
<h4>Problem: Slow Convergence</h4>
<p><strong>Symptoms:</strong> Loss decreases slowly, plateaus</p>
<p><strong>Solutions:</strong></p>
<ul>
    <li>Increase learning rate</li>
    <li>Use adaptive optimizers (Adam)</li>
    <li>Batch normalization</li>
</ul>"""
    },
    {
        "title": "Key Takeaways",
        "content": """<h4>Core Concepts:</h4>
<ul>
    <li><strong>Perceptron:</strong> Basic building block</li>
    <li><strong>Multi-layer:</strong> Enable complex mappings</li>
    <li><strong>Activation functions:</strong> Provide non-linearity</li>
    <li><strong>Forward propagation:</strong> Compute predictions</li>
    <li><strong>Backpropagation:</strong> Compute gradients efficiently</li>
    <li><strong>Regularization:</strong> Prevent overfitting</li>
</ul>
<h4>Mathematical Foundation:</h4>
<ul>
    <li>Matrix operations for efficiency</li>
    <li>Chain rule for gradient computation</li>
    <li>Optimization theory for training</li>
    <li>Probability theory for interpretation</li>
</ul>
<h4>Best Practices:</h4>
<ul>
    <li><strong>Architecture:</strong> Start simple, add complexity gradually</li>
    <li><strong>Initialization:</strong> Xavier/He for proper gradient flow</li>
    <li><strong>Optimization:</strong> Adam optimizer with proper learning rate</li>
    <li><strong>Regularization:</strong> L2 + Dropout for generalization</li>
    <li><strong>Monitoring:</strong> Track loss, gradients, activations</li>
</ul>
<div class="highlight">
    <strong>Foundation:</strong> These fundamentals scale to modern architectures: CNNs, RNNs, Transformers, ResNets, etc.
</div>"""
    },
    {
        "title": "Real-World Applications",
        "content": """<h4>Computer Vision:</h4>
<ul>
    <li><strong>Image classification:</strong> ResNet, EfficientNet, Vision Transformer</li>
    <li><strong>Object detection:</strong> YOLO, R-CNN, RetinaNet</li>
    <li><strong>Segmentation:</strong> U-Net, Mask R-CNN, DeepLab</li>
    <li><strong>Face recognition:</strong> DeepFace, FaceNet, ArcFace</li>
    <li><strong>Medical imaging:</strong> Cancer detection, radiology AI</li>
</ul>
<h4>Natural Language Processing:</h4>
<ul>
    <li><strong>Language models:</strong> GPT, BERT, T5, LLaMA</li>
    <li><strong>Translation:</strong> Google Translate, DeepL</li>
    <li><strong>Chatbots:</strong> ChatGPT, Claude, Bard</li>
    <li><strong>Text analysis:</strong> Sentiment, summarization, QA</li>
</ul>
<h4>Other Domains:</h4>
<ul>
    <li><strong>Speech:</strong> Recognition, synthesis, processing</li>
    <li><strong>Recommendation:</strong> Netflix, Amazon, Spotify</li>
    <li><strong>Games:</strong> AlphaGo, OpenAI Five, StarCraft AI</li>
    <li><strong>Finance:</strong> Trading, fraud detection, risk</li>
</ul>
<div class="highlight">
    <strong>Impact:</strong> Neural networks have revolutionized AI and are now fundamental to most modern machine learning applications.
</div>"""
    },
    {
        "title": "Beyond Fundamentals",
        "content": """<h4>Specialized Architectures:</h4>
<h5>Convolutional Neural Networks (CNNs):</h5>
<ul>
    <li>Exploit spatial structure in images</li>
    <li>Translation invariance through convolutions</li>
    <li>Significantly fewer parameters than fully connected</li>
</ul>
<h5>Recurrent Neural Networks (RNNs):</h5>
<ul>
    <li>Process sequential data with memory</li>
    <li>LSTM and GRU variants for long-term dependencies</li>
    <li>Natural for time series and language</li>
</ul>
<h5>Transformer Networks:</h5>
<ul>
    <li>Attention mechanisms for selective focus</li>
    <li>Parallel processing of sequences</li>
    <li>Modern NLP backbone (BERT, GPT, T5)</li>
</ul>
<h4>Advanced Techniques:</h4>
<ul>
    <li><strong>Batch Normalization:</strong> Stabilize and accelerate training</li>
    <li><strong>Residual Connections:</strong> Enable very deep networks</li>
    <li><strong>Attention Mechanisms:</strong> Long-range dependencies</li>
    <li><strong>Generative Models:</strong> VAEs, GANs, Diffusion models</li>
</ul>
<h4>Next Steps:</h4>
<ul>
    <li>Practice implementation with PyTorch/TensorFlow</li>
    <li>Experiment with real datasets</li>
    <li>Explore specialized architectures for your domain</li>
    <li>Study advanced topics like Transformers</li>
</ul>"""
    }
]

def create_module_12_html():
    """Create updated Module 12 HTML with comprehensive slide content."""

    # HTML template with styling and JavaScript
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Module 12: Neural Networks</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        html, body {{
            height: 100%;
            width: 100%;
        }}

        body {{
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #8B0000 100%);
            color: #333;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .presenter-header {{
            background: linear-gradient(135deg, #1B4332 0%, #8B0000 100%);
            color: white;
            padding: 15px 30px;
            border-bottom: 4px solid #FFD700;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
            gap: 20px;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex: 1;
        }}

        .home-button {{
            background: #FFD700;
            color: #1B4332;
            border: none;
            padding: 8px 16px;
            font-size: 0.9em;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .home-button:hover {{
            background: white;
            transform: scale(1.05);
        }}

        .presenter-header h2 {{
            font-size: 1.3em;
            font-weight: 600;
        }}

        .slide-counter {{
            font-size: 1em;
            color: #FFD700;
            font-weight: 600;
            white-space: nowrap;
            min-width: 80px;
            text-align: right;
        }}

        .slide-counter-label {{
            font-size: 0.85em;
            opacity: 0.9;
            display: block;
        }}

        .presentation-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            padding: 20px;
        }}

        .slide-viewer {{
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .slide-content {{
            flex: 1;
            padding: 40px;
            overflow-y: auto;
            background: white;
        }}

        .slide-content h2 {{
            color: #1B4332;
            font-size: 2.2em;
            margin-bottom: 25px;
            border-bottom: 4px solid #FFD700;
            padding-bottom: 15px;
            font-weight: 700;
        }}

        .slide-content h3 {{
            color: #8B0000;
            font-size: 1.6em;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }}

        .slide-content h4 {{
            color: #1B4332;
            font-size: 1.3em;
            margin-top: 20px;
            margin-bottom: 12px;
            font-weight: 600;
        }}

        .slide-content h5 {{
            color: #1B4332;
            font-size: 1.1em;
            margin-top: 15px;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .slide-content p {{
            margin-bottom: 15px;
            color: #555;
            font-size: 1.05em;
            line-height: 1.8;
        }}

        .slide-content ul {{
            list-style: none;
            margin: 15px 0 15px 20px;
        }}

        .slide-content ol {{
            margin: 15px 0 15px 30px;
            padding-left: 0;
        }}

        .slide-content li {{
            margin-bottom: 12px;
            color: #555;
            line-height: 1.6;
            padding-left: 15px;
            position: relative;
        }}

        .slide-content ul li:before {{
            content: "•";
            position: absolute;
            left: 0;
            color: #FFD700;
            font-weight: bold;
            font-size: 1.2em;
        }}

        .slide-content ol li {{
            padding-left: 25px;
        }}

        .definition {{
            background: linear-gradient(120deg, #E8F5E9 0%, #F1F8F6 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #1B4332;
        }}

        .definition strong {{
            color: #1B4332;
        }}

        .highlight {{
            background: linear-gradient(120deg, #FFF9E6 0%, #FFFDF2 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #FFD700;
        }}

        .highlight strong {{
            color: #8B0000;
        }}

        .warning {{
            background: linear-gradient(120deg, #FFEBEE 0%, #FFF5F7 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #8B0000;
        }}

        .warning strong {{
            color: #8B0000;
        }}

        .slide-footer {{
            background: #f5f5f5;
            padding: 15px 40px;
            border-top: 2px solid #E0E0E0;
            font-size: 0.9em;
            color: #888;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}

        .footer-left {{
            color: #1B4332;
            font-weight: 600;
        }}

        .footer-right {{
            color: #FFD700;
            font-weight: 600;
        }}

        .controls {{
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 20px;
            flex-shrink: 0;
        }}

        button {{
            background: linear-gradient(135deg, #1B4332 0%, #8B0000 100%);
            color: white;
            border: 2px solid #FFD700;
            padding: 12px 30px;
            font-size: 1em;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(27, 67, 50, 0.2);
        }}

        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(27, 67, 50, 0.3);
            color: #FFD700;
        }}

        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}

        .hidden {{
            display: none !important;
        }}

        /* Scrollbar styling */
        .slide-content::-webkit-scrollbar {{
            width: 8px;
        }}

        .slide-content::-webkit-scrollbar-track {{
            background: #f1f1f1;
        }}

        .slide-content::-webkit-scrollbar-thumb {{
            background: #1B4332;
            border-radius: 4px;
        }}

        .slide-content::-webkit-scrollbar-thumb:hover {{
            background: #8B0000;
        }}

        @media (max-width: 768px) {{
            .presenter-header {{
                flex-direction: column;
                gap: 10px;
            }}

            .slide-content {{
                padding: 25px;
            }}

            .slide-content h2 {{
                font-size: 1.8em;
            }}

            button {{
                padding: 10px 20px;
                font-size: 0.9em;
            }}
        }}
    </style>
</head>
<body>
    <div class="presenter-header">
        <div class="header-left">
            <a href="/" class="home-button">🏠 Home</a>
            <h2>Module 12: Neural Networks</h2>
        </div>
        <div class="slide-counter">
            <span class="slide-counter-label">Slide</span>
            <span id="current-slide">1</span> / <span id="total-slides">1</span>
        </div>
    </div>

    <div class="presentation-container">
        <div class="slide-viewer">
            <div class="slide-content" id="slide-content">
                <!-- Slides will be inserted here -->
            </div>
            <div class="slide-footer">
                <div class="footer-left">CMSC 173: Machine Learning</div>
                <div class="footer-right">University of the Philippines - Cebu</div>
            </div>
        </div>

        <div class="controls">
            <button id="prev-btn" onclick="previousSlide()">← Previous</button>
            <button id="next-btn" onclick="nextSlide()">Next →</button>
        </div>
    </div>

    <script>
        // Slide content data
        const slides = [
'''

    # Add all slides to JavaScript array
    for i, slide in enumerate(SLIDES_DATA):
        html += f'''            {{
                title: "{slide['title']}",
                content: `
                    {slide['content']}
                `
            }}{'' if i == len(SLIDES_DATA) - 1 else ','}
'''

    html += '''        ];

        let currentSlideIndex = 0;

        function renderSlide() {
            const slide = slides[currentSlideIndex];
            const contentDiv = document.getElementById('slide-content');

            contentDiv.innerHTML = `
                <h2>${slide.title}</h2>
                ${slide.content}
            `;

            // Update slide counter
            document.getElementById('current-slide').textContent = currentSlideIndex + 1;
            document.getElementById('total-slides').textContent = slides.length;

            // Update button states
            document.getElementById('prev-btn').disabled = currentSlideIndex === 0;
            document.getElementById('next-btn').disabled = currentSlideIndex === slides.length - 1;

            // Scroll to top of slide content
            contentDiv.scrollTop = 0;
        }

        function nextSlide() {
            if (currentSlideIndex < slides.length - 1) {
                currentSlideIndex++;
                renderSlide();
            }
        }

        function previousSlide() {
            if (currentSlideIndex > 0) {
                currentSlideIndex--;
                renderSlide();
            }
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') nextSlide();
            if (e.key === 'ArrowLeft') previousSlide();
        });

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('total-slides').textContent = slides.length;
            renderSlide();
        });
    </script>
</body>
</html>
'''

    return html


if __name__ == "__main__":
    templates_dir = Path("/Users/njpinton/projects/git/CMSC173/presenter_app/templates")
    module_12_path = templates_dir / "12-neural-networks.html"

    print(f"Generating Module 12 HTML with {len(SLIDES_DATA)} slides...")

    html_content = create_module_12_html()

    with open(module_12_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Successfully updated {module_12_path}")
    print(f"✓ Total slides: {len(SLIDES_DATA)}")
    print(f"✓ File size: {len(html_content) / 1024:.1f} KB")
