# Neural Network From Scratch

A complete, beginner-friendly deep dive into building and understanding artificial neural networks from the mathematical ground up using pure NumPy—from single neuron mechanics to backpropagation, gradient checking, and handwritten digit classification.

---

## 📌 Project Overview
- **Core Focus**: Demystifying deep learning by writing every forward pass, loss computation, and backpropagation equation in pure NumPy before ever touching PyTorch.
- **Dataset**:
  - Part 1: Scikit-learn `load_digits` (8x8 grayscale handwritten digits)
  - Part 2: Full MNIST benchmark dataset (28x28 grayscale digits)
- **Architecture**: Multi-layer perceptron with customizable activations, weight initializations, and regularization.

---

## 🧭 Roadmap & Modules
1. **Theory & First Principles**: Neurons, layers, forward pass, cross-entropy loss, non-linear activation necessity, chain rule, and hand-calculated 2-input network backprop.
2. **2-Layer Network from Scratch**: Forward pass, Softmax, Cross-Entropy, backpropagation, and mini-batch SGD loop with shape annotations.
3. **Gradient Checking**: Finite-difference numerical gradient verification to prove backpropagation correctness.
4. **Activation Functions**: ReLU, Sigmoid, and Tanh implementations, mathematical derivations, derivative plots, and training dynamics.
5. **Weight Initialization**: Zeros vs. Random vs. Xavier/Glorot vs. He/Kaiming with layer-wise activation histograms.
6. **Failure Modes & Diagnostics**: Vanishing gradient visualization in deep sigmoid networks, dying ReLU detection, and LeakyReLU remedy.
7. **Optimization & Regularization**: Momentum / L2 Regularization / Dropout implemented from scratch.
8. **PyTorch Framework Comparison**: Minimal 30-line PyTorch equivalent comparing accuracy and exposing the abstraction layer.
9. **Mastery Challenge**: 8 "prove it" questions, 3 hands-on coding exercises, and 1 extension challenge.

---

## 🚀 Quickstart
```bash
git clone https://github.com/Arnav-Shrivastava/Neural-Networks-from-Scratch-Zero-To-Master.git
cd Neural-Networks-from-Scratch-Zero-To-Master
pip install -r requirements.txt
jupyter notebook neural_network_from_scratch.ipynb
```
