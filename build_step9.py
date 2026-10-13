import nbformat
from nbclient import NotebookClient

def append_step9_cells(nb):
    cells = []

    # Section Header
    cells.append(nbformat.v4.new_markdown_cell("""---
## Part 9: The Mastery Challenge — "Prove It" Questions, Exercises & Extension

Congratulations! You have journeyed from a single equation $z = w \\cdot x + b$ all the way to a complete, production-grade neural network in pure NumPy, numerical gradient checking, failure mode diagnostics, optimization enhancements, and framework parity with PyTorch.

Now it is time to prove your mastery. Below are:
1. **8 "Prove It" Questions** (progressive difficulty, no answers given—test yourself first!).
2. **3 Hands-On Coding Exercises** to extend your scratch library.
3. **1 Big Extension Idea** for your next portfolio breakthrough.
"""))

    # 9.1 The 8 "Prove It" Questions
    cells.append(nbformat.v4.new_markdown_cell("""### 9.1 The 8 "Prove It" Conceptual Questions

*(Answers are intentionally omitted so you can test your genuine understanding. Try answering each out loud or in writing before asking your mentor!)*

#### Easy (Confidence Builders)
1. **Question 1 (The Dials):** In plain English, what is the physical purpose of a **weight** in a neuron, and what is the physical purpose of the **bias**? What would go wrong if we didn't have a bias term?
2. **Question 2 (The Scoreboard):** If our output layer produces probabilities $[0.10, 0.80, 0.10]$ for classes 0, 1, and 2, and the ground-truth label is class 1 (one-hot: $[0, 1, 0]$), what is the exact numerical gradient $\\frac{\\partial L}{\\partial z}$ at the pre-activation output for each class?
3. **Question 3 (The Flat Glass):** Why is it mathematically impossible for a 100-layer neural network with no activation functions to learn a circular or curved decision boundary?

---

#### Medium (Mechanics & Algorithms)
4. **Question 4 (The Order of Operations):** During backpropagation, why is it fatal to update the weights of Layer 2 *before* computing the gradients for Layer 1? What exact value from Layer 2 does Layer 1 depend on?
5. **Question 5 (The Symmetry Lock):** If you initialize all weights in a network to exactly $0.0$, what will the hidden activations and backpropagation gradients look like across all neurons in the same layer? Why can the network never break this symmetry on its own?

---

#### Harder (Deep Dynamics & Diagnostics)
6. **Question 6 (The Fading Whisper):** Why does an 8-layer deep network using Sigmoid activations suffer from vanishing gradients, while the exact same architecture using ReLU trains easily? Cite the maximum derivative values of both functions in your explanation.
7. **Question 7 (The Calculus Judge):** In numerical gradient checking, why is the two-sided central difference formula $\\frac{f(\\theta + \\epsilon) - f(\\theta - \\epsilon)}{2\\epsilon}$ vastly more accurate than the one-sided formula $\\frac{f(\\theta + \\epsilon) - f(\\theta)}{\\epsilon}$? What are their respective approximation error orders in terms of $\\epsilon$?

---

#### Mastery (Senior ML Debugging)
8. **Question 8 (The Post-Mortem):** You train a deep ReLU network on MNIST with an excessively high learning rate of $\\eta = 5.0$. After 2 epochs, the training loss stops changing completely, and 90% of the hidden neurons output $0.0$ on every sample in the dataset. Explain mechanically:
   - What happened to the weights and biases of those neurons during that fatal update step?
   - Why is it mathematically impossible for those neurons to ever receive a non-zero gradient again?
   - How does switching to LeakyReLU permanently prevent this failure?
"""))

    # 9.2 The 3 Coding Exercises
    cells.append(nbformat.v4.new_markdown_cell("""---
### 9.2 Three Hands-On Coding Exercises

Put your pure NumPy deep learning engine to work by implementing these three practical extensions:

#### Exercise 1: Implement the ELU (Exponential Linear Unit) Activation
- **The Concept:** Cleve et al. (2015) introduced ELU to combine the benefits of ReLU (no saturation for positive inputs) with smooth negative saturation that brings mean activations closer to zero:
  $$\\text{ELU}(z) = \\begin{cases} z & \\text{if } z > 0 \\\\ \\alpha (e^z - 1) & \\text{if } z \\le 0 \\end{cases}$$
- **Your Task:** Derive the derivative $\\frac{d\\text{ELU}}{dz}$ by hand, and write a Python class `ELUActivation` with `.forward(z)` and `.backward(dz)` methods in NumPy (use $\\alpha = 1.0$).

---

#### Exercise 2: Implement L2 Regularization (Weight Decay) from Scratch
- **The Concept:** To prevent weights from growing excessively large, we penalize the sum of squared weights:
  $$L_{total} = L_{cross\\_entropy} + \\frac{\\lambda}{2 m} \\left( \\|W_1\\|_F^2 + \\|W_2\\|_F^2 \\right)$$
- **Your Task:**
  1. Add the regularization penalty to `compute_loss`.
  2. Add the regularization gradient $\\frac{\\lambda}{m} W$ to the analytical weight gradient in `backward`.
  3. Verify with numerical gradient checking that your regularized analytical gradients match finite differences to $< 10^{-7}$!

---

#### Exercise 3: Build a Pure NumPy Confusion Matrix & Error Inspector
- **The Concept:** Accuracy tells you what percentage you got right, but not *how* you failed.
- **Your Task:** Without using scikit-learn, write a pure NumPy function `compute_confusion_matrix(y_true, y_pred, num_classes=10)` that builds a $10 \\times 10$ matrix where entry $(i, j)$ is the number of times true digit $i$ was predicted as digit $j$. Identify the top 3 most commonly confused digit pairs in the MNIST test set!
"""))

    # 9.3 Extension Project Idea
    cells.append(nbformat.v4.new_markdown_cell("""---
### 9.3 Portfolio Extension Project: A 2D Convolutional Layer in Pure NumPy

**The Challenge:**
Our 2-layer MLP treats images as flat 1D vectors ($784$ unorganized pixels), completely ignoring spatial relationships (e.g. neighboring pixels forming lines and loops).

**Your Next Big Milestone:**
Build a 2D Convolutional Neural Network (CNN) from scratch in NumPy:
1. Implement `conv2d_forward(X, W, b, stride=1, padding=1)`: Slide a $3 \\times 3$ filter kernel over an image tensor, computing spatial dot products.
2. Implement `maxpool2d_forward(X, pool_size=2, stride=2)`: Downsample spatial dimensions by taking the maximum value in each $2 \\times 2$ patch.
3. Connect the output of the convolutional layer into our `TwoLayerNeuralNetwork`.
4. Train on MNIST and observe how a CNN achieves $>98\\%$ accuracy with significantly fewer parameters than a fully connected network!
"""))

    # 9.4 Final Summary & Master Sign-Off
    cells.append(nbformat.v4.new_markdown_cell("""---
### 📌 Step 9 & Project Wrap-Up
- **Recap in 3 bullets:**
  1. Deep learning is not magic: it is high school algebra, basic single-variable calculus, and tidy matrix multiplications.
  2. The foundation you built here—vectorized forward passes, softmax stability, backpropagation chain rule, gradient checking, and optimization mechanics—is identical to what powers billion-parameter frontier models.
  3. Frameworks like PyTorch and JAX provide GPU speed and automatic differentiation, but your first-principles foundation gives you the power to debug, innovate, and master any architecture.
- **If you remember only one thing:**
  *Never treat an algorithm as a black box. If you can write it from scratch in NumPy, you truly understand it.*
"""))

    nb.cells.extend(cells)
    return nb

if __name__ == "__main__":
    import nbformat
    nb = nbformat.read("neural_network_from_scratch.ipynb", as_version=4)
    nb = append_step9_cells(nb)
    print("Executing notebook for Step 9...")
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()
    nbformat.write(nb, "neural_network_from_scratch.ipynb")
    print("Step 9 successfully executed and saved!")
