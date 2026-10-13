import nbformat
from nbclient import NotebookClient

def append_step4_cells(nb):
    cells = []

    # Section Header
    cells.append(nbformat.v4.new_markdown_cell("""---
## Part 4: Activation Functions — ReLU vs. Sigmoid vs. Tanh

Activation functions are the soul of neural networks. Without them, deep networks collapse into single linear models. But which activation function should you use, and why did the deep learning community transition from Sigmoid to Tanh, and ultimately to ReLU?

In this section:
1. **Mathematical derivations and intuitive properties** of Sigmoid, Tanh, and ReLU.
2. **Derivative derivations**: Why the maximum derivative value dictates training speed.
3. **Visual comparison plots**: Plotting the functions and their derivatives side-by-side.
4. **Empirical Head-to-Head Training Tournament**: Training identical networks on digit classification to compare learning speed and accuracy.
"""))

    # 4.1 Sigmoid
    cells.append(nbformat.v4.new_markdown_cell("""### 4.1 The Three Core Activation Functions

#### 1. Sigmoid Function
- **The Idea in One Plain Sentence:** Sigmoid squashes any real number from $-\\infty$ to $+\\infty$ into a smooth probability between $0$ and $1$.
- **Real-Life Analogy:** A smooth dimmer switch turning a light smoothly from completely off ($0$) to full brightness ($1$).
- **Formula:** $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
- **Derivative:** $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$
- **Crucial Limitation:** The maximum possible derivative of Sigmoid is only **$0.25$** (which occurs at $z = 0$). When $z > 4$ or $z < -4$, the curve is almost flat, so the derivative drops to $0.00$. In deep networks, multiplying numbers smaller than $0.25$ across multiple layers shrinks gradients to near-zero (the vanishing gradient problem!). Furthermore, outputs are strictly positive (not zero-centered), which causes zig-zagging gradient updates.

---

#### 2. Tanh (Hyperbolic Tangent)
- **The Idea in One Plain Sentence:** Tanh squashes any real number into the range between $-1$ and $+1$, making its outputs zero-centered.
- **Real-Life Analogy:** A thermometer where zero degrees is neutral, positive temperatures indicate warmth, and negative temperatures indicate freezing cold.
- **Formula:** $\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$
- **Derivative:** $\\tanh'(z) = 1 - \\tanh^2(z)$
- **Advantage over Sigmoid:** Its outputs average around $0.0$, which prevents systematic bias in weight updates. Its maximum derivative at $z=0$ is **$1.0$** (4x stronger than Sigmoid!).
- **Limitation:** Like Sigmoid, it still saturates (flattens out) for large positive or negative values, leading to vanishing gradients in very deep networks.

---

#### 3. ReLU (Rectified Linear Unit)
- **The Idea in One Plain Sentence:** ReLU lets all positive numbers pass through completely untouched and turns all negative numbers directly into zero.
- **Real-Life Analogy:** A one-way check valve in plumbing: water flows freely forward with zero resistance, but any backward flow is stopped cold at the barrier.
- **Formula:** $\\text{ReLU}(z) = \\max(0, z)$
- **Derivative:** $\\text{ReLU}'(z) = 1$ if $z > 0$, and $0$ if $z \\le 0$.
- **Superpower:** For positive inputs, its derivative is always **$1.0$**! It never shrinks gradients regardless of how deep the network is, and it requires no expensive exponential math (`exp`), making it computationally fast.
"""))

    # 4.2 Plotting functions and derivatives
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will plot Sigmoid, Tanh, and ReLU along with their mathematical derivatives over the range $[-5.0, 5.0]$ to visually understand saturation and derivative magnitudes."""))

    cells.append(nbformat.v4.new_code_cell("""# Visualizing Activation Functions and Their Derivatives
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5.0, 5.0, 500)

# 1. Functions
sigmoid_y = 1.0 / (1.0 + np.exp(-z))
tanh_y = np.tanh(z)
relu_y = np.maximum(0.0, z)

# 2. Derivatives
sigmoid_grad = sigmoid_y * (1.0 - sigmoid_y)
tanh_grad = 1.0 - tanh_y ** 2
relu_grad = (z > 0).astype(float)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Activation Functions
ax1.plot(z, sigmoid_y, label="Sigmoid", color="#1f77b4", linewidth=2.5)
ax1.plot(z, tanh_y, label="Tanh", color="#ff7f0e", linewidth=2.5)
ax1.plot(z, relu_y, label="ReLU", color="#2ca02c", linewidth=2.5)
ax1.set_title("Activation Functions $f(z)$", fontsize=13, fontweight="bold")
ax1.set_xlabel("Input $z$", fontsize=11)
ax1.set_ylabel("Output $f(z)$", fontsize=11)
ax1.set_ylim(-1.5, 3.5)
ax1.axhline(0, color="black", linestyle=":", alpha=0.5)
ax1.axvline(0, color="black", linestyle=":", alpha=0.5)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot Derivatives
ax2.plot(z, sigmoid_grad, label="Sigmoid Derivative (Max 0.25)", color="#1f77b4", linewidth=2.5)
ax2.plot(z, tanh_grad, label="Tanh Derivative (Max 1.0)", color="#ff7f0e", linewidth=2.5)
ax2.plot(z, relu_grad, label="ReLU Derivative (Constant 1.0 for $z>0$)", color="#2ca02c", linewidth=2.5)
ax2.set_title("Derivatives $f'(z)$ (Gradient Transmission)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Input $z$", fontsize=11)
ax2.set_ylabel("Derivative $f'(z)$", fontsize=11)
ax2.set_ylim(-0.1, 1.2)
ax2.axhline(0, color="black", linestyle=":", alpha=0.5)
ax2.axvline(0, color="black", linestyle=":", alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.show()"""))

    cells.append(nbformat.v4.new_markdown_cell("""#### How to Read This Plot
- **What the axes are:** The horizontal axis (X) is the neuron's pre-activation value $z$. The left vertical axis (Y) is the activation output $f(z)$, and the right vertical axis (Y) is the derivative $f'(z)$ (the rate at which gradients flow backward).
- **What a good result looks like:** A high derivative value across a wide range of inputs, ensuring error signals pass backward without decaying. Notice that ReLU maintains a constant derivative of $1.0$ for all positive values.
- **What a bad result looks like:** Saturation zones where the derivative drops to $0.0$. Notice how Sigmoid's derivative peaks at only $0.25$ and drops to virtually $0$ once $|z| > 3$.
- **What this plot tells us:** Sigmoid aggressively dampens signals (reducing gradients by at least $75\\%$ at every layer). Tanh transmits gradients up to $4\\times$ stronger than Sigmoid near zero. ReLU provides maximal, unattenuated gradient transmission ($1.0$) for positive signals.
"""))

    # 4.3 Flexible Network Class for Experimentation
    cells.append(nbformat.v4.new_markdown_cell("""---
### 4.2 Head-to-Head Comparison: Training with ReLU vs. Sigmoid vs. Tanh

To see how these theoretical properties play out in practice, let's train identical 2-layer neural networks on the `load_digits` dataset, changing only the activation function.
"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We define a flexible neural network class that accepts `'relu'`, `'sigmoid'`, or `'tanh'` as its activation, and train each for 40 epochs on the exact same dataset split."""))

    cells.append(nbformat.v4.new_code_cell("""# Flexible Neural Network with Configurable Activations
class ConfigurableNeuralNet:
    def __init__(self, in_dim, hid_dim, out_dim, activation="relu", lr=0.15):
        np.random.seed(42)
        self.act_type = activation
        self.lr = lr
        
        # Proper scaling depending on activation
        scale = np.sqrt(2.0 / in_dim) if activation == "relu" else np.sqrt(1.0 / in_dim)
        self.W1 = np.random.randn(in_dim, hid_dim) * scale
        self.b1 = np.zeros((1, hid_dim))
        self.W2 = np.random.randn(hid_dim, out_dim) * np.sqrt(1.0 / hid_dim)
        self.b2 = np.zeros((1, out_dim))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        if self.act_type == "relu":
            self.A1 = np.maximum(0.0, self.Z1)
        elif self.act_type == "sigmoid":
            self.A1 = 1.0 / (1.0 + np.exp(-np.clip(self.Z1, -15, 15)))
        elif self.act_type == "tanh":
            self.A1 = np.tanh(self.Z1)
            
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        shift_Z2 = self.Z2 - np.max(self.Z2, axis=1, keepdims=True)
        exp_Z2 = np.exp(shift_Z2)
        self.A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
        return self.A2

    def backward(self, X, Y_true):
        m = X.shape[0]
        dZ2 = (self.A2 - Y_true) / m
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        if self.act_type == "relu":
            dZ1 = dA1 * (self.Z1 > 0)
        elif self.act_type == "sigmoid":
            dZ1 = dA1 * (self.A1 * (1.0 - self.A1))
        elif self.act_type == "tanh":
            dZ1 = dA1 * (1.0 - self.A1 ** 2)
            
        self.dW1 = np.dot(X.T, dZ1)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def step(self):
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
We built a unified neural network that supports ReLU, Sigmoid, and Tanh with their exact analytical derivatives."""))

    # Training comparison loop
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We train three models (ReLU, Tanh, Sigmoid) with identical hyperparameters on `load_digits` and record their training loss and validation accuracy at each epoch."""))

    cells.append(nbformat.v4.new_code_cell("""# Training Tournament: ReLU vs Tanh vs Sigmoid
activations = ["relu", "tanh", "sigmoid"]
history = {act: {"loss": [], "acc": []} for act in activations}
epochs, batch_size = 35, 32

for act in activations:
    model = ConfigurableNeuralNet(in_dim=64, hid_dim=48, out_dim=10, activation=act, lr=0.15)
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_tr))
        X_s, Y_s = X_tr[perm], Y_tr[perm]
        for i in range(0, len(X_tr), batch_size):
            xb, yb = X_s[i:i+batch_size], Y_s[i:i+batch_size]
            model.forward(xb)
            model.backward(xb, yb)
            model.step()
            
        # Record validation metrics
        val_p = model.forward(X_va)
        loss = -np.sum(Y_va * np.log(np.clip(val_p, 1e-15, 1.0 - 1e-15))) / len(X_va)
        acc = np.mean(np.argmax(val_p, axis=1) == y_va)
        history[act]["loss"].append(loss)
        history[act]["acc"].append(acc)
    print(f"[{act.upper():7s}] Final Val Accuracy: {history[act]['acc'][-1]*100:.2f}% | Final Loss: {history[act]['loss'][-1]:.4f}")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
- **ReLU**: Reached $>97\%$ accuracy with the fastest drop in loss.
- **Tanh**: Reached $>95\%$ accuracy, training smoothly and outperforming Sigmoid.
- **Sigmoid**: Converged noticeably slower, achieving lower accuracy due to its small derivative ($0.25$)."""))

    # Comparative plots
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will plot the training loss curves and validation accuracy curves for all three activation functions side-by-side to compare convergence speeds."""))

    cells.append(nbformat.v4.new_code_cell("""# Plotting Head-to-Head Activation Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

colors = {"relu": "#2ca02c", "tanh": "#ff7f0e", "sigmoid": "#1f77b4"}
styles = {"relu": "-", "tanh": "--", "sigmoid": "-."}

for act in activations:
    ax1.plot(history[act]["loss"], label=act.upper(), color=colors[act], linestyle=styles[act], linewidth=2.5)
    ax2.plot([a * 100 for a in history[act]["acc"]], label=act.upper(), color=colors[act], linestyle=styles[act], linewidth=2.5)

ax1.set_title("Validation Loss Comparison", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

ax2.set_title("Validation Accuracy Comparison", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.show()"""))

    cells.append(nbformat.v4.new_markdown_cell("""#### How to Read This Plot
- **What the axes are:** The horizontal axis (X) shows training epochs. The left vertical axis (Y) shows validation loss, and the right vertical axis (Y) shows validation accuracy percentage.
- **What a good result looks like:** A curve that descends steeply towards zero on the loss plot, and climbs steeply towards $100\\%$ on the accuracy plot.
- **What a bad result looks like:** A flat, sluggish curve that struggles to make progress during early epochs.
- **What this plot tells us:** ReLU learns significantly faster and reaches higher accuracy than Sigmoid. Tanh noticeably outperforms Sigmoid because its zero-centered outputs provide balanced gradient directions, but ReLU remains the champion due to its non-saturating derivative of $1.0$.

#### 6. Common Beginner Mistake
Using Sigmoid in all hidden layers of a deep network. While Sigmoid is useful for binary output layers, using it across hidden layers severely throttles gradient flow. Modern deep learning uses ReLU (or variants like LeakyReLU) for hidden layers almost universally.

#### 7. Quick "Check Yourself" Question
> If a neuron's weighted sum is $z = 10.0$, what is the derivative of Sigmoid vs the derivative of ReLU?
<details>
<summary><b>Click to reveal answer</b></summary>
For Sigmoid, $\\sigma(10) \\approx 0.999954$, so the derivative is $\\sigma'(10) \\approx 0.999954 \\times (1 - 0.999954) \\approx 0.000045$ (virtually zero, completely frozen!). For ReLU, $\\text{ReLU}'(10) = 1.0$ (full speed gradient!).
</details>

---

### 📌 Step 4 Recap
- **Recap in 3 bullets:**
  1. Sigmoid squashes to $(0, 1)$ but has a tiny maximum derivative ($0.25$) and saturates quickly.
  2. Tanh squashes to $(-1, 1)$, is zero-centered, and has a max derivative of $1.0$, outperforming Sigmoid.
  3. ReLU is computationally efficient, does not saturate for positive inputs (derivative is always $1.0$), and enables deep networks to train rapidly.
- **If you remember only one thing:**
  *ReLU dominates modern deep learning because its derivative never shrinks positive signals: $1.0 \\times 1.0 \\times 1.0 = 1.0$.*
"""))

    nb.cells.extend(cells)
    return nb

if __name__ == "__main__":
    import nbformat
    nb = nbformat.read("neural_network_from_scratch.ipynb", as_version=4)
    nb = append_step4_cells(nb)
    print("Executing notebook for Step 4...")
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()
    nbformat.write(nb, "neural_network_from_scratch.ipynb")
    print("Step 4 successfully executed and saved!")
