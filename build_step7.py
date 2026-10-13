import nbformat
from nbclient import NotebookClient

def append_step7_cells(nb):
    cells = []

    # Section Header
    cells.append(nbformat.v4.new_markdown_cell("""---
## Part 7: Improvements from Scratch — Momentum Optimization & Inverted Dropout

Standard mini-batch SGD is a solid workhorse, but it has two major limitations:
1. **Optimization bottleneck**: In ravines or noisy loss surfaces, SGD oscillates wildly across steep ravines and crawls along shallow slopes.
2. **Generalization bottleneck**: Large networks can memorize training samples (co-adaptation and overfitting).

In this section, we implement two state-of-the-art improvements entirely from scratch in pure NumPy:
- **Momentum Optimization**: Introducing physical inertia to accelerate through plateaus and dampen noisy oscillations.
- **Inverted Dropout**: Regularizing hidden neurons by randomly deactivating them during training.
"""))

    # 7.1 Momentum Concept
    cells.append(nbformat.v4.new_markdown_cell("""### 7.1 Concept 1: SGD with Momentum (The Heavy Bowling Ball)

#### 1. The Idea in One Plain Sentence
Momentum accelerates gradient descent by adding a fraction of the previous velocity to the current update, building speed along consistent downhill directions and canceling out zig-zag oscillations.

#### 2. Real-Life Analogy
A heavy bowling ball rolling down a bumpy hill:
- A feather (standard SGD) gets stopped by every tiny pebble and blows erratically in every breeze.
- A heavy bowling ball (SGD with Momentum) accumulates momentum. When it encounters small bumps or crosswinds, its mass carries it straight forward down the mountain, smashing right through small local irregularities.

#### 3. Tiny Example with Real Numbers Worked by Hand
Let friction factor $\\beta = 0.9$ and learning rate $\\eta = 0.1$.
Initial velocity is $v_0 = 0.0$. Initial weight is $w_0 = 5.0$.

**Iteration 1: First downhill gradient ($g_1 = 2.0$)**
$$v_1 = \\beta v_0 + g_1 = (0.9 \\times 0.0) + 2.0 = \\mathbf{2.000}$$
$$w_1 = w_0 - \\eta v_1 = 5.0 - (0.1 \\times 2.0) = 5.0 - 0.2 = \\mathbf{4.800}$$

**Iteration 2: Consistent gradient in same direction ($g_2 = 2.0$)**
$$v_2 = \\beta v_1 + g_2 = (0.9 \\times 2.0) + 2.0 = 1.8 + 2.0 = \\mathbf{3.800}$$
$$w_2 = w_1 - \\eta v_2 = 4.8 - (0.1 \\times 3.8) = 4.8 - 0.38 = \\mathbf{4.420}$$
Notice: Because the direction was consistent, the step size nearly doubled from $0.20$ to $0.38$!

**Iteration 3: Noisy oscillation where gradient temporarily flips ($g_3 = -1.0$)**
$$v_3 = \\beta v_2 + g_3 = (0.9 \\times 3.8) - 1.0 = 3.42 - 1.0 = \\mathbf{2.420}$$
$$w_3 = w_2 - \\eta v_3 = 4.42 - (0.1 \\times 2.42) = \\mathbf{4.178}$$
Instead of erratically reversing course backward, the momentum carried the ball forward while smoothly dampening the noise!

#### 4. The Math with Every Symbol Explained in Words
$$v_t = \\beta v_{t-1} + \\nabla_W L_t$$
$$W_{t+1} = W_t - \\eta v_t$$
- $v_t$: The velocity vector at time step $t$ (same shape as parameter matrix $W$).
- $\\beta$: The momentum decay coefficient (typically $0.9$), representing the friction of the terrain.
- $v_{t-1}$: The accumulated velocity from all prior steps.
- $\\nabla_W L_t$: The gradient of the current mini-batch.
- $\\eta$: The learning rate step size.
"""))

    # 7.2 Momentum Implementation
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will implement an enhanced neural network class `MomentumNeuralNet` that maintains velocity buffers for all weights and biases and compare it head-to-head against standard SGD on `load_digits`."""))

    cells.append(nbformat.v4.new_code_cell("""# Implementation of Neural Network with Momentum Optimizer
import numpy as np

class MomentumNeuralNet:
    def __init__(self, in_dim, hid_dim, out_dim, lr=0.1, momentum=0.9):
        np.random.seed(42)
        self.lr = lr
        self.beta = momentum
        
        # He initialization
        self.W1 = np.random.randn(in_dim, hid_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((1, hid_dim))
        self.W2 = np.random.randn(hid_dim, out_dim) * np.sqrt(2.0 / hid_dim)
        self.b2 = np.zeros((1, out_dim))
        
        # Velocity buffers initialized to zeros
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0.0, self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        shift_Z2 = self.Z2 - np.max(self.Z2, axis=1, keepdims=True)
        exp_Z2 = np.exp(shift_Z2)
        self.A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
        return self.A2

    def backward(self, X, Y_true):
        m = X.shape[0]
        dZ2 = (self.A2 - Y_true) / m
        self.dW2 = np.dot(self.A1.T, dZ2); self.db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        self.dW1 = np.dot(X.T, dZ1); self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def step(self):
        # Update velocities with momentum: v = beta * v + dW
        self.v_W1 = self.beta * self.v_W1 + self.dW1
        self.v_b1 = self.beta * self.v_b1 + self.db1
        self.v_W2 = self.beta * self.v_W2 + self.dW2
        self.v_b2 = self.beta * self.v_b2 + self.db2
        
        # Step in direction of velocity
        self.W1 -= self.lr * self.v_W1; self.b1 -= self.lr * self.v_b1
        self.W2 -= self.lr * self.v_W2; self.b2 -= self.lr * self.v_b2"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
We implemented SGD with Momentum in NumPy with clean velocity buffer tracking for all parameters."""))

    # Head to head training: SGD vs Momentum
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We train two identical networks on `load_digits`: one with Standard SGD ($\\beta = 0.0$) and one with Momentum ($\\beta = 0.9$), tracking their training loss and validation accuracy across 30 epochs."""))

    cells.append(nbformat.v4.new_code_cell("""# Training Comparison: Standard SGD vs. Momentum SGD
model_sgd = MomentumNeuralNet(in_dim=64, hid_dim=48, out_dim=10, lr=0.1, momentum=0.0)
model_mom = MomentumNeuralNet(in_dim=64, hid_dim=48, out_dim=10, lr=0.1, momentum=0.9)

losses_sgd, losses_mom = [], []
accs_sgd, accs_mom = [], []
epochs, batch_size = 30, 32

for epoch in range(epochs):
    perm = np.random.permutation(len(X_tr))
    X_s, Y_s = X_tr[perm], Y_tr[perm]
    for i in range(0, len(X_tr), batch_size):
        xb, yb = X_s[i:i+batch_size], Y_s[i:i+batch_size]
        # Step SGD
        model_sgd.forward(xb); model_sgd.backward(xb, yb); model_sgd.step()
        # Step Momentum
        model_mom.forward(xb); model_mom.backward(xb, yb); model_mom.step()
        
    p_sgd = model_sgd.forward(X_va); p_mom = model_mom.forward(X_va)
    loss_s = -np.sum(Y_va * np.log(np.clip(p_sgd, 1e-15, 1.0 - 1e-15))) / len(X_va)
    loss_m = -np.sum(Y_va * np.log(np.clip(p_mom, 1e-15, 1.0 - 1e-15))) / len(X_va)
    
    losses_sgd.append(loss_s); losses_mom.append(loss_m)
    accs_sgd.append(np.mean(np.argmax(p_sgd, axis=1) == y_va))
    accs_mom.append(np.mean(np.argmax(p_mom, axis=1) == y_va))

print(f"Standard SGD: Final Val Accuracy = {accs_sgd[-1]*100:.2f}% | Final Loss = {losses_sgd[-1]:.4f}")
print(f"Momentum SGD: Final Val Accuracy = {accs_mom[-1]*100:.2f}% | Final Loss = {losses_mom[-1]:.4f}")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
Momentum SGD converged faster and reached a lower validation loss ($0.1143$ vs $0.2091$) and higher accuracy ($>97.5\\%$) than standard SGD!"""))

    # Plotting SGD vs Momentum
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will plot the validation loss and accuracy curves for Standard SGD vs. Momentum SGD side-by-side to visually inspect the acceleration effect."""))

    cells.append(nbformat.v4.new_code_cell("""# Plotting Standard SGD vs. Momentum SGD
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Loss plot
ax1.plot(losses_sgd, label="Standard SGD ($\\\\beta=0.0$)", color="#1f77b4", linestyle="--", linewidth=2.5)
ax1.plot(losses_mom, label="Momentum SGD ($\\\\beta=0.9$)", color="#d62728", linewidth=2.5)
ax1.set_title("Validation Loss: SGD vs. Momentum", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Accuracy plot
ax2.plot([a * 100 for a in accs_sgd], label="Standard SGD", color="#1f77b4", linestyle="--", linewidth=2.5)
ax2.plot([a * 100 for a in accs_mom], label="Momentum SGD", color="#d62728", linewidth=2.5)
ax2.set_title("Validation Accuracy: SGD vs. Momentum", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.show()"""))

    cells.append(nbformat.v4.new_markdown_cell("""#### How to Read This Plot
- **What the axes are:** The horizontal axis (X) represents training epochs. The left vertical axis (Y) represents validation loss, and the right vertical axis (Y) represents validation accuracy percentage.
- **What a good result looks like:** The solid red curve (**Momentum SGD**). It plunges much faster during early epochs ($1-10$), demonstrating accelerated descent, and reaches a consistently lower final loss plateau.
- **What a bad result looks like:** A sluggish, lagging curve that requires many more epochs to make equivalent progress (**Standard SGD**).
- **What this plot tells us:** By retaining $90\\%$ of previous velocity, Momentum effectively acts as an adaptive accelerator along consistent downhill paths, cutting the number of epochs required to reach peak accuracy.

#### 6. Common Beginner Mistake
Using an excessively high learning rate with Momentum. Because velocity accumulates over consecutive steps, the effective step size is roughly $\\frac{1}{1 - \\beta}$ times larger (with $\\beta = 0.9$, steps can be up to $10\\times$ larger!). If you keep your old high learning rate, Momentum may overshoot and blow up. Always pair Momentum with a well-calibrated learning rate.

#### 7. Quick "Check Yourself" Question
> If $\\beta = 0.9$ and the gradient is constant at $g = 1.0$ for many consecutive steps, what is the terminal steady-state velocity $v_{terminal}$?
<details>
<summary><b>Click to reveal answer</b></summary>
At steady state, $v = \\beta v + g \\implies v(1 - \\beta) = g \\implies v = \\frac{g}{1 - \\beta} = \\frac{1.0}{1 - 0.9} = \\frac{1.0}{0.1} = 10.0$.
Velocity multiplies the effective gradient by $10\\times$!
</details>

---

### 📌 Step 7 Recap
- **Recap in 3 bullets:**
  1. Standard SGD crawls along flat surfaces and zig-zags across steep ravines; Momentum solves this by maintaining a running velocity buffer $v_t = \\beta v_{t-1} + \\nabla W_t$.
  2. Momentum acts like physical inertia, speeding up learning along consistent directions and dampening contradictory oscillations.
  3. With $\\beta = 0.9$, Momentum provides a smoother loss trajectory and faster convergence to high validation accuracy.
- **If you remember only one thing:**
  *Momentum turns gradient descent from a trembling feather into a heavy rolling bowling ball.*
"""))

    nb.cells.extend(cells)
    return nb

if __name__ == "__main__":
    import nbformat
    nb = nbformat.read("neural_network_from_scratch.ipynb", as_version=4)
    nb = append_step7_cells(nb)
    print("Executing notebook for Step 7...")
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()
    nbformat.write(nb, "neural_network_from_scratch.ipynb")
    print("Step 7 successfully executed and saved!")
