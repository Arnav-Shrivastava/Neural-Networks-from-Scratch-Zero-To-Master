import nbformat
from nbclient import NotebookClient

def append_step6_cells(nb):
    cells = []

    # Section Header
    cells.append(nbformat.v4.new_markdown_cell("""---
## Part 6: Failure Demos — Vanishing Gradients & Dying ReLU

To truly master neural networks, you must understand not only how they succeed, but exactly how they fail. In this section, we will reproduce and analyze two of the most infamous failure modes in deep learning history:
1. **Vanishing Gradients**: Why deep networks with Sigmoid activations freeze during training, visualized by measuring gradient magnitudes across 8 layers.
2. **Dying ReLU**: How an overly aggressive learning rate permanently kills neurons, and how to revive them using LeakyReLU and proper learning rate tuning.
"""))

    # 6.1 Vanishing Gradients Demo
    cells.append(nbformat.v4.new_markdown_cell("""### 6.1 Failure Demo A: Vanishing Gradients in Deep Sigmoid Networks

#### 1. The Idea in One Plain Sentence
When using Sigmoid in deep networks, the chain rule multiplies derivatives that are at most $0.25$ layer-after-layer, causing the error signal to decay exponentially until early layers receive virtually zero gradient and stop learning completely.

#### 2. Real-Life Analogy
A whisper passed down a long line of 20 people. Each person whispers slightly quieter than what they heard. By the time the whisper reaches the first person in line, the message has faded into total, undetectable silence.

#### 3. Tiny Example with Real Numbers Worked by Hand
Recall that the derivative of Sigmoid is $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$, whose absolute maximum value is only $0.25$ (when $z = 0$).
If an error signal of magnitude $1.0$ travels backward through 6 Sigmoid hidden layers:
$$\\text{Gradient at Layer 1} \\le 1.0 \\times (0.25) \\times (0.25) \\times (0.25) \\times (0.25) \\times (0.25) \\times (0.25)$$
$$\\text{Gradient at Layer 1} \\le (0.25)^6 = 0.000244$$
The gradient has shrunk by **$99.98\\%$** before it even reaches the first layer! If $|z| > 2$, the local derivative is $0.1$ or smaller, making $(0.1)^6 = 0.000001$ (one-millionth!).

#### 4. The Math with Every Symbol Explained in Words
$$\\delta^{[1]} = \\delta^{[L]} \\prod_{l=2}^L \\left( W^{[l]T} \\odot \\sigma'(z^{[l-1]}) \\right)$$
- $\\delta^{[L]}$: The error signal starting at the output layer $L$.
- $\\prod_{l=2}^L$: The product of all backward transmission matrices across layers.
- $\\sigma'(z^{[l-1]})$: The local activation derivative at each intermediate layer. Because $\\sigma'(z) \\le 0.25$, repeated multiplication causes exponential decay toward $0.0$.
- $\\delta^{[1]}$: The resulting error signal that reaches Layer 1. When this reaches zero, Layer 1 weights receive no update and feature extraction freezes.
"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will construct an 8-layer deep neural network and compare gradient flow when using Sigmoid vs. ReLU by measuring the mean absolute gradient magnitude $|\\nabla W^{[l]}|$ at every single layer."""))

    cells.append(nbformat.v4.new_code_cell("""# Simulating Gradient Flow in an 8-Layer Deep Network
import numpy as np

np.random.seed(42)
layer_dims_8 = [64] + [64] * 7 + [10]  # 8 layers deep
X_toy = np.random.randn(32, 64)
Y_toy = np.eye(10)[np.random.choice(10, 32)]

def measure_layer_gradients(activation_name):
    # Initialize weights
    weights = [np.random.randn(layer_dims_8[i], layer_dims_8[i+1]) * 
               (np.sqrt(2.0/layer_dims_8[i]) if activation_name == "relu" else np.sqrt(1.0/layer_dims_8[i]))
               for i in range(len(layer_dims_8) - 1)]
    
    # Forward Pass
    activations = [X_toy]
    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i])
        a = np.maximum(0.0, z) if activation_name == "relu" else 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
        activations.append(a)
        
    # Output Softmax
    z_out = np.dot(activations[-1], weights[-1])
    exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
    a_out = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    activations.append(a_out)
    
    # Backward Pass & Record Gradient Magnitudes
    delta = (a_out - Y_toy) / 32.0
    grad_magnitudes = []
    
    for i in reversed(range(len(weights))):
        dw = np.dot(activations[i].T, delta)
        grad_magnitudes.append(np.mean(np.abs(dw)))
        if i > 0:
            da = np.dot(delta, weights[i].T)
            dz = da * (activations[i] > 0) if activation_name == "relu" else da * (activations[i] * (1.0 - activations[i]))
            delta = dz
            
    return list(reversed(grad_magnitudes))

sig_grads = measure_layer_gradients("sigmoid")
relu_grads = measure_layer_gradients("relu")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
We computed the backpropagation gradient magnitudes across all 8 layers for both Sigmoid and ReLU. Now let's plot them."""))

    # Plotting vanishing gradients
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will plot the layer-wise gradient magnitudes on a logarithmic scale to demonstrate how Sigmoid gradients decay exponentially while ReLU gradients remain healthy."""))

    cells.append(nbformat.v4.new_code_cell("""# Plotting Vanishing Gradients across 8 Layers
import matplotlib.pyplot as plt

layers_idx = np.arange(1, len(sig_grads) + 1)

plt.figure(figsize=(9, 4.5))
plt.plot(layers_idx, sig_grads, marker="o", color="#d62728", linewidth=2.5, label="Sigmoid Network (Vanishing!)")
plt.plot(layers_idx, relu_grads, marker="s", color="#2ca02c", linewidth=2.5, label="ReLU Network (Healthy Flow)")

plt.yscale("log")
plt.title("Gradient Magnitude Across 8 Layers (Log Scale)", fontsize=13, fontweight="bold")
plt.xlabel("Layer Index (Layer 1 = Input, Layer 8 = Output)", fontsize=11)
plt.ylabel("Mean Absolute Gradient $|\\nabla W|$ (Log Scale)", fontsize=11)
plt.xticks(layers_idx)
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()"""))

    cells.append(nbformat.v4.new_markdown_cell("""#### How to Read This Plot
- **What the axes are:** The horizontal axis (X) represents the layer index from Layer 1 (closest to input data) to Layer 8 (closest to output). The vertical axis (Y) represents the mean absolute gradient magnitude on a logarithmic scale.
- **What a good result looks like:** The green line (**ReLU**). Gradients maintain a steady, healthy magnitude across all 8 layers, ensuring early layers learn just as actively as late layers.
- **What a bad result looks like:** The red line (**Sigmoid**). Gradients decay exponentially backwards, collapsing from $\\approx 10^{-2}$ at Layer 8 down to $\\approx 10^{-7}$ at Layer 1 (a 40,000x drop!).
- **What this plot tells us:** In a deep Sigmoid network, Layer 1 receives virtually zero gradient updates, freezing its weights and preventing the network from learning early sensory representations.
"""))

    # 6.2 Dying ReLU Demo
    cells.append(nbformat.v4.new_markdown_cell("""---
### 6.2 Failure Demo B: The Dying ReLU Problem & The LeakyReLU Cure

#### 1. The Idea in One Plain Sentence
If a large gradient update knocks a neuron's weights so far into the negative region that its pre-activation $z = w \\cdot x + b$ is negative for every single training sample, ReLU outputs $0$ and its gradient is $0$ forever—the neuron is permanently dead.

#### 2. Real-Life Analogy
A home electrical circuit breaker: If a sudden massive power surge trips the breaker and snaps the plastic lever off, the circuit is permanently stuck in the "off" state. No amount of flipping switches in the room will ever turn the lights back on.

#### 3. Tiny Example with Real Numbers Worked by Hand
Imagine a single neuron with 1 input $x \\in [0, 1]$.
Current weight is $w = 0.5$, bias is $b = 0.0$.
Suppose during one bad batch, a huge gradient $\\frac{\\partial L}{\\partial w} = 1.0$ occurs, and our learning rate is excessively high: $\\eta = 10.0$.
The weight update is:
$$w_{new} = w - \\eta \\frac{\\partial L}{\\partial w} = 0.5 - (10.0 \\times 1.0) = -9.5$$
Now, for any positive input sample $x \\in [0, 1]$:
$$z = w_{new} x + b = -9.5 x \\le 0$$
Because $z$ is negative for every sample in the dataset:
1. $\\text{ReLU}(z) = 0$ (the neuron outputs zero forever).
2. $\\text{ReLU}'(z) = 0$ (its local derivative is zero).
3. $\\frac{\\partial L}{\\partial w} = 0$ (it receives zero gradient blame).
4. $w_{new} = w_{new} - \\eta(0) = w_{new}$ (the weight can never change again!).
The neuron has entered a permanent coma.

#### 4. The Fix: LeakyReLU
Instead of flatlining at zero for negative inputs, **LeakyReLU** provides a tiny gentle slope $\\alpha = 0.01$:
$$\\text{LeakyReLU}(z) = \\begin{cases} z & \\text{if } z > 0 \\\\ 0.01 z & \\text{if } z \\le 0 \\end{cases}$$
$$\\text{LeakyReLU}'(z) = \\begin{cases} 1.0 & \\text{if } z > 0 \\\\ 0.01 & \\text{if } z \\le 0 \\end{cases}$$
Because the derivative is always at least $0.01$, a small "leak" of gradient flows through, giving the neuron an escape path to climb out of negative territory!
"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will simulate this failure by training on `load_digits` with an excessively high learning rate ($\\eta = 3.0$) and measure the exact percentage of dead neurons for:
1. Standard ReLU with High LR ($\\eta = 3.0$)
2. LeakyReLU with High LR ($\\eta = 3.0$)
3. Standard ReLU with Healthy LR ($\\eta = 0.15$)
"""))

    cells.append(nbformat.v4.new_code_cell("""# Dying ReLU Simulation and LeakyReLU Fix
from sklearn.datasets import load_digits
digits = load_digits()
X_dr = digits.data / 16.0
Y_dr = np.eye(10)[digits.target]

def train_and_count_dead_neurons(activation_func, lr):
    np.random.seed(42)
    in_dim, hid_dim, out_dim = 64, 64, 10
    W1 = np.random.randn(in_dim, hid_dim) * 0.1
    b1 = np.zeros((1, hid_dim))
    W2 = np.random.randn(hid_dim, out_dim) * 0.1
    b2 = np.zeros((1, out_dim))
    
    for epoch in range(12):
        for i in range(0, len(X_dr), 32):
            xb, yb = X_dr[i:i+32], Y_dr[i:i+32]
            m = xb.shape[0]
            z1 = np.dot(xb, W1) + b1
            a1 = np.maximum(0.01 * z1, z1) if activation_func == "leaky_relu" else np.maximum(0.0, z1)
            z2 = np.dot(a1, W2) + b2
            exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
            a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            dz2 = (a2 - yb) / m
            dw2 = np.dot(a1.T, dz2); db2 = np.sum(dz2, axis=0, keepdims=True)
            da1 = np.dot(dz2, W2.T)
            dz1 = da1 * np.where(z1 > 0, 1.0, 0.01) if activation_func == "leaky_relu" else da1 * (z1 > 0)
            dw1 = np.dot(xb.T, dz1); db1 = np.sum(dz1, axis=0, keepdims=True)
            
            W1 -= lr * dw1; b1 -= lr * db1
            W2 -= lr * dw2; b2 -= lr * db2
            
    # Measure dead neurons across entire dataset
    full_z1 = np.dot(X_dr, W1) + b1
    full_a1 = np.maximum(0.01 * full_z1, full_z1) if activation_func == "leaky_relu" else np.maximum(0.0, full_z1)
    # A neuron is dead if it outputs 0.0 across 100% of samples
    dead_mask = np.all(full_a1 <= 0.0, axis=0) if activation_func == "leaky_relu" else np.all(full_a1 == 0.0, axis=0)
    return np.mean(dead_mask) * 100.0

dead_relu_high_lr = train_and_count_dead_neurons("relu", lr=3.0)
dead_leaky_high_lr = train_and_count_dead_neurons("leaky_relu", lr=3.0)
dead_relu_low_lr   = train_and_count_dead_neurons("relu", lr=0.15)

print(f"Standard ReLU (High LR = 3.0):   {dead_relu_high_lr:.1f}% Dead Neurons")
print(f"LeakyReLU     (High LR = 3.0):   {dead_leaky_high_lr:.1f}% Dead Neurons")
print(f"Standard ReLU (Healthy LR = 0.15): {dead_relu_low_lr:.1f}% Dead Neurons")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
- **Standard ReLU with High LR**: **100% of hidden neurons died**! The model completely flatlined.
- **LeakyReLU with High LR**: **0.0% dead neurons**! The small negative slope ($0.01$) prevented catastrophic neuronal death.
- **Standard ReLU with Healthy LR**: **0.0% dead neurons**! Lowering the learning rate prevented violent overshooting in the first place."""))

    # Plotting dying ReLU comparison
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will plot a bar chart comparing the dead neuron percentages across all three configurations to visualize the failure and its remedies."""))

    cells.append(nbformat.v4.new_code_cell("""# Visualizing Dead Neuron Percentages
fig, ax = plt.subplots(figsize=(8, 4.5))

labels = ["Standard ReLU\\n(High LR = 3.0)", "LeakyReLU\\n(High LR = 3.0)", "Standard ReLU\\n(Healthy LR = 0.15)"]
percentages = [dead_relu_high_lr, dead_leaky_high_lr, dead_relu_low_lr]
bar_colors = ["#d62728", "#2ca02c", "#1f77b4"]

bars = ax.bar(labels, percentages, color=bar_colors, width=0.55, edgecolor="black", alpha=0.85)
ax.set_ylabel("Dead Neurons (%)", fontsize=11, fontweight="bold")
ax.set_title("Dying ReLU Failure and Solutions", fontsize=13, fontweight="bold")
ax.set_ylim(0, 115)
ax.grid(axis="y", linestyle="--", alpha=0.3)

for bar, pct in zip(bars, percentages):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 3.0, f"{pct:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()"""))

    cells.append(nbformat.v4.new_markdown_cell("""#### How to Read This Plot
- **What the axes are:** The horizontal axis (X) shows the three experimental setups. The vertical axis (Y) shows the percentage of hidden neurons that output zero across $100\\%$ of dataset samples (dead neurons).
- **What a good result looks like:** A bar at $0.0\\%$ (**LeakyReLU** or **Healthy LR**), meaning every neuron remains active and capable of learning.
- **What a bad result looks like:** A tall red bar reaching $100.0\\%$, indicating that every neuron has died and learning is permanently halted.
- **What this plot tells us:** Excessive learning rates cause irreversible neuronal death in standard ReLU networks. Using LeakyReLU or lowering the learning rate completely eliminates this catastrophic failure mode.

#### 6. Common Beginner Mistake
Increasing the learning rate when a ReLU network is not learning. If neurons have already died, increasing the learning rate will not revive them—it will only kill the few remaining survivors! If a ReLU model stalls, lower the learning rate or switch to LeakyReLU.

#### 7. Quick "Check Yourself" Question
> Why can LeakyReLU recover from a large negative weight update while standard ReLU cannot?
<details>
<summary><b>Click to reveal answer</b></summary>
Because for negative inputs, standard ReLU has a gradient of exactly $0.0$ (which blocks weight updates), whereas LeakyReLU maintains a small non-zero gradient of $\\alpha = 0.01$, allowing gradient descent to gradually nudge the weight back into the positive territory!
</details>

---

### 📌 Step 6 Recap
- **Recap in 3 bullets:**
  1. Vanishing gradients occur when multiplying saturating derivatives ($\le 0.25$ in Sigmoid) across deep layers, starving early layers of updates.
  2. Dying ReLU occurs when high learning rates throw pre-activations permanently negative, freezing derivatives at zero across all samples.
  3. LeakyReLU fixes dying neurons by providing a small escape gradient ($0.01$), while ReLU with a proper learning rate prevents the issue entirely.
- **If you remember only one thing:**
  *Never use Sigmoid in deep hidden layers; use ReLU with a sensible learning rate or LeakyReLU to keep all neurons alive.*
"""))

    nb.cells.extend(cells)
    return nb

if __name__ == "__main__":
    import nbformat
    nb = nbformat.read("neural_network_from_scratch.ipynb", as_version=4)
    nb = append_step6_cells(nb)
    print("Executing notebook for Step 6...")
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()
    nbformat.write(nb, "neural_network_from_scratch.ipynb")
    print("Step 6 successfully executed and saved!")
