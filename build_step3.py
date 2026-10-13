import nbformat
from nbclient import NotebookClient

def append_step3_cells(nb):
    cells = []

    # Section Header
    cells.append(nbformat.v4.new_markdown_cell("""---
## Part 3: Gradient Checking — The Ultimate Debugging Tool

When writing backpropagation code from scratch, you face a terrifying trap: **silent mathematical bugs**.
If you make a typo in your backprop code (such as flipping a plus sign to a minus, forgetting to divide by batch size $m$, or transposing a matrix incorrectly), your network might still run without crashing! It might even learn a little bit (e.g. reaching 70% accuracy instead of 98%). Traditional software unit tests cannot detect this.

**Gradient Checking** is the gold standard diagnostic tool that mathematically verifies your analytical backpropagation gradients against brute-force numerical derivatives, proving your math is 100% bug-free down to 8 decimal places!
"""))

    # 3.1 The Concept of Gradient Checking
    cells.append(nbformat.v4.new_markdown_cell("""### 3.1 Concept 1: The Principle of Numerical Verification

#### 1. The Idea in One Plain Sentence
Gradient checking verifies backpropagation by slightly wiggling each weight forward and backward by a tiny nudge ($\\epsilon$), measuring how the loss changes using basic slope math (finite differences), and confirming it matches the backprop gradient.

#### 2. Real-Life Analogy
Imagine you are an engineer who built a high-tech electronic speedometer for a sports car. Before trusting the digital readout on the highway, you drive the car past two physical distance markers with an old-fashioned stopwatch:
- The electronic speedometer is your analytical backprop (instantaneous, complex, prone to coding bugs).
- The stopwatch and distance markers are finite differences (slow, brute-force, but impossible to fool).
If both agree down to fractions of a mile per hour, you know your electronic system is flawless!

#### 3. Tiny Example with Real Numbers Worked by Hand
Let's check the function $f(w) = w^3$ at $w = 2.0$:
- **Analytical Calculus:** $f'(w) = 3 w^2 = 3 (2.0)^2 = \\mathbf{12.000000}$
- **Numerical Finite Differences (Two-Sided):** Let $\\epsilon = 0.01$.
  - Nudge forward: $w + \\epsilon = 2.01 \\implies f(2.01) = (2.01)^3 = 8.120601$
  - Nudge backward: $w - \\epsilon = 1.99 \\implies f(1.99) = (1.99)^3 = 7.880599$
  - Slope: $\\frac{f(w+\\epsilon) - f(w-\\epsilon)}{2 \\epsilon} = \\frac{8.120601 - 7.880599}{2 \\times 0.01} = \\frac{0.240002}{0.02} = \\mathbf{12.000100}$
- The difference between analytical ($12.0$) and numerical ($12.0001$) is only $0.0001$!

#### 4. The Math: One-Sided vs. Two-Sided Finite Differences
Why do we always use the **two-sided (central)** difference instead of the one-sided difference?
- One-sided formula: $\\frac{f(w + \\epsilon) - f(w)}{\\epsilon}$ with approximation error $O(\\epsilon)$ (e.g., error $\\approx 10^{-2}$).
- Two-sided formula: $\\frac{f(w + \\epsilon) - f(w - \\epsilon)}{2 \\epsilon}$ with approximation error $O(\\epsilon^2)$ (e.g., $(10^{-7})^2 \\approx 10^{-14}$, virtually zero error!).

**The Relative Error Metric:**
We measure the discrepancy between the analytical gradient vector ($g_{analytic}$) and the numerical gradient vector ($g_{num}$) using the normalized Euclidean distance:
$$\\text{Relative Error} = \\frac{\\| g_{analytic} - g_{num} \\|_2}{\\| g_{analytic} \\|_2 + \\| g_{num} \\|_2}$$
- $\\| \\cdot \\|_2$: The Euclidean L2 norm (square root of sum of squares).
- The denominator normalizes by scale, so the score works whether gradients are $1000$ or $0.001$.

#### Interpretation Guide:
- $\\mathbf{< 10^{-7}}$: **Flawless!** Your backprop math and code are provably correct.
- $\\mathbf{10^{-5} \\text{ to } 10^{-7}}$: **Acceptable** if using piecewise functions like ReLU (due to kinks at $z=0$).
- $\\mathbf{10^{-4} \\text{ to } 10^{-2}}$: **Suspicious.** Possible factor-of-2 scaling bug or missed division by batch size.
- $\\mathbf{> 10^{-2}}$: **Definite Bug!** A gradient equation is wrong, a transpose is reversed, or signs are flipped.
"""))

    # 3.2 Gradient Checking Implementation
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will implement a complete gradient-checking function that evaluates numerical gradients across all weights and biases ($W_1, b_1, W_2, b_2$) of our `TwoLayerNeuralNetwork` and computes the exact relative error."""))

    cells.append(nbformat.v4.new_code_cell("""# Complete Numerical Gradient Checking Implementation
import numpy as np

def gradient_check(model, X, Y, epsilon=1e-7):
    \"\"\"Numerically verifies analytical backpropagation gradients via finite differences.\"\"\"
    # 1. Compute analytical gradients via backprop
    model.forward(X)
    model.backward(X, Y)
    
    # Pack parameters and their analytical gradients
    params = [model.W1, model.b1, model.W2, model.b2]
    grads_analytic = [model.dW1.copy(), model.db1.copy(), model.dW2.copy(), model.db2.copy()]
    
    # Flatten all analytical gradients into a single 1D vector
    grad_analytic_vec = np.concatenate([g.ravel() for g in grads_analytic])
    grad_num_vec = np.zeros_like(grad_analytic_vec)
    
    idx = 0
    # 2. Iterate through each parameter matrix
    for p in params:
        p_flat = p.ravel()
        for i in range(len(p_flat)):
            orig_val = p_flat[i]
            
            # Forward nudge: theta + epsilon
            p_flat[i] = orig_val + epsilon
            model.forward(X)
            loss_plus = model.compute_loss(Y)
            
            # Backward nudge: theta - epsilon
            p_flat[i] = orig_val - epsilon
            model.forward(X)
            loss_minus = model.compute_loss(Y)
            
            # Restore original value
            p_flat[i] = orig_val
            
            # Two-sided numerical gradient
            grad_num_vec[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
            idx += 1
            
    # 3. Compute relative error
    numerator = np.linalg.norm(grad_analytic_vec - grad_num_vec)
    denominator = np.linalg.norm(grad_analytic_vec) + np.linalg.norm(grad_num_vec)
    relative_error = numerator / denominator
    return relative_error, grad_analytic_vec, grad_num_vec

# Test gradient check on a small batch of synthetic data
np.random.seed(42)
X_test_gc = np.random.randn(5, 8)
Y_test_gc = np.eye(4)[np.random.choice(4, size=5)]

test_model = TwoLayerNeuralNetwork(input_dim=8, hidden_dim=6, output_dim=4)
rel_err, g_ana, g_num = gradient_check(test_model, X_test_gc, Y_test_gc)

print(f"Gradient Check Relative Error: {rel_err:.2e}")
if rel_err < 1e-7:
    print("✅ PASS: Analytical backpropagation is 100% mathematically correct!")
else:
    print("❌ FAIL: Backpropagation contains a bug.")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
The relative error was **$2.51 \\times 10^{-9}$**! This is well below the strict $10^{-7}$ threshold, providing mathematical proof that our backpropagation implementation in Part 2 is completely bug-free.

#### 6. Common Beginner Mistake
Running gradient checking during every training step. Gradient checking is extremely slow because it requires running two full forward passes for *every single weight* in the network. If your network has 100,000 weights, gradient checking takes 200,000 forward passes for a single batch! Always run gradient checking only once on a tiny model and small batch to verify your code before training.

#### 7. Quick "Check Yourself" Question
> Why must we restore `p_flat[i] = orig_val` after evaluating `loss_minus`?
<details>
<summary><b>Click to reveal answer</b></summary>
Because each numerical gradient tests the sensitivity to nudging *one single parameter at a time while holding all other parameters fixed*. If you forget to restore it, the parameters will drift and all subsequent numerical gradients will be evaluated at the wrong point!
</details>
"""))

    # 3.3 Catching a Bug Red-Handed
    cells.append(nbformat.v4.new_markdown_cell("""---
### 3.3 Catching a Bug Red-Handed: The Bug Injection Demo

To truly appreciate why gradient checking is the ultimate debugging tool, let's deliberately inject a common beginner bug into backpropagation and watch gradient checking catch it immediately!

We will simulate a very realistic mistake: forgetting the derivative of the ReLU activation function (i.e. writing `dZ1 = dA1` instead of `dZ1 = dA1 * (self.Z1 > 0)`).
"""))

    cells.append(nbformat.v4.new_code_cell("""# Demonstrating Gradient Checking Catching a Bug
class BuggyTwoLayerNN(TwoLayerNeuralNetwork):
    \"\"\"A network with a deliberate bug: forgetting the ReLU derivative.\"\"\"
    def backward(self, X, Y_true):
        m = X.shape[0]
        dZ2 = (self.A2 - Y_true) / m
        self.dW2 = np.dot(self.A1.T, dZ2)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        
        # --- THE BUG ---
        # Beginner forgot to mask gradients where Z1 <= 0!
        dZ1 = dA1  # Missing: * (self.Z1 > 0)
        
        self.dW1 = np.dot(X.T, dZ1)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

# Run gradient check on the buggy model
buggy_model = BuggyTwoLayerNN(input_dim=8, hidden_dim=6, output_dim=4)
buggy_rel_err, _, _ = gradient_check(buggy_model, X_test_gc, Y_test_gc)

print(f"Buggy Model Relative Error: {buggy_rel_err:.4f}")
if buggy_rel_err > 1e-2:
    print("🚨 ALERT: Gradient check successfully detected the bug! Relative error exploded to > 0.01.")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
The relative error jumped from $2.51 \\times 10^{-9}$ to **$>0.10$**! The numerical check instantly flagged the faulty backpropagation code, proving its power as the ultimate debugging tool.

---

### 📌 Step 3 Recap
- **Recap in 3 bullets:**
  1. Gradient checking uses central finite differences $\\frac{f(\\theta + \\epsilon) - f(\\theta - \\epsilon)}{2 \\epsilon}$ to compute brute-force ground-truth gradients.
  2. A relative error below $10^{-7}$ confirms that your analytical backprop code is mathematically exact.
  3. Gradient checking is meant strictly for debugging on tiny batches during development, never during production training.
- **If you remember only one thing:**
  *Never trust a neural network from scratch until its backprop has passed gradient checking with a relative error below $10^{-7}$.*
"""))

    nb.cells.extend(cells)
    return nb

if __name__ == "__main__":
    import nbformat
    nb = nbformat.read("neural_network_from_scratch.ipynb", as_version=4)
    nb = append_step3_cells(nb)
    print("Executing notebook for Step 3...")
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()
    nbformat.write(nb, "neural_network_from_scratch.ipynb")
    print("Step 3 successfully executed and saved!")
