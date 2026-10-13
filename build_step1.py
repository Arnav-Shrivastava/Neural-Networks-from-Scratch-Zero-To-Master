import nbformat
from nbclient import NotebookClient

def create_step1_notebook():
    nb = nbformat.v4.new_notebook()
    cells = []

    # Title & Banner
    cells.append(nbformat.v4.new_markdown_cell("""# 🧠 Neural Network from Scratch: Zero to Master
### A Complete, First-Principles Guide to Deep Learning in Pure NumPy
*Created for beginners: No black boxes, no PyTorch shortcuts, just pure math, plain English, and step-by-step code.*

---

## 📚 Master Glossary of Neural Network Concepts

Here is our live dictionary. Every time a new concept is introduced in this notebook, its definition is permanently recorded here in plain English.

| Term | One-Sentence Definition | Everyday Analogy |
| :--- | :--- | :--- |
| **Neuron (Perceptron)** | A tiny calculator that takes numbers in, multiplies them by importance dials, adds a baseline nudge, and squashes the sum through an activation curve. | A judge scoring an audition based on voice quality and stage presence. |
| **Weight ($W$)** | A tunable number that controls how much attention or importance a neuron gives to a specific input feature. | A volume knob on an audio mixer. |
| **Bias ($b$)** | A tunable baseline number added to the weighted sum that shifts the neuron's decision threshold independent of the inputs. | A baseline cover charge to enter a club regardless of how many drinks you order. |
| **Weighted Sum ($z$)** | The combined raw score of a neuron before passing through the activation filter: $z = \\sum w_i x_i + b$. | The raw exam score before grading on a curve. |
| **Activation Function ($f$ or $\\sigma$)** | A mathematical curve that bends or squashes the raw weighted sum to introduce non-linearity. | A dimmer switch that smoothly turns raw voltage into light intensity. |
| **Activation ($a$)** | The final output value fired by a neuron after its weighted sum has been processed by an activation function: $a = f(z)$. | The final verdict or brightness emitted. |
| **Layer** | A row or collection of neurons operating in parallel at the same depth inside a neural network. | A committee of inspectors working at the same assembly station. |
| **Forward Pass** | The step-by-step process of passing data through each layer from inputs to outputs to compute a final prediction. | An assembly line where raw materials move forward and become a finished product. |
| **Loss Function ($L$)** | A mathematical penalty score measuring how far the network's prediction is from the true answer. | A referee's score deduction for every mistake made. |
| **Derivative / Gradient** | A number measuring how much the output changes when you make an infinitesimally tiny tweak to an input or weight. | How much faster your car moves when you tap the gas pedal 1 millimeter down. |
| **Chain Rule** | A calculus law that lets us find the sensitivity of the final output to an early input by multiplying the sensitivities of all intermediate links. | A bicycle gear train: pedal turns chainring, chainring turns cassette, cassette turns wheel. |
| **Backpropagation** | An algorithm that runs backward through the network, using the chain rule to distribute blame to every single weight so it knows how to improve. | A team debriefing after a lost match, analyzing which player's mistake caused the defeat. |
| **Learning Rate ($\\eta$)** | A small multiplier (like $0.01$ or $0.1$) that sets how big of a step weights take during an update. | The step size you take while walking downhill in dense fog. |
| **Linear Collapse** | The mathematical certainty that stacking linear layers without activation functions simply collapses into a single linear equation. | Stacking plain glass windows: no matter how many you stack, you never get a colored lens. |
| **Softmax** | An output activation function that converts a vector of raw scores into positive probabilities that sum to exactly $1.0$. | Converting raw votes into percentage shares of an election. |
| **Cross-Entropy Loss** | A loss function for classification that harshly penalizes confident wrong predictions and rewards confident correct predictions. | A teacher giving massive point deductions when a student arrogantly insists on the wrong answer. |
"""))

    # Section 1.0 Introduction
    cells.append(nbformat.v4.new_markdown_cell("""---
## Part 1: Theory in Plain English

Welcome to the foundation of deep learning! In this section, we will build every single core concept of neural networks from the ground up:
1. What a single neuron actually does.
2. How neurons connect into layers to perform a forward pass.
3. How a loss function scores mistakes.
4. Why we strictly need non-linear activation functions (with an algebraic proof of linear collapse).
5. How the calculus chain rule links everything together.
6. A complete, hand-derived backpropagation walkthrough on a 2-input network using real numbers.

Let's begin with the single neuron.
"""))

    # 1.1 The Artificial Neuron
    cells.append(nbformat.v4.new_markdown_cell("""### 1.1 Concept 1: The Artificial Neuron

#### 1. The Idea in One Plain Sentence
An artificial neuron is a tiny calculator that multiplies each incoming number by a tunable dial called a weight, adds a baseline number called a bias, and runs the total through an activation curve to produce a single output number.

#### 2. Real-Life Analogy
Imagine deciding whether to go to an outdoor music concert this weekend:
- Input 1 ($x_1$): How warm and sunny is the weather? (Scale 0 to 5)
- Input 2 ($x_2$): Are your best friends going? (Scale 0 to 5)
- Weight 1 ($w_1$): How much you care about weather. If you hate rain, this knob is turned high.
- Weight 2 ($w_2$): How much you care about social company.
- Bias ($b$): Your natural weekend laziness threshold. If you love staying home, this is a negative penalty that requires great weather and good friends to overcome.

#### 3. Tiny Example with Real Numbers Worked by Hand
Let's plug in numbers:
- Weather input: $x_1 = 2.0$
- Friends input: $x_2 = 3.0$
- Weather weight: $w_1 = 0.5$
- Friends weight: $w_2 = 1.0$
- Laziness bias: $b = -2.5$

**Step 1: Calculate the weighted sum ($z$)**
$$z = (x_1 \\times w_1) + (x_2 \\times w_2) + b$$
$$z = (2.0 \\times 0.5) + (3.0 \\times 1.0) + (-2.5) = 1.0 + 3.0 - 2.5 = 1.5$$

**Step 2: Calculate the activation ($a$) using Sigmoid**
The Sigmoid function squashes any number into a probability between $0$ and $1$:
$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$
$$\\sigma(1.5) = \\frac{1}{1 + e^{-1.5}} \\approx \\frac{1}{1 + 0.22313} = \\frac{1}{1.22313} \\approx 0.8176$$
The neuron fires an output of **0.8176** (an 81.8% certainty of attending the concert!).

#### 4. The Math with Every Symbol Explained in Words
$$z = \\sum_{i=1}^{n} w_i x_i + b = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b$$
$$a = \\sigma(z) = \\frac{1}{1 + e^{-z}}$$
- $x_i$: The $i$-th input feature (a number feeding into the neuron).
- $w_i$: The $i$-th weight (how much influence input $x_i$ has on the decision).
- $b$: The bias (a constant baseline that shifts the threshold up or down).
- $\\sum$: The Greek capital sigma symbol meaning "sum all the multiplied pairs together".
- $z$: The pre-activation weighted sum (the raw combined score).
- $\\sigma$: The Sigmoid activation function that curves the line into a smooth S-shape between 0 and 1.
- $e$: Euler's mathematical constant (approximately $2.71828$).
- $a$: The final post-activation output emitted by the neuron.
"""))

    # Code cell 1: Single neuron forward pass
    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will implement this single artificial neuron in Python using NumPy to compute the exact concert decision from our hand calculation above."""))

    cells.append(nbformat.v4.new_code_cell("""# Single artificial neuron forward computation
import numpy as np

# Inputs: weather score (2.0), friends score (3.0)
x = np.array([2.0, 3.0])  # Shape: (2,)

# Tunable dials: weights and bias
w = np.array([0.5, 1.0])  # Shape: (2,)
b = -2.5

# Step 1: Compute weighted sum z = w . x + b
z = np.dot(w, x) + b

# Step 2: Squash through sigmoid activation
a = 1.0 / (1.0 + np.exp(-z))

print(f"Weighted sum (z): {z:.4f}")
print(f"Neuron activation (a): {a:.4f}")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
NumPy computed $z = 1.5000$ and $a = 0.8176$, confirming our hand-calculated numbers to four decimal places.

#### 6. Common Beginner Mistake
Thinking the bias $b$ is optional. Without the bias, the weighted sum is strictly $w_1 x_1 + w_2 x_2$. If all inputs are zero ($x_1 = 0, x_2 = 0$), $z$ is forced to be exactly zero, and the sigmoid output is forced to be $0.5$. The bias frees the neuron, allowing it to output high or low values even when inputs are zero or small!

#### 7. Quick "Check Yourself" Question
> If $x = [1.0, 2.0]$, $w = [-1.0, 0.5]$, and $b = 1.0$, what is the weighted sum $z$?
<details>
<summary><b>Click to reveal answer</b></summary>
$z = (1.0 \\times -1.0) + (2.0 \\times 0.5) + 1.0 = -1.0 + 1.0 + 1.0 = 1.0$.
Then $\\sigma(1.0) = 1 / (1 + e^{-1}) \\approx 0.7311$.
</details>
"""))

    # 1.2 Layers and the Forward Pass
    cells.append(nbformat.v4.new_markdown_cell("""---
### 1.2 Concept 2: Layers and the Forward Pass

#### 1. The Idea in One Plain Sentence
A neural network groups multiple neurons side-by-side into a layer and connects layers in a chain, so the outputs of one layer become the inputs to the next in a forward pass.

#### 2. Real-Life Analogy
An automotive assembly line:
- Station 1 (Input): Receives sheet metal and engine parts.
- Station 2 (Hidden Layer): Inspectors assemble the frame and check structural integrity.
- Station 3 (Output Layer): The final inspection team assigns the quality rating (Pass, Rework, Reject).

#### 3. 5-Line Prerequisite Refresher: Matrix Multiplication
> 1. In matrix multiplication, each row of the first matrix takes the dot product with each column of the second matrix.
> 2. If matrix $A$ has shape $(M, K)$ and matrix $B$ has shape $(K, N)$, the output shape is $(M, N)$.
> 3. Inner dimensions must match: $(M, \\mathbf{K}) \\times (\\mathbf{K}, N) \\rightarrow (M, N)$.
> 4. Tiny example: $[1, 2] \\times \\begin{bmatrix} 3 \\\\ 4 \\end{bmatrix} = 1(3) + 2(4) = 11$.
> 5. Matrix math allows us to calculate 10,000 samples across 100 neurons in one single instant on modern CPUs and GPUs!

#### 4. Tiny Example with Real Numbers Worked by Hand
Let's take a batch of 1 sample with 2 input features: $X = \\begin{bmatrix} 2.0 & 3.0 \\end{bmatrix}$ (shape $1 \\times 2$).
Let's feed this into a hidden layer with 2 neurons.
The weight matrix $W$ needs shape $(2, 2)$ where column 0 is neuron 1's weights and column 1 is neuron 2's weights:
$$W = \\begin{bmatrix} 0.5 & -0.2 \\\\ 1.0 & 0.4 \\end{bmatrix}, \\quad b = \\begin{bmatrix} -2.5 & 0.5 \\end{bmatrix}$$

Calculate $Z = XW + b$:
- For Neuron 1: $(2.0 \\times 0.5) + (3.0 \\times 1.0) - 2.5 = 1.0 + 3.0 - 2.5 = 1.5$.
- For Neuron 2: $(2.0 \\times -0.2) + (3.0 \\times 0.4) + 0.5 = -0.4 + 1.2 + 0.5 = 1.3$.
$$Z = \\begin{bmatrix} 1.5 & 1.3 \\end{bmatrix}$$

#### 5. The Math with Every Symbol Explained in Words
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = f(Z^{[l]})$$
- $l$: The current layer index ($l=1$ is first hidden layer).
- $A^{[l-1]}$: The activation outputs from the previous layer (shape: $N_{samples} \\times N_{in}$). For $l=1$, this is the input matrix $X$.
- $W^{[l]}$: The weight matrix of layer $l$ (shape: $N_{in} \\times N_{out}$).
- $b^{[l]}$: The bias row vector of layer $l$ (shape: $1 \\times N_{out}$, broadcast across all samples).
- $Z^{[l]}$: The linear weighted sum matrix (shape: $N_{samples} \\times N_{out}$).
- $f$: The activation function applied element-wise.
- $A^{[l]}$: The activated outputs of layer $l$ (shape: $N_{samples} \\times N_{out}$).
"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will implement this 2-neuron layer in NumPy using matrix operations with explicit shape annotations in comments to show how batch processing works."""))

    cells.append(nbformat.v4.new_code_cell("""# 2-neuron layer forward pass in NumPy
import numpy as np

# 1 sample with 2 input features: Shape (1, 2)
X = np.array([[2.0, 3.0]])

# Weights connecting 2 inputs to 2 hidden neurons: Shape (2, 2)
W = np.array([[0.5, -0.2],
              [1.0,  0.4]])

# Biases for 2 neurons: Shape (1, 2)
b = np.array([[-2.5, 0.5]])

# Matrix multiply and broadcast bias: (1, 2) @ (2, 2) + (1, 2) -> (1, 2)
Z = np.matmul(X, W) + b
A = 1.0 / (1.0 + np.exp(-Z))

print("Z pre-activations:", Z)
print("A post-activations:", np.round(A, 4))"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
The matrix multiplication produced $Z = [[1.5, 1.3]]$ and $A = [[0.8176, 0.7858]]$, processing both neurons simultaneously in one operation!

#### 6. Common Beginner Mistake
Trying to multiply matrices with inverted shapes: writing `np.matmul(W, X)` instead of `np.matmul(X, W)`. Since $X$ is $(1, 2)$ and $W$ is $(2, 2)$, $X W$ works ($2=2$), but $W X$ throws a `ValueError: shapes (2,2) and (1,2) not aligned`.

#### 7. Quick "Check Yourself" Question
> If you have a batch of 32 images, each flattened into 64 pixels, and feed them into a layer with 16 neurons, what are the shapes of $X$, $W$, $b$, and $Z$?
<details>
<summary><b>Click to reveal answer</b></summary>
$X$ is $(32, 64)$, $W$ is $(64, 16)$, $b$ is $(1, 16)$, and $Z$ is $(32, 16)$.
</details>
"""))

    # 1.3 The Loss Function
    cells.append(nbformat.v4.new_markdown_cell("""---
### 1.3 Concept 3: The Loss Function (Measuring How Wrong We Are)

#### 1. The Idea in One Plain Sentence
A loss function is a mathematical scoreboard that compares what the network predicted against the true reality and computes a single positive penalty score, where lower is better.

#### 2. Real-Life Analogy
An archery target: if your arrow lands dead in the bullseye (perfect prediction), your penalty distance is $0$. The further your arrow lands from the center, the larger your penalty.

#### 3. 5-Line Prerequisite Refresher: Natural Logarithms
> 1. The natural logarithm $\\ln(p)$ asks: "To what power must we raise $e \\approx 2.718$ to get $p$?"
> 2. When probability $p = 1.0$, $\\ln(1.0) = 0.0$ (no surprise, zero loss).
> 3. As probability $p \\rightarrow 0$, $\\ln(p) \\rightarrow -\\infty$ (huge penalty for failing to predict the true class).
> 4. We negate it: $-\\ln(1.0) = 0$, $-\\ln(0.8) \\approx 0.223$, $-\\ln(0.1) \\approx 2.302$, $-\\ln(0.001) \\approx 6.908$.
> 5. This makes Categorical Cross-Entropy ideal for classification!

#### 4. Tiny Example with Real Numbers Worked by Hand
Imagine we are classifying digits. The image is a true digit '7'.
Its true label in one-hot encoding is: $y = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]$ (a $1$ at index 7).

**Case A: The network is confident and correct**
- Model predicts probability for class 7: $\\hat{y}_7 = 0.90$.
- Loss: $L = -\\ln(0.90) \\approx 0.105$. (Low penalty!)

**Case B: The network is confident and completely wrong**
- Model predicts probability for class 7: $\\hat{y}_7 = 0.05$.
- Loss: $L = -\\ln(0.05) \\approx 2.996$. (Severe penalty!)

#### 5. The Math with Every Symbol Explained in Words
**Categorical Cross-Entropy Loss for a single sample:**
$$L = - \\sum_{k=1}^{K} y_k \\ln(\\hat{y}_k)$$
- $K$: Total number of classes (e.g., 10 for digits 0 through 9).
- $k$: Class index from 1 to $K$.
- $y_k$: The true probability for class $k$ (which is $1$ for the correct class and $0$ for all other classes).
- $\\hat{y}_k$: The predicted probability for class $k$ produced by our Softmax output layer.
- $\\ln$: The natural logarithm.
- Because $y_k = 0$ for all wrong classes, the formula simplifies to just $-\\ln(\\hat{y}_{correct})$!
"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will implement Categorical Cross-Entropy loss in NumPy and compute the penalty for both confident and unconfident predictions to see the math in action."""))

    cells.append(nbformat.v4.new_code_cell("""# Categorical Cross-Entropy Loss computation in NumPy
import numpy as np

# True class is index 2: One-hot vector [0, 0, 1]
y_true = np.array([0.0, 0.0, 1.0])

# Scenario 1: Model is confident and correct
pred_good = np.array([0.05, 0.05, 0.90])
loss_good = -np.sum(y_true * np.log(pred_good))

# Scenario 2: Model is wrong
pred_bad = np.array([0.70, 0.25, 0.05])
loss_bad = -np.sum(y_true * np.log(pred_bad))

print(f"Good prediction loss: {loss_good:.4f}")
print(f"Bad prediction loss:  {loss_bad:.4f}")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
The good prediction yielded a tiny loss of $0.1054$, while the bad prediction yielded a huge loss of $2.9957$.

#### 6. Common Beginner Mistake
Using classification accuracy (e.g., 90% correct) as the loss function. Accuracy has flat plateaus—a tiny nudge to a weight does not change whether an answer crossed the 50% threshold, so the gradient is $0$ almost everywhere. Cross-entropy provides smooth, continuous gradients for every tiny improvement in confidence.

#### 7. Quick "Check Yourself" Question
> If a model predicts a probability of $0.5$ for the true class, what is the cross-entropy loss? ($-\\ln(0.5)$)
<details>
<summary><b>Click to reveal answer</b></summary>
$-\\ln(0.5) = \\ln(2) \\approx 0.6931$. This is the classic "coin flip" loss value you frequently see at the beginning of training!
</details>
"""))

    # 1.4 Why We Need Non-Linear Activations (Linear Collapse Proof)
    cells.append(nbformat.v4.new_markdown_cell("""---
### 1.4 Concept 4: Why We Need Non-Linear Activations (The Linear Collapse Proof)

#### 1. The Idea in One Plain Sentence
Without non-linear activation functions in between layers, stacking any number of linear layers mathematically collapses into a single linear equation, rendering a 100-layer deep network no more powerful than a simple single-layer model.

#### 2. Real-Life Analogy
Imagine magnifying glasses with fixed zoom factors. If Lens 1 doubles the image ($2\\times$) and Lens 2 triples the image ($3\\times$), stacking them together is strictly equivalent to using a single $6\\times$ lens ($2 \\times 3 = 6$). No matter how many flat glass lenses you stack, you can never bend the light into a fish-eye curve or complex non-linear distortion.

#### 3. Tiny Example with Real Numbers Worked by Hand
Let's build a 2-layer network with NO activation functions:
- Layer 1: $h = 2x + 1$ (weight = 2, bias = 1)
- Layer 2: $y = 3h + 4$ (weight = 3, bias = 4)

Let's plug Layer 1 directly into Layer 2:
$$y = 3(2x + 1) + 4$$
$$y = 6x + 3 + 4 = 6x + 7$$
Notice that $y = 6x + 7$ is just a single linear layer with combined weight $w_{comb} = 6$ and combined bias $b_{comb} = 7$! Stacking the two layers achieved zero added representational power.

#### 4. The Formal Algebraic Proof (The Linear Collapse Theorem)
Let's prove this for general matrices of any dimension:
- Let Layer 1 compute: $h = x W_1 + b_1$ (where $x \\in \\mathbb{R}^{1 \\times D_{in}}, W_1 \\in \\mathbb{R}^{D_{in} \\times H}, b_1 \\in \\mathbb{R}^{1 \\times H}$)
- Let Layer 2 compute: $y = h W_2 + b_2$ (where $W_2 \\in \\mathbb{R}^{H \\times D_{out}}, b_2 \\in \\mathbb{R}^{1 \\times D_{out}}$)

**Step 1:** Substitute $h$ into the equation for $y$:
$$y = (x W_1 + b_1) W_2 + b_2 \\quad \\text{[Plain English: Replace hidden output } h \\text{ with its definition]}$$

**Step 2:** Distribute matrix $W_2$:
$$y = x (W_1 W_2) + (b_1 W_2 + b_2) \\quad \\text{[Plain English: Multiply } W_2 \\text{ across the addition inside the parentheses]}$$

**Step 3:** Define new consolidated parameters:
$$W_{comb} = W_1 W_2 \\quad \\text{[Plain English: The product of two matrices is just another single matrix of shape } D_{in} \\times D_{out}\\text{]}$$
$$b_{comb} = b_1 W_2 + b_2 \\quad \\text{[Plain English: A row vector multiplied by a matrix plus a vector is just another single row vector } 1 \\times D_{out}\\text{]}$$

**Step 4:** Result:
$$y = x W_{comb} + b_{comb} \\quad \\text{[Q.E.D. Stacking } N \\text{ linear layers collapses to 1 linear layer]}$$
Because the composition of linear transformations is always linear, a network without non-linear activations cannot learn curves, decision boundaries, or non-linear patterns like XOR! Non-linear activations (like ReLU, Sigmoid, Tanh) bend the space at each layer, enabling deep networks to approximate any continuous function.
"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will prove linear collapse computationally in NumPy by creating a 3-layer linear network, computing its collapsed single-layer parameters $W_{comb}$ and $b_{comb}$, and showing both give identical outputs for random inputs."""))

    cells.append(nbformat.v4.new_code_cell("""# Computational proof of Linear Collapse in NumPy
import numpy as np
np.random.seed(42)

# Random input: 1 sample with 3 features
x = np.random.randn(1, 3)

# 3 linear layers without activation functions
W1, b1 = np.random.randn(3, 4), np.random.randn(1, 4)
W2, b2 = np.random.randn(4, 5), np.random.randn(1, 5)
W3, b3 = np.random.randn(5, 2), np.random.randn(1, 2)

# Pass through all 3 layers step by step
h1 = x @ W1 + b1
h2 = h1 @ W2 + b2
y_3layers = h2 @ W3 + b3

# Calculate collapsed single-layer parameters analytically
W_comb = W1 @ W2 @ W3
b_comb = ((b1 @ W2 + b2) @ W3) + b3
y_1layer = x @ W_comb + b_comb

# Check if outputs match
max_diff = np.max(np.abs(y_3layers - y_1layer))
print(f"Max difference between 3-layer network and collapsed 1-layer: {max_diff:.2e}")
print(f"Are they identical? {np.allclose(y_3layers, y_1layer)}")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
The difference is $0.00 \\times 10^0$ (strictly zero within machine precision). This proves empirically that adding layers without non-linearities adds zero representational power.

#### 6. Common Beginner Mistake
Thinking that making a network "wider" or "deeper" automatically makes it more capable. Without non-linear activations (like ReLU or Sigmoid), a 1000-layer network is strictly a linear regression model that can only draw straight lines or flat planes.

#### 7. Quick "Check Yourself" Question
> Can a 5-layer neural network with no activation functions solve the XOR problem?
<details>
<summary><b>Click to reveal answer</b></summary>
No! XOR is a classic non-linearly separable problem. Since a 5-layer linear network collapses into a single linear classifier, it can only draw a straight line, which cannot separate XOR points.
</details>
"""))

    # 1.5 The Chain Rule
    cells.append(nbformat.v4.new_markdown_cell("""---
### 1.5 Concept 5: The Chain Rule (The Calculus Engine)

#### 1. The Idea in One Plain Sentence
The chain rule is the calculus formula for finding how much a tweak at the very start of a chain of events affects the final output, by multiplying together the sensitivity of each intermediate link.

#### 2. Real-Life Analogy
A bicycle gear train:
- When you turn the pedal by 1 rotation, the front chainring turns 2 rotations ($Rate_1 = 2$).
- The chainring turns the rear wheel cassette by 3 rotations for every 1 front rotation ($Rate_2 = 3$).
- How much does the rear wheel turn when you turn the pedal by 1 rotation?
- Total rate = $Rate_1 \\times Rate_2 = 2 \\times 3 = 6$ rotations! You multiply the local transmission rates along the chain.

#### 3. 5-Line Prerequisite Refresher: Derivatives
> 1. A derivative $\\frac{dy}{dx}$ tells you: "If I nudge $x$ by a tiny amount $\\Delta x$, by how much does $y$ change?"
> 2. Linear rule: If $y = 3x$, then $\\frac{dy}{dx} = 3$ (output changes 3 times faster than input).
> 3. Power rule: If $y = x^2$, then $\\frac{dy}{dx} = 2x$. At $x=4$, $\\frac{dy}{dx} = 8$.
> 4. Sigmoid derivative: If $a = \\sigma(z)$, then $\\frac{da}{dz} = a(1 - a)$.
> 5. Derivatives are simply local multipliers of sensitivity!

#### 4. Tiny Example with Real Numbers Worked by Hand
Let's consider a chained mathematical function:
$$u = 2x, \\quad y = u^2$$
Let's evaluate the sensitivity $\\frac{dy}{dx}$ at $x = 3.0$:

- **Step 1: Compute intermediate values**
  - $x = 3.0$
  - $u = 2 \\times 3.0 = 6.0$
  - $y = (6.0)^2 = 36.0$

- **Step 2: Compute local derivative 1**
  - $\\frac{du}{dx} = 2$ (how much $u$ changes when $x$ nudges)

- **Step 3: Compute local derivative 2**
  - $\\frac{dy}{du} = 2u = 2(6.0) = 12$ (how much $y$ changes when $u$ nudges)

- **Step 4: Multiply using the Chain Rule**
  - $\\frac{dy}{dx} = \\frac{dy}{du} \\times \\frac{du}{dx} = 12 \\times 2 = 24.0$

**Numerical verification check:**
Let's nudge $x$ by $\\Delta x = 0.001$ so $x_{new} = 3.001$:
- $u_{new} = 2 \\times 3.001 = 6.002$
- $y_{new} = (6.002)^2 = 36.024004$
- Change in $y$: $\\Delta y = y_{new} - y = 36.024004 - 36.0 = 0.024004$
- Finite difference ratio: $\\frac{\\Delta y}{\\Delta x} = \\frac{0.024004}{0.001} = 24.004 \\approx 24.0$! Exact match!

#### 5. The Math with Every Symbol Explained in Words
$$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w}$$
- $\\frac{\\partial L}{\\partial w}$: How much the final loss $L$ changes when weight $w$ is nudged (what we need to update the weight).
- $\\frac{\\partial L}{\\partial a}$: How much loss changes when neuron activation $a$ changes (error penalty sensitivity).
- $\\frac{\\partial a}{\\partial z}$: How much activation $a$ changes when pre-activation $z$ changes (derivative of the activation function).
- $\\frac{\\partial z}{\\partial w}$: How much pre-activation $z$ changes when weight $w$ changes (which is simply the input $x$!).
"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will verify this exact chain rule derivative in NumPy and compare it to numerical finite differences to prove that multiplying local sensitivities gives the exact total rate of change."""))

    cells.append(nbformat.v4.new_code_cell("""# Chain rule numerical demonstration in NumPy
import numpy as np

x_val = 3.0
epsilon = 1e-5

# Forward function: y = (2x)^2
def forward(x):
    u = 2.0 * x
    return u ** 2

# Analytical derivative using Chain Rule: dy/dx = (dy/du) * (du/dx) = 2*(2x) * 2 = 8x
# At x = 3.0: 8 * 3.0 = 24.0
u_val = 2.0 * x_val
dy_du = 2.0 * u_val
du_dx = 2.0
analytical_grad = dy_du * du_dx

# Numerical derivative: finite differences [f(x + eps) - f(x - eps)] / (2 * eps)
numerical_grad = (forward(x_val + epsilon) - forward(x_val - epsilon)) / (2.0 * epsilon)

print(f"Chain rule analytical gradient: {analytical_grad:.6f}")
print(f"Finite difference numerical grad: {numerical_grad:.6f}")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
Both analytical chain rule gradient and numerical gradient evaluated to $24.000000$, proving the chain rule calculates the exact slope.

#### 6. Common Beginner Mistake
Adding derivatives along a path instead of multiplying them. Remember: when one variable feeds into the next in a sequential chain ($x \\rightarrow u \\rightarrow y$), their rates of change multiply, because each step amplifies or dampens the change from the previous step.

#### 7. Quick "Check Yourself" Question
> If $\\frac{\\partial L}{\\partial a} = 4.0$, $\\frac{\\partial a}{\\partial z} = 0.25$, and $\\frac{\\partial z}{\\partial w} = 2.0$, what is $\\frac{\\partial L}{\\partial w}$?
<details>
<summary><b>Click to reveal answer</b></summary>
$\\frac{\\partial L}{\\partial w} = 4.0 \\times 0.25 \\times 2.0 = 1.0 \\times 2.0 = 2.0$.
</details>
"""))

    # 1.6 Complete Hand-Calculated Backpropagation
    cells.append(nbformat.v4.new_markdown_cell("""---
### 1.6 Concept 6: Backpropagation Derived by Hand Step-by-Step for a Tiny 2-Input Network

Now we put everything together into the holy grail of deep learning: **Backpropagation**.
We will take a tiny neural network with 2 inputs, 1 hidden neuron, and 1 output neuron, and calculate **every single forward value, derivative, and weight update entirely by hand with real numbers**. No matrix abstractions, no hidden steps!

#### Network Architecture & Setup
- **Inputs:** $x_1 = 0.5, \\quad x_2 = 1.0$
- **Target Value:** $y = 1.0$ (we want the network to output 1.0)
- **Hidden Neuron ($H$):**
  - Weights: $w_1 = 0.4, \\quad w_2 = -0.2$
  - Bias: $b_1 = 0.1$
  - Activation: Sigmoid $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
- **Output Neuron ($O$):**
  - Weight: $w_3 = 0.7$
  - Bias: $b_2 = -0.3$
  - Activation: Sigmoid $\\sigma(z) = \\frac{1}{1 + e^{-z}}$
- **Loss Function:** Squared Error $L = \\frac{1}{2}(\\hat{y} - y)^2$
- **Learning Rate:** $\\eta = 0.5$

```
  x1 = 0.5 ---(w1 = 0.4)---\\
                           +---> [ Hidden Neuron H ] ---(w3 = 0.7)---> [ Output Neuron O ] ---> y_hat
  x2 = 1.0 --(w2 = -0.2)---/     bias b1 = 0.1                         bias b2 = -0.3
```

---

#### 1. Forward Pass (Step by Step with Real Numbers)

**Step F1: Hidden Neuron Pre-Activation ($z_1$)**
$$z_1 = w_1 x_1 + w_2 x_2 + b_1$$
$$z_1 = (0.4 \\times 0.5) + (-0.2 \\times 1.0) + 0.10 = 0.20 - 0.20 + 0.10 = \\mathbf{0.1000}$$

**Step F2: Hidden Neuron Activation ($a_1$)**
$$a_1 = \\sigma(z_1) = \\frac{1}{1 + e^{-0.10}} = \\frac{1}{1 + 0.904837} = \\frac{1}{1.904837} = \\mathbf{0.524979}$$

**Step F3: Output Neuron Pre-Activation ($z_2$)**
$$z_2 = w_3 a_1 + b_2$$
$$z_2 = (0.7 \\times 0.524979) + (-0.3) = 0.367485 - 0.30 = \\mathbf{0.067485}$$

**Step F4: Output Prediction ($\\hat{y} = a_2$)**
$$\\hat{y} = a_2 = \\sigma(z_2) = \\frac{1}{1 + e^{-0.067485}} = \\frac{1}{1 + 0.934743} = \\frac{1}{1.934743} = \\mathbf{0.516864}$$

**Step F5: Compute Loss ($L$)**
The network guessed $0.516864$, but the target is $1.0$:
$$\\text{Error} = (a_2 - y) = 0.516864 - 1.0 = -0.483136$$
$$L = \\frac{1}{2}(a_2 - y)^2 = \\frac{1}{2}(-0.483136)^2 = \\frac{1}{2}(0.233420) = \\mathbf{0.116710}$$

---

#### 2. Backward Pass (Passing Blame Backward Step by Step)

Now we compute the gradients $\\frac{\\partial L}{\\partial w_3}$, $\\frac{\\partial L}{\\partial b_2}$, $\\frac{\\partial L}{\\partial w_1}$, $\\frac{\\partial L}{\\partial w_2}$, and $\\frac{\\partial L}{\\partial b_1}$.

##### Phase A: Gradients at the Output Neuron ($O$)

**Step B1: Derivative of Loss with respect to Output Activation ($a_2$)**
$$\\frac{\\partial L}{\\partial a_2} = \\frac{\\partial}{\\partial a_2} \\left[ \\frac{1}{2}(a_2 - y)^2 \\right] = (a_2 - y) \\quad \\text{[Plain English: How much loss changes when output moves]}$$
$$\\frac{\\partial L}{\\partial a_2} = 0.516864 - 1.0 = \\mathbf{-0.483136}$$

**Step B2: Derivative of Output Activation with respect to Pre-Activation ($z_2$)**
For Sigmoid, $\\frac{d\\sigma(z)}{dz} = \\sigma(z)(1 - \\sigma(z)) = a_2(1 - a_2)$:
$$\\frac{\\partial a_2}{\\partial z_2} = a_2(1 - a_2) = 0.516864 \\times (1 - 0.516864) = 0.516864 \\times 0.483136 = \\mathbf{0.249714}$$

**Step B3: Output Error Term ($\\delta_2$)**
Using the chain rule, $\\delta_2 = \\frac{\\partial L}{\\partial z_2} = \\frac{\\partial L}{\\partial a_2} \\times \\frac{\\partial a_2}{\\partial z_2}$:
$$\\delta_2 = -0.483136 \\times 0.249714 = \\mathbf{-0.120646} \\quad \\text{[Plain English: The net blame arriving at output pre-activation } z_2\\text{]}$$

**Step B4: Gradient for Output Weight ($w_3$)**
Since $z_2 = w_3 a_1 + b_2$, the local derivative $\\frac{\\partial z_2}{\\partial w_3} = a_1$:
$$\\frac{\\partial L}{\\partial w_3} = \\delta_2 \\times a_1 = -0.120646 \\times 0.524979 = \\mathbf{-0.063337}$$

**Step B5: Gradient for Output Bias ($b_2$)**
Since $\\frac{\\partial z_2}{\\partial b_2} = 1$:
$$\\frac{\\partial L}{\\partial b_2} = \\delta_2 \\times 1 = \\mathbf{-0.120646}$$

---

##### Phase B: Backpropagating Blame to the Hidden Neuron ($H$)

**Step B6: How much blame travels back to Hidden Activation ($a_1$)**
The connection from hidden neuron $H$ to output neuron $O$ is through weight $w_3$. Therefore $\\frac{\\partial z_2}{\\partial a_1} = w_3$:
$$\\frac{\\partial L}{\\partial a_1} = \\delta_2 \\times w_3 = -0.120646 \\times 0.7 = \\mathbf{-0.084452} \\quad \\text{[Plain English: Blame transmitted backward across wire } w_3\\text{]}$$

**Step B7: Derivative of Hidden Activation with respect to Hidden Pre-Activation ($z_1$)**
$$\\frac{\\partial a_1}{\\partial z_1} = a_1(1 - a_1) = 0.524979 \\times (1 - 0.524979) = 0.524979 \\times 0.475021 = \\mathbf{0.249376}$$

**Step B8: Hidden Error Term ($\\delta_1$)**
$$\\delta_1 = \\frac{\\partial L}{\\partial z_1} = \\frac{\\partial L}{\\partial a_1} \\times \\frac{\\partial a_1}{\\partial z_1} = -0.084452 \\times 0.249376 = \\mathbf{-0.021060}$$

**Step B9: Gradient for Hidden Weight $w_1$**
Since $z_1 = w_1 x_1 + w_2 x_2 + b_1$, the local derivative is $x_1$:
$$\\frac{\\partial L}{\\partial w_1} = \\delta_1 \\times x_1 = -0.021060 \\times 0.5 = \\mathbf{-0.010530}$$

**Step B10: Gradient for Hidden Weight $w_2$**
$$\\frac{\\partial L}{\\partial w_2} = \\delta_1 \\times x_2 = -0.021060 \\times 1.0 = \\mathbf{-0.021060}$$

**Step B11: Gradient for Hidden Bias $b_1$**
$$\\frac{\\partial L}{\\partial b_1} = \\delta_1 \\times 1 = \\mathbf{-0.021060}$$

---

#### 3. Gradient Descent Weight Update ($\\eta = 0.5$)
Every weight moves opposite to its gradient: $w_{new} = w_{old} - \\eta \\frac{\\partial L}{\\partial w}$.
- $w_3^{new} = 0.7 - [0.5 \\times (-0.063337)] = 0.7 + 0.031669 = \\mathbf{0.731669}$
- $b_2^{new} = -0.3 - [0.5 \\times (-0.120646)] = -0.3 + 0.060323 = \\mathbf{-0.239677}$
- $w_1^{new} = 0.4 - [0.5 \\times (-0.010530)] = 0.4 + 0.005265 = \\mathbf{0.405265}$
- $w_2^{new} = -0.2 - [0.5 \\times (-0.021060)] = -0.2 + 0.010530 = \\mathbf{-0.189470}$
- $b_1^{new} = 0.1 - [0.5 \\times (-0.021060)] = 0.1 + 0.010530 = \\mathbf{0.110530}$

---

#### 4. The Grand Finale: Proof That the Network Learned!
Let's run a new forward pass with these updated weights to verify that the loss dropped:
- $z_1^{new} = (0.405265 \\times 0.5) + (-0.189470 \\times 1.0) + 0.110530 = 0.202633 - 0.189470 + 0.110530 = 0.123693$
- $a_1^{new} = \\sigma(0.123693) = 0.530881$
- $z_2^{new} = (0.731669 \\times 0.530881) + (-0.239677) = 0.388430 - 0.239677 = 0.148753$
- $\\hat{y}^{new} = \\sigma(0.148753) = \\mathbf{0.537120}$ (Moved closer to 1.0 from 0.516864!)
- **New Loss:** $L^{new} = \\frac{1}{2}(0.537120 - 1.0)^2 = \\frac{1}{2}(-0.462880)^2 = \\mathbf{0.107129}$

**Loss decreased from $0.116710 \\rightarrow 0.107129$ (an 8.2% error reduction in a single step!). The math works!**
"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we are about to do and why:**
We will now run this exact hand-calculated forward pass, backpropagation, and weight update in NumPy to verify every single variable and prove machine-calculated numbers match our hand math to 6 decimal places."""))

    cells.append(nbformat.v4.new_code_cell("""# Complete Backprop Verification in NumPy matching Hand Calculations
import numpy as np

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

# 1. Inputs and Target
x1, x2, y_target, lr = 0.5, 1.0, 1.0, 0.5
w1, w2, b1 = 0.4, -0.2, 0.1
w3, b2 = 0.7, -0.3

# 2. Forward Pass
z1 = w1 * x1 + w2 * x2 + b1
a1 = sigmoid(z1)
z2 = w3 * a1 + b2
a2 = sigmoid(z2)
loss_initial = 0.5 * (a2 - y_target)**2

# 3. Backward Pass
delta2 = (a2 - y_target) * (a2 * (1.0 - a2))
dL_dw3 = delta2 * a1
dL_db2 = delta2
delta1 = (delta2 * w3) * (a1 * (1.0 - a1))
dL_dw1 = delta1 * x1
dL_dw2 = delta1 * x2
dL_db1 = delta1

# 4. Weight Update
w3_up = w3 - lr * dL_dw3; b2_up = b2 - lr * dL_db2
w1_up = w1 - lr * dL_dw1; w2_up = w2 - lr * dL_dw2; b1_up = b1 - lr * dL_db1

# 5. New Forward Pass
loss_after = 0.5 * (sigmoid(w3_up * sigmoid(w1_up*x1 + w2_up*x2 + b1_up) + b2_up) - y_target)**2

print(f"Initial Loss: {loss_initial:.6f} | Loss After 1 Step: {loss_after:.6f}")
print(f"dL/dw3: {dL_dw3:.6f} | delta2: {delta2:.6f} | delta1: {delta1:.6f}")"""))

    cells.append(nbformat.v4.new_markdown_cell("""**What we just saw:**
NumPy produced:
- Initial Loss: $0.116710$
- Loss After 1 Step: $0.107129$
- $\\frac{\\partial L}{\\partial w_3} = -0.063337$
- $\\delta_2 = -0.120646$
- $\\delta_1 = -0.021060$
Every single digit matches our step-by-step hand calculation perfectly!

#### 6. Common Beginner Mistake
Trying to update weights during backpropagation before computing the gradients for earlier layers. Notice in Step B6 that we needed the OLD weight $w_3 = 0.7$ to pass blame backward to neuron 1. If you update $w_3$ first, you corrupt the backward message! Always compute all gradients first, and update all weights after.

#### 7. Quick "Check Yourself" Question
> If the error $(a_2 - y)$ is negative (meaning prediction is smaller than target), why does the gradient $\\frac{\\partial L}{\\partial w_3}$ come out negative, and what happens to the weight when we update it?
<details>
<summary><b>Click to reveal answer</b></summary>
Because prediction was too small, increasing $w_3$ will increase the prediction and decrease the loss. A negative gradient means the slope goes downhill as $w_3$ increases. When we subtract a negative gradient ($w - \\eta(-\\text{grad})$), we ADD to the weight, increasing it in the exact direction needed!
</details>

---

### 📌 Step 1 Recap
- **Recap in 3 bullets:**
  1. An artificial neuron computes a linear combination $z = \\sum w_i x_i + b$ and squashes it through a non-linear activation $a = f(z)$.
  2. Non-linear activations are mathematically essential; without them, any multi-layer network collapses algebraically to a single linear model ($y = x W_{comb} + b_{comb}$).
  3. Backpropagation is nothing more than the calculus chain rule applied from output to input, computing how much each dial contributed to the total loss so gradient descent can improve it.
- **If you remember only one thing:**
  *Gradient descent is walking downhill on the loss landscape, and backpropagation is the chain-rule messenger that tells each weight which way is downhill.*
"""))

    nb.cells.extend(cells)
    return nb

if __name__ == "__main__":
    nb = create_step1_notebook()
    print("Executing notebook for Step 1...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()
    nbformat.write(nb, "neural_network_from_scratch.ipynb")
    print("Notebook successfully generated and executed with outputs saved!")
