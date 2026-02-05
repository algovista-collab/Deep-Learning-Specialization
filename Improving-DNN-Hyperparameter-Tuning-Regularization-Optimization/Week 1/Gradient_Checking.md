# Vanishing or Exploding Gradients

When training a **very deep neural network** (NN), the derivatives (slopes) of the cost function with respect to the early layer weights can sometimes become **very, very large (exploding)** or **very, very small (vanishing)**. This instability makes training extremely difficult or slow.

## The Mechanism

The magnitude of the activations and, consequently, the gradients, often grows or shrinks exponentially with the depth of the network ($L$).

### Illustration (Simplified Deep NN)

Consider a deep neural network with $L$ layers, two neurons per layer, and using a linear activation function ($g(z) = z$) with bias $b^{[l]}=0$.

The output of layer 3 is:

$$
a^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} = \mathbf{W}^{[3]} (\mathbf{W}^{[2]} \mathbf{a}^{[1]}) = \mathbf{W}^{[3]} \mathbf{W}^{[2]} \mathbf{W}^{[1]} \mathbf{x}
$$

The prediction $\hat{y}$ is essentially the product of all weight matrices:

$$
\hat{y} \approx \mathbf{W}^{[L]} \mathbf{W}^{[L-1]} \cdots \mathbf{W}^{[1]} \mathbf{x}
$$

#### Example with Diagonal Weight Matrices

Assume the weight matrices are identical and diagonal:

$$
\mathbf{W}^{[l]} = \begin{bmatrix} 1.5 & 0 \\ 0 & 1.5 \end{bmatrix} \quad \text{(or } \mathbf{W}^{[l]} = 1.5 \cdot \mathbf{I} \text{)}
$$

The network output becomes:

$$
\hat{y} \approx \left( \begin{bmatrix} 1.5 & 0 \\ 0 & 1.5 \end{bmatrix} \right)^{L} \mathbf{x}
$$

* **Exploding Gradients:** If $L$ is very large and the components of $\mathbf{W}^{[l]}$ are greater than 1 (e.g., 1.5), $\hat{y}$ will **explode exponentially** as a function of $L$.
* **Vanishing Gradients:** If the components of $\mathbf{W}^{[l]}$ are less than 1 (e.g., 0.5), $\hat{y}$ will **vanish exponentially** (tend toward zero) as a function of $L$.

### Effect on Gradient Descent

Since the gradient (the derivative of the cost function with respect to the early layer weights) involves similar products of weights, it will also **increase or decrease exponentially**.

* **Exploding:** Gradients become too large, leading to massive updates that cause the model to diverge (leave the minimum).
* **Vanishing:** Gradients become too small, leading to extremely slow learning. Gradient Descent takes an **enormous amount of time to converge**, especially for early layers.

## Solution

The primary solution to mitigate the vanishing/exploding gradient problem is careful **initialization of the neural network weights**.

### Solution: Partially Random Initialization

Instead of initializing weights to the same value or a large random value, we use initialization schemes that ensure the variance of the activations remains stable across all layers.

* This involves initializing the weights **randomly** but scaling the variance of the initial weights based on the number of inputs to the layer ($n^{[l-1]}$). Common methods include **He initialization** or **Xavier initialization**.

## Weight Initialization for Deep Networks

When initializing weights in a deep neural network, the goal is to keep the variance of the input to each neuron, $z$, from growing too large (exploding) or shrinking too small (vanishing) as the network gets deeper.

For a single neuron's linear input $z$:

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_{n} x_{n}
$$

If the number of input features $n$ (the number of terms being summed) is very large, the sum $z$ will tend to be very large. To counteract this, the individual weights $w_i$ must be kept small.

## General Principle

To maintain a consistent variance of $z$, the initialization variance of the weights, $\text{Var}(W)$, should be inversely proportional to the number of inputs $n^{[l-1]}$ leading into the layer $l$.

$$
\text{Var}(\mathbf{W}) \propto \frac{1}{n^{[l-1]}}
$$

The standard deviation for initializing the weights is the square root of this variance.

## Specific Initialization Schemes

The chosen constant depends on the activation function used in the network:

### 1. Simple Initialization (often used for Tanh/Sigmoid)

This scheme aims to keep the variance of the activations close to 1.

$$
\mathbf{W}^{[l]} = \text{np.random.randn}(\text{shape}) \times \sqrt{\frac{1}{n^{[l-1]}}}
$$

| Symbol | Description |
| :--- | :--- |
| $\mathbf{W}^{[l]}$ | The weight matrix for layer $l$. |
| $n^{[l-1]}$ | The number of input units to layer $l$ (i.e., the size of the previous layer $l-1$). |
| $\text{np.random.randn}(\text{shape})$ | Generates a matrix with samples from a standard normal distribution. |

### 2. He Initialization (Best for ReLU)

He initialization is optimized for the **Rectified Linear Unit (ReLU)** activation function. Because ReLU sets half of the activations to zero, a larger scaling factor (specifically $\sqrt{2}$) is needed to maintain the variance.

$$
\mathbf{W}^{[l]} = \text{np.random.randn}(\text{shape}) \times \sqrt{\frac{2}{n^{[l-1]}}}
$$

* **Variance:** $\text{Var}(\mathbf{W}) = \frac{2}{n^{[l-1]}}$

### 3. Xavier/Glorot Initialization (Often used for Tanh)

Xavier initialization is an alternative approach that attempts to maintain the variance both in the forward pass and the backward pass (gradients).

$$
\mathbf{W}^{[l]} = \text{np.random.randn}(\text{shape}) \times \sqrt{\frac{2}{n^{[l-1]} + n^{[l]}}}
$$

| Symbol | Description |
| :--- | :--- |
| $n^{[l-1]}$ | Number of input units to layer $l$. |
| $n^{[l]}$ | Number of output units from layer $l$. |

## Gradient Checking (Numerical Gradient Verification)

Gradient Checking is a test used to ensure that the mathematical implementation of the **backpropagation** algorithm (which computes the analytical gradients, $\partial J / \partial \theta$) is numerically correct.

## 1. The Core Concept: Numerical Approximation

The test compares the analytical gradient computed by backpropagation with a numerical approximation derived from the definition of the derivative.

The definition of the derivative $f'(\theta)$ is:

$$
f'(\theta) = \lim_{\epsilon \to 0} \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2 \epsilon}
$$

The numerical approximation used for gradient checking is:

$$
g(\theta)_{\text{approx}} = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2 \epsilon}
$$

* When the analytical gradient $g(\theta)$ computed by backpropagation is **approximately equal** to the numerical approximation $g(\theta)_{\text{approx}}$, the implementation is correct.

## 2. Gradient Checking Procedure

The procedure involves converting all parameters and their gradients into large vectors to check them simultaneously.

### Step A: Vectorization of Parameters

All parameters of the neural network—weights $\mathbf{W}^{[1]}, \dots, \mathbf{W}^{[L]}$ and biases $$\mathbf{b}^{[1]}, \dots, \mathbf{b}^{[L]}$$—are **reshaped** and concatenated into one large vector, $\boldsymbol{\theta}$.

$$
\boldsymbol{\theta} = \text{Vectorize}(\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \dots, \mathbf{W}^{[L]}, \mathbf{b}^{[L]})
$$

The cost function $J$ can then be viewed as a function of this single vector: $J(\boldsymbol{\theta})$.

### Step B: Vectorization of Gradients

Similarly, all analytical gradients computed by backpropagation— $$d\mathbf{W}^{[1]}, d\mathbf{b}^{[1]}, \dots$$ —are vectorized into one large gradient vector, $d\boldsymbol{\theta}$.

$$
d\boldsymbol{\theta} = \text{Vectorize}(d\mathbf{W}^{[1]}, d\mathbf{b}^{[1]}, \dots, d\mathbf{W}^{[L]}, d\mathbf{b}^{[L]})
$$

**Goal:** We are checking if $d\boldsymbol{\theta}$ is the true gradient of $J(\boldsymbol{\theta})$.

### Step C: Numerical Gradient Calculation

For each element $\theta_i$ in the vector $\boldsymbol{\theta}$, the numerical approximation of the partial derivative $\frac{\partial J}{\partial \theta_i}$ is calculated:

$$
d\boldsymbol{\theta}_{\text{approx}}[i] = \frac{J(\theta_1, \dots, \theta_i + \epsilon, \dots) - J(\theta_1, \dots, \theta_i - \epsilon, \dots)}{2 \epsilon}
$$

This results in the numerical gradient vector $d\boldsymbol{\theta}_{\text{approx}}$.

### Step D: Comparison

The final step is to measure the **Euclidean distance** (L2 norm) between the analytical gradient vector ($d\boldsymbol{\theta}$) and the numerical gradient vector ($d\boldsymbol{\theta}_{\text{approx}}$).

The recommended metric for comparison is the **relative difference**:

$$
\text{Difference} = \frac{\| d\boldsymbol{\theta}_{\text{approx}} - d\boldsymbol{\theta} \|_2}{\| d\boldsymbol{\theta}_{\text{approx}} \|_2 + \| d\boldsymbol{\theta} \|_2}
$$

#### Interpretation:

| Difference Value | Conclusion | Action |
| :--- | :--- | :--- |
| $\le 10^{-7}$ (or $10^{-8}$) | **Excellent.** Implementation is very likely correct. | Proceed with training. |
| $\approx 10^{-5}$ | **Good.** Acceptable for a complex model, but warrants re-checking. | Proceed, but monitor closely. |
| $\ge 10^{-3}$ | **Worry.** The implementation likely contains a **bug**. | Stop and debug backpropagation. |

## Gradient Checking Usage Guidelines

While Gradient Checking is invaluable for verifying the correctness of your backpropagation implementation, it is **too slow** to use during the actual training process.

## 1. Do NOT Use Gradient Checking in Training

Gradient Checking is only a **debugging tool** and must be turned off once verification is complete.

* **Reason:** Computing $d\boldsymbol{\theta}_{\text{approx}}[i]$ using the numerical approximation formula requires two evaluations of the full cost function $J$ for **every single parameter** $\theta_i$ in your entire network.
    * Since the total number of parameters ($||\boldsymbol{\theta}||$) can be millions or even billions in deep networks, this process is **extremely slow** and computationally prohibitive for iterative training.

* **Training Strategy:**
    1.  Use **Backpropagation** to compute the analytical gradient $d\boldsymbol{\theta}$.
    2.  Use these efficient backpropagation derivatives to update the parameters during training.

## 2. Gradient Checking Workflow

The proper workflow for integrating Gradient Checking is:

1.  **Develop:** Write your code for **Forward Propagation** and **Backpropagation**.
2.  **Verify:** Turn Gradient Checking **ON**. Compute both $d\boldsymbol{\theta}_{\text{approx}}$ and $d\boldsymbol{\theta}$, and compare them using the relative difference formula.
3.  **Debug:** If the difference is large ($\ge 10^{-3}$), your backpropagation implementation has a bug.
4.  **Train:** Once the difference is sufficiently small ($\le 10^{-7}$), turn Gradient Checking **OFF** and proceed with training the model using only the much faster Backpropagation algorithm.

## 3. Debugging a Failure

If the algorithm fails Gradient Checking (meaning $d\boldsymbol{\theta}_{\text{approx}}$ and $d\boldsymbol{\theta}$ do not match):

* **Locate the Bug:** Instead of checking the entire vector $d\boldsymbol{\theta}$ at once, look at the individual components to try to identify the bug.
    * Examine the individual derivative components like $d\mathbf{W}^{[l]}$ or $d\mathbf{b}^{[l]}$. Often, simple errors in the dimensions or the calculation of the bias gradients (e.g., $d\mathbf{b}^{[l]}$) are the culprits.

# 📉 Training vs. Validation Curve Alignment

In many deep learning frameworks (like Keras or PyTorch Lightning), training and validation errors are logged at different frequencies, leading to a "temporal lag" in visualizations.

### 1. The Core Problem: Snapshot vs. Average
* **Validation Error:** A **snapshot**. It is computed on the entire validation set only *after* the epoch is finished.
* **Training Error:** A **running mean**. It is the average of the error from every batch *during* the epoch.

### 2. The "Half-Epoch" Logic
Because the training error is a cumulative average, the value reported at the end of Epoch 1 includes the very first (high-error) batches and the very last (low-error) batches. 
* **Training Point:** Represents the model state at roughly **$t = 0.5$**.
* **Validation Point:** Represents the model state at exactly **$t = 1.0$**.



### 3. Why Shift the Curve?
If you shift the training curve **0.5 epochs to the left**, you align the "average" performance with the "snapshot" performance.

| Observation | Meaning |
| :--- | :--- |
| **Curves Overlap** | The model is generalizing perfectly; the gap was just a logging artifact. |
| **Training < Validation** | Even after the shift, the model is likely **overfitting**. |
| **Training > Validation** | Common in early training due to **Regularization** (like Dropout) being active during training but disabled during validation. |

### 🛠 Summary for Implementation
> "When plotting, subtract 0.5 from the X-axis of your training metrics to see the true relationship between learning and generalization."

# 🤖 Keras API Comparison: Sequential vs. Functional

In Keras, there are two primary ways to build models. Choosing the right one depends on the complexity of your architecture.

---

## 1. Sequential API (The "Stack")
The Sequential API is a linear stack of layers. It is the most common way to build models for beginners and simple tasks.

* **Logic:** One layer flows directly into the next.
* **Constraint:** Exactly **one input** and **one output**. No branching, no skipping.



### ✅ Pros
* Extremely simple and readable.
* Ideal for 90% of standard deep learning problems (Classifiers, simple CNNs).
* Minimal boilerplate code.

### ❌ Cons
* Cannot share layers.
* Cannot handle multiple inputs (e.g., Image + Metadata).
* Cannot handle multiple outputs (e.g., Object Detection: Class + Bounding Box).
* Cannot create **Residual (Skip) Connections**.

---

## 2. Functional API (The "Graph")
The Functional API treats layers like functions. You define a tensor, pass it through a layer, and receive a new tensor.

* **Logic:** A Directed Acyclic Graph (DAG).
* **Flexibility:** Layers can be connected in any way imaginable.



### ✅ Pros
* **Complex Topologies:** Supports branching, merging, and skip connections.
* **Multi-Input/Output:** Essential for advanced models (e.g., Siamese networks, Transformers).
* **Layer Reusability:** Use the same layer instance multiple times in the graph.

### ❌ Cons
* More "verbose" (requires explicit `Input` definition).
* Slightly steeper learning curve.

---

## 📊 Feature Comparison Table

| Feature | Sequential API | Functional API |
| :--- | :--- | :--- |
| **Structure** | Linear Stack | Directed Acyclic Graph (DAG) |
| **Ease of Use** | High (Beginner) | Medium (Intermediate) |
| **Skip Connections** | ❌ No | ✅ Yes |
| **Multiple Inputs** | ❌ No | ✅ Yes |
| **Multiple Outputs** | ❌ No | ✅ Yes |
| **Layer Sharing** | ❌ No | ✅ Yes |

---

## 💡 Quick Code Reference

### Sequential Example
```python
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(10, activation='softmax')
])
```

### Functional Example
```python
inputs = layers.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=inputs, outputs=outputs)
```

# 🏗️ Advanced Functional API Architectures

### 1. Wide & Deep Models
* **Wide Path:** Captures simple rules/memorization.
* **Deep Path:** Captures complex patterns/generalization.
* **Mechanism:** Concatenates the raw input with the output of the deep stack before the final layer.

### 2. Multi-Input Models
* Used when different paths require different feature subsets.
* **Key Code:** `model = Model(inputs=[in_1, in_2], outputs=[out])`.
* **Data Entry:** Feed data as a list `[X1, X2]` or a dict `{"name": X1}`.

### 3. Multi-Output Models
* **Object Detection:** Regression (coords) + Classification (label).
* **Multitask Learning:** Learning related tasks simultaneously to improve feature extraction.
* **Auxiliary Outputs:** Used as a regularization tool to ensure lower layers are learning effectively.

# 🛠️ Keras Subclassing API

The Subclassing API is the most flexible way to build models, favoring **imperative programming** over static blueprints. Sequential API and Functional API are declarative: we start by declaring which layers to use and how they should be connected. Then we feed the model some data for training or inference. So we can save, clone, share and analyze but it is static.

### 🏗️ Structure
1. **`__init__`**: Define all layers as attributes (`self.layer_name`).
2. **`call()`**: Define the forward pass logic. This is where you connect layers and add Pythonic logic (`if`, `for`).

### ✅ When to Use (Pros)
* **Research & Innovation:** For experimental architectures not supported by standard APIs.
* **Dynamic Logic:** When the model behavior needs to change based on the input data (loops, conditions).
* **Low-level Control:** When you need to use raw `tf.operations`.

### ❌ The Trade-offs (Cons)
* **Harder to Debug:** Errors are only caught at runtime.
* **Opaque Structure:** `model.summary()` cannot show layer connectivity.
* **Static Analysis:** Keras cannot inspect the graph for optimizations or easy cloning.

### 📝 Key Rule
> **"Start with Sequential, move to Functional if you need branches, move to Subclassing ONLY if you need dynamic Python logic."**
