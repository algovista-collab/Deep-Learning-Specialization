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

when we implement a backpropagation, there is a test called
