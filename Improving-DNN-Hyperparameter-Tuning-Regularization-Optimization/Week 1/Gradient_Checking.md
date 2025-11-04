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
