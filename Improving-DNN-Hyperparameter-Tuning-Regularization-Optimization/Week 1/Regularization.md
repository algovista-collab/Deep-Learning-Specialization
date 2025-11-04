# Evaluation and Regularization in Machine Learning

## 1. Data Distribution and Splitting

**Rule of Thumb:**
The **development (Dev) set** and **Test set** should ideally come from the **same distribution** as this ensures the performance metrics measured accurately reflect real-world performance.

* **Training Set:** May contain data from different sources or distributions, but should be large enough to train the model effectively.
* **Test Set:** If an **unbiased estimate** of the final model's performance is not strictly needed, it might be acceptable to **omit the test set**.

### Deep Learning and Bias/Variance Trade-off
Modern techniques, particularly in **Deep Learning**, make it popular in Supervised Learning because:
* **Bias and Variance can both be reduced** without severely hurting the other.
* As long as regularization is used, **using a bigger network never hurts performance** (it just increases computation time).

---

## 2. Regularization

Regularization is a technique used to prevent **overfitting** by adding a penalty term to the cost function, discouraging the weights ($w$) from taking on very large values.

### A. Regularization in Logistic Regression

The cost function $J$ for Logistic Regression with L2 Regularization is:

$$
J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \| \mathbf{w} \|_2^2
$$

* $\mathcal{L}(\hat{y}^{(i)}, y^{(i)})$ is the **Loss function** (e.g., Binary Cross-Entropy).
* $\lambda$ is the **regularization hyperparameter**.
* $m$ is the number of training examples.
* **Bias $b$** is typically **not** included in the regularization term.

#### L2 Norm (Euclidean Norm)
$$
\| \mathbf{w} \|_2^2 = \sum_{j=1}^{n_x} w_j^2 = \mathbf{w}^T \mathbf{w}
$$
* $\mathbf{w}$ is the weight vector, $\mathbf{w} \in \mathbb{R}^{n_x}$.
* **L2 norm is used more often** due to better convergence properties.

#### L1 Norm
The L1 regularization term added to the cost function is:

$$
\frac{\lambda}{2m} \| \mathbf{w} \|_1 = \frac{\lambda}{2m} \sum_{j=1}^{n_x} |w_j|
$$

* **Sparsity:** L1 regularization encourages the weights $\mathbf{w}$ to become **sparse** (i.e., many weights $w_j$ will become exactly **zero**).
* **Advantage:** Sparse models require **less memory** because sparse matrices store only the non-zero elements.

---

### B. Regularization in Neural Networks

The cost function $J$ for a Neural Network with $L$ layers and L2 Regularization:

$$
J(\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \dots, \mathbf{W}^{[L]}, \mathbf{b}^{[L]}) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum_{l=1}^{L} \| \mathbf{W}^{[l]} \|_F^2
$$

* $\mathbf{W}^{[l]}$ is the weight matrix for layer $l$.
* $\mathbf{b}^{[l]}$ (bias vectors) are typically **not** regularized.

#### Frobenius Norm
The term $\| \mathbf{W}^{[l]} \|_F^2$ is the **Frobenius Norm** of the weight matrix $\mathbf{W}^{[l]}$ squared.

$$
\| \mathbf{W}^{[l]} \|_F^2 = \sum_{i=1}^{n^{[l]}} \sum_{j=1}^{n^{[l-1]}} (W_{i, j}^{[l]})^2
$$
* This is essentially the **sum of the squares of all elements** in the matrix $\mathbf{W}^{[l]}$.

### 3. L2 Regularization as "Weight Decay"

The update step for the weights during Gradient Descent is modified by the regularization term:

1.  **Gradient Calculation (from Backpropagation):**

$$
d\mathbf{W}^{[l]} = \left( \frac{\partial J}{\partial \mathbf{W}^{[l]}} \right)_{\text{from backprop}} + \frac{\lambda}{m} \mathbf{W}^{[l]}
$$

2.  **Weight Update:**

$$
\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \cdot d\mathbf{W}^{[l]}
$$

Substituting the first equation into the second:

$$
\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \left( \left( \frac{\partial J}{\partial \mathbf{W}^{[l]}} \right)_{\text{from backprop}} + \frac{\lambda}{m} \mathbf{W}^{[l]} \right)
$$

Rearranging the terms:

$$
\mathbf{W}^{[l]} := \underbrace{\left( 1 - \alpha \frac{\lambda}{m} \right) \mathbf{W}^{[l]}}_{\text{Decay Term}} - \alpha \left( \frac{\partial J}{\partial \mathbf{W}^{[l]}} \right)_{\text{from backprop}}
$$

Because the term $\left( 1 - \alpha \frac{\lambda}{m} \right)$ is slightly less than 1, L2 regularization effectively multiplies the weight vector $\mathbf{W}^{[l]}$ by a factor that slightly shrinks its magnitude in every iteration, hence the name **Weight Decay**.

## How Regularization Helps Prevent Overfitting

Regularization, particularly **L2 Regularization** (or Weight Decay), prevents overfitting by constraining the magnitude of the model's weight parameters ($\mathbf{w}$). This limitation simplifies the model and makes its output function smoother.

## Mechanism of Overfitting Prevention

### 1. Reducing the Magnitude of Weights ($\mathbf{w}$)
* **Cost Function:** The regularization term, $\frac{\lambda}{2m} \| \mathbf{w} \|_2^2$, is added to the loss function $J$.
* **Minimization:** To minimize the overall cost $J$, the optimization algorithm is forced to choose smaller values for the weights $\mathbf{w}$ as $\lambda$ (the regularization hyperparameter) increases.

$$
\text{If } \lambda \uparrow \implies \|\mathbf{w}\|_2 \downarrow
$$

### 2. Damping the Input to Activation Functions ($z$)
* **Linear Combination:** The input $z$ to any activation function in a neural network is a linear combination of the weights and the previous layer's output (or input features):

$$
z = \mathbf{w}^T \mathbf{x} + b
$$

* **Smaller Weights, Smaller $z$:** Since the weights $\mathbf{w}$ are significantly reduced by the regularization, the absolute value of $z$ (the input to the activation function) will also be **reduced** and take on **smaller values**.

### 3. Promoting Near-Linearity in Non-Linear Activation Functions

* **The Tanh Example:** Consider the $\text{tanh}(z)$ activation function. The $\text{tanh}$ function is non-linear over a wide range of $z$ values, allowing the network to learn complex curves. However, when $z$ is close to 0, the function is **approximately linear**.

$$
\text{As } z \to 0, \quad \text{tanh}(z) \approx z
$$

* **Simplification:** By forcing $z$ to be small, regularization pushes the activation function into its nearly linear region. When every layer's activation function operates closer to linearity, the overall network's output function becomes **less non-linear** and much **smoother**.
* **Preventing Overfitting:** An overly complex and highly non-linear decision boundary is often what allows a model to perfectly fit noise in the training data (overfitting). By simplifying the function, regularization prevents the model from fitting these minor fluctuations, leading to better **generalization** to unseen data.

<img width="437" height="196" alt="image" src="https://github.com/user-attachments/assets/61d87e70-1583-4498-9ec4-dbc0d304f79d" />

## Dropout Regularization

Dropout is a powerful regularization technique primarily used in deep neural networks to prevent **overfitting**. It randomly "drops out" (eliminates) nodes and their connections during the training phase.

## Core Mechanism

During each training iteration, every node in a layer is temporarily removed with a specific probability. This forces the network to become more robust, as no single node can rely too heavily on the output of any other specific node.

* **Probability:** A **keep probability** ($keep\_prob$) is defined, representing the chance that a given unit is kept (i.e., *not* eliminated).
    * Example: If $keep\_prob = 0.5$, there is a 50% chance of keeping or removing a node.

---

## Implementing Dropout: Inverted Dropout

Inverted Dropout is the preferred implementation because it performs the necessary scaling during **training time**, simplifying the prediction process at test time.

### Illustration (Layer $l=3$, $keep\_prob=0.8$)

| Step | Operation | Formula | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Generate Mask** | Create a dropout mask $d^{[3]}$ with the same dimensions as the activation vector $a^{[3]}$. | `d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob` | Generates a boolean mask (1s and 0s) where the probability of a 1 (keeping the unit) is $keep\_prob$. |
| **2. Apply Mask** | Apply the mask element-wise to the layer's activation vector $a^{[3]}$. | `a3 = np.multiply(old_a3, d3)` or `a3 *= d3` | **Eliminates** the nodes corresponding to the zeros in $d^{[3]}$ (e.g., $1 - 0.8 = 20\%$ of units are shut off). |
| **3. Scaling (Inverted Dropout)** | Scale up the remaining active units by dividing by $keep\_prob$. | `a3 /= keep_prob` | Ensures that the expected value of the activation remains the same, compensating for the units that were shut off. |

### Scaling Rationale

If a layer has 50 units and $keep\_prob = 0.8$:
* Approximately **10 units** will be shut off.
* The total sum of activations entering the next layer ($z^{[4]} = \mathbf{W}^{[4]} \mathbf{a}^{[3]} + \mathbf{b}^{[4]}$) will be reduced by $\approx 20\%$.
* By dividing by $keep\_prob$ (e.g., $0.8$), the magnitude of the activations $\mathbf{a}^{[3]}$ is **scaled up**, maintaining the expected input magnitude to the next layer $\mathbf{z}^{[4]}$.

---

## Making Predictions at Test Time

### 1. No Dropout
During the test phase, **no dropout is applied**. All nodes are kept active ($keep\_prob = 1.0$).

* **Rationale:** We do not want random behavior or reduced performance during final prediction. We want the most stable and reliable output from the entire learned network.

### 2. Eliminating the Mismatch

The purpose of **Inverted Dropout scaling** during training is to eliminate the performance mismatch that would occur if the scaling was done during testing.

| Scenario | Average Output Calculation (Example: $h=[2, 4, 6, 8]$, $keep\_prob=0.5$) |
| :--- | :--- |
| **Training (Before Scaling)** | $\mathbf{h}' = \mathbf{h} \times \mathbf{m} = [2, 0, 6, 0]$. Average: $(2+0+6+0)/4 = \mathbf{2}$. |
| **Testing (All Active)** | Average: $(2+4+6+8)/4 = \mathbf{5}$. |

Without scaling, the test-time output (5) is much higher than the training-time average (2). This mismatch causes a performance drop.

**With Inverted Dropout Scaling (at Training Time):**
$$
\mathbf{h}_{\text{scaled}} = \mathbf{h}' \times \frac{1}{keep\_prob} = [2, 0, 6, 0] \times \frac{1}{0.5} = [4, 0, 12, 0]
$$
The average magnitude now is $(4+0+12+0)/4 = \mathbf{4}$. This value is closer to the test-time average (5) and accounts for the expected reduction in magnitude.

---

## Other Considerations

* **Different $keep\_prob$ by Layer:** It is common practice to use different values of $keep\_prob$ for different layers. For example, layers with many parameters (often earlier or hidden layers) might use a lower $keep\_prob$ (e.g., 0.5), while input layers might use a higher $keep\_prob$ (e.g., 0.8-0.9), or not use dropout at all.
* **Downside:** Since the network structure changes randomly in every iteration, the cost function $J$ is **not perfectly defined** or easily computable (it's hard to track the true function being optimized). However, this empirical effectiveness outweighs the theoretical downside.
