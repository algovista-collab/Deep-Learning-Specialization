# Theory: Mini-Batch Gradient Descent

In large-scale machine learning, the size of the training set $m$ can reach millions. Standard **Batch Gradient Descent** becomes a bottleneck because it requires a full pass through the entire dataset to take just one single step toward the minimum.

## 1. Batch vs. Mini-Batch vs. Stochastic

| Type | Batch Size | Characteristics |
| :--- | :--- | :--- |
| **Batch GD** | $m$ | Smooth convergence, but very slow per iteration. |
| **Stochastic GD** | $1$ | Very fast per iteration, but extremely noisy and loses vectorization speed. |
| **Mini-Batch GD** | $1 < \text{size} < m$ | The "Sweet Spot." Faster than Batch GD and more stable than Stochastic GD. |



---

## 2. Mathematical Notation

To distinguish between individual examples and batches, we use the following notation:
* $x^{(i)}$: The $i^{th}$ training example.
* $z^{[l]}$: The activation of the $l^{th}$ layer.
* $X^{\{t\}}, Y^{\{t\}}$: The $t^{th}$ **mini-batch**.

If $m = 5,000,000$ and we choose a mini-batch size of $1,000$, we have $T = 5,000$ mini-batches.

---

## 3. The Algorithm Workflow

For each **Epoch** (one full pass through the data):

### Step A: Forward Propagation
We process the mini-batch $X^{\{t\}}$ through the network. This remains vectorized. For a single layer $l$:
$$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

### Step B: Compute Cost ($J$)
The cost is calculated only for the $1,000$ examples in the current mini-batch, often including an $L_2$ regularization term:
$$J^{\{t\}} = \frac{1}{1000} \sum_{i=1}^{1000} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2 \cdot 1000} \sum_{l} \|W^{[l]}\|_F^2$$

### Step C: Backward Propagation & Update
We compute the gradients $\text{d}W$ and $\text{d}b$ based only on $J^{\{t\}}$ and update the parameters:
$$W^{[l]} := W^{[l]} - \alpha \text{d}W^{[l]}$$
$$b^{[l]} := b^{[l]} - \alpha \text{d}b^{[l]}$$

---

## 4. Choosing Mini-Batch Size

* **If training set is small ($m \le 2000$):** Use Batch Gradient Descent.
* **Typical mini-batch sizes:** $64, 128, 256, 512$. If the dataset is small, choose Batch Gradient Descent.
* **Note:** It is common to use powers of $2$ to take advantage of how CPU/GPU memory is accessed.
* **Constraint:** Ensure that the mini-batch fits in your CPU/GPU memory (RAM).

---

<img width="1889" height="962" alt="Screenshot 2026-01-11 113750" src="https://github.com/user-attachments/assets/515a3af4-8a4b-45d9-a5de-6b53405d09a2" />

---

# Exponentially Weighted Averages (EWA)

Exponentially Weighted Averages (also known as Exponentially Weighted Moving Averages) are used to smooth out noise in data and are fundamental to faster optimization algorithms than standard Gradient Descent.

## 1. The Core Equation

The value of the average at time $t$ is calculated as:

$$V_t = \beta V_{t-1} + (1 - \beta)\theta_t$$

### Variable Definitions:
* $V_t$: The current exponentially weighted average.
* $V_{t-1}$: The previous weighted average.
* $\theta_t$: The actual value (data point) at time $t$.
* $\beta$: The **weighting factor** ($0 < \beta < 1$).

### Understanding $\beta$ (The Smoothing Parameter)
The value of $\beta$ determines how many previous days (or iterations) are being averaged:
* An average of approximately $\frac{1}{1-\beta}$ days.
* **High $\beta$ (e.g., 0.98):** Heavy smoothing. The curve is very "stiff" and lags behind the data but filters out most noise. (Averages ~50 days).
* **Low $\beta$ (e.g., 0.5):** Very noisy. The curve follows the data closely. (Averages ~2 days).



---

## 2. Why use EWA in Optimization?

In Gradient Descent, the "noise" (oscillations) can slow down progress toward the minimum. By using EWA of the gradients:
1. **Vertical oscillations** are averaged out (summing toward zero).
2. **Horizontal progress** is maintained (summing toward the minimum).
3. This allows for a **higher learning rate**, leading to faster convergence.

---

## 3. Bias Correction

In the early stages of iteration, $V_t$ is highly inaccurate because $V_0$ is usually initialized to $0$. This causes the initial values of the average to be much lower than the actual values.

To fix this, we use **Bias Correction**:

$$\hat{V}_t = \frac{V_t}{1 - \beta^t}$$

### Why it works:
* **When $t$ is small:** $\beta^t$ is significant, making the denominator smaller than 1. This "boosts" $V_t$ to its correct value.
* **When $t$ is large:** $\beta^t$ approaches 0, so the denominator approaches 1. The bias correction becomes negligible as the average "warms up."



---

## 4. Implementation Logic (Pseudocode)

Unlike a standard moving average, EWA is extremely memory efficient because you only need to store **one** value in memory ($V_{prev}$):

```python
v = 0 # Initialize
beta = 0.9
for t in range(1, num_iterations):
    theta_t = get_current_gradient()
    
    # Update EWA
    v = (beta * v) + (1 - beta) * theta_t
    
    # Apply Bias Correction
    v_corrected = v / (1 - beta**t)
    
    # Update parameters using v_corrected...
```
---

# Gradient Descent with Momentum

**Momentum** is an optimization technique that improves the speed and efficiency of the standard gradient descent algorithm. By adding a "memory" of past gradients, it helps the optimizer take larger, more confident steps in the right direction.

## 1. The Intuition
In standard Gradient Descent, the steps might oscillate wildly in the vertical direction (high variance) while making slow progress in the horizontal direction. 

Momentum uses **Exponentially Weighted Averages** to:
* **Dampen Oscillations:** Average out the vertical steps toward zero.
* **Accelerate Learning:** Accumulate velocity in the horizontal direction toward the minimum.



---

## 2. The Algorithm

On each iteration $t$, given a mini-batch:

### Step 1: Compute Gradients
Compute the standard gradients $dW$ and $db$ on the current mini-batch:
$$dW = \frac{\partial J}{\partial W}, \quad db = \frac{\partial J}{\partial b}$$

### Step 2: Update Velocity (EWA)
Update the moving average of the gradients. This "velocity" term ($V$) provides the momentum:
$$V_{dW} = \beta V_{dW} + (1 - \beta) dW$$
$$V_{db} = \beta V_{db} + (1 - \beta) db$$

### Step 3: Update Parameters
Instead of using the raw gradients ($dW, db$), we use the velocity terms to update the weights and biases:
$$W = W - \alpha V_{dW}$$
$$b = b - \alpha V_{db}$$

---

## 3. Hyperparameters

* **$\alpha$ (Learning Rate):** The size of the step.
* **$\beta$ (Momentum Coefficient):** * Typically set to **0.9**. 
    * This effectively averages the last $\frac{1}{1-0.9} = 10$ gradients.
    * $\beta = 0$ reduces the algorithm back to standard Gradient Descent.

---

## 4. Why it works: The "Bowl" Analogy
Imagine a ball rolling down a bowl. 
* The gradient ($dW$) provides **acceleration** (the slope of the bowl).
* The velocity ($V_{dW}$) represents the **momentum** the ball has built up.
* $\beta$ acts like **friction**, preventing the ball from oscillating infinitely and helping it settle at the bottom.



---

## 5. Implementation (Python)

```python
# Initialization
v_dw = 0
v_db = 0
beta = 0.9

# Optimization Loop
for t in range(num_iterations):
    # 1. Compute gradients via backprop
    dw, db = backward_propagation(X_batch, Y_batch, parameters)
    
    # 2. Update velocities
    v_dw = (beta * v_dw) + (1 - beta) * dw
    v_db = (beta * v_db) + (1 - beta) * db
    
    # 3. Update parameters
    W = W - alpha * v_dw
    b = b - alpha * v_db
```

---

# RMSProp and Learning Rate Decay

## 1. RMSProp (Root Mean Square Propagation)

RMSProp is an optimization technique that speeds up gradient descent by adjusting the learning rate for each parameter individually. It is particularly effective at dampening oscillations in the vertical direction while maintaining speed in the horizontal direction.

### The Algorithm
On iteration $t$, compute $dW$ and $db$ on the current mini-batch:

1. **Update Squared Gradient Averages:**
   We keep an exponentially weighted average of the *squares* of the gradients.
   $$S_{dW} = \beta_2 S_{dW} + (1 - \beta_2) (dW)^2$$
   $$S_{db} = \beta_2 S_{db} + (1 - \beta_2) (db)^2$$

2. **Update Parameters:**
   We divide the gradient by the square root of the average to "normalize" the update.
   $$W := W - \alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon}$$
   $$b := b - \alpha \frac{db}{\sqrt{S_{db}} + \epsilon}$$

> **Note:** $\epsilon$ (typically $10^{-8}$) is added to the denominator to prevent division by zero.



### Why it works:
* If $dW$ is very large (steep oscillations), $S_{dW}$ becomes large. Dividing by $\sqrt{S_{dW}}$ **slows down** the update in that direction.
* If $dW$ is small, $S_{dW}$ is small, and the update remains **faster**.
* This allows the algorithm to "dampen" the oscillations that usually prevent us from using a high learning rate.

---

## 2. Learning Rate Decay

As an optimization algorithm nears the minimum of the cost function, a fixed learning rate $\alpha$ might cause the parameters to "overshoot" or wander around the minimum rather than converging exactly. 

**Learning Rate Decay** slowly reduces $\alpha$ over time to allow for finer steps as training progresses.

### Mathematical Formula:
$$\alpha = \frac{\alpha_0}{1 + \text{decay\_rate} \times \text{epoch\_num}}$$

* **$\alpha_0$:** Initial learning rate.
* **$\text{decay\_rate}$:** A hyperparameter controlling how fast the rate drops.
* **$\text{epoch\_num}$:** The current full pass through the dataset.



### Other Common Decay Methods:
* **Exponential Decay:** $\alpha = 0.95^{\text{epoch\_num}} \cdot \alpha_0$
* **Discrete Staircase:** Reducing $\alpha$ by half every $k$ epochs.
* **Manual Decay:** Manually lowering the rate when the user notices the cost has plateaued.

---

## 3. Python Implementation

```python
# Initialization
s_dw, s_db = 0, 0
beta2 = 0.999 # Typical value for RMSProp
epsilon = 1e-8

for epoch in range(num_epochs):
    # Update alpha using decay
    alpha = initial_alpha / (1 + decay_rate * epoch)
    
    for t in range(num_mini_batches):
        dw, db = backward_propagation(X_batch, Y_batch)
        
        # Update Squared Gradients
        s_dw = (beta2 * s_dw) + (1 - beta2) * (dw**2)
        s_db = (beta2 * s_db) + (1 - beta2) * (db**2)
        
        # Update Parameters
        W = W - alpha * (dw / (np.sqrt(s_dw) + epsilon))
        b = b - alpha * (db / (np.sqrt(s_db) + epsilon))
```
---

# Challenges in Optimization: Saddle Points and Dynamics

In high-dimensional optimization (like training deep neural networks), the geometry of the cost function creates unique challenges that differ significantly from simple 2D or 3D visualizations.

## 1. Saddle Points

A **Saddle Point** is a point on the surface of a function where the gradient is zero ($\nabla J = 0$), but it is neither a local maximum nor a local minimum.

### Key Characteristics:
* **Curvature:** The function curves **upwards** in some directions and **downwards** in others.
* **Dimensionality:** In very high-dimensional spaces (e.g., a million parameters), it is statistically much more likely that some directions curve up and others curve down, rather than all directions curving up (a local minimum).
* **The Plateau Problem:** Near a saddle point, the gradient is very small, which can lead to a "plateau" where learning becomes painfully slow.



---

## 2. The Role of Beta ($\beta$) in Smoothing

When using optimization techniques like **Momentum** or **RMSProp**, the hyperparameter $\beta$ acts as the "memory" of the algorithm.

### Dynamics of Large $\beta$:
When $\beta$ is large (e.g., 0.9 or 0.99):
1. **Memory:** The algorithm remembers more of its previous direction and velocity.
2. **Cancellation of Noise:** Because vertical gradients (oscillations) frequently change signs (positive to negative and back), they **cancel each other out** in the moving average.
3. **Directional Stability:** While the noise cancels out, the consistent "downhill" direction (the horizontal trend) accumulates, allowing the algorithm to blast through plateaus and ignore irrelevant oscillations.



---

## 3. Summary of Geometric Intuition

| Feature | Standard Gradient Descent | Momentum / RMSProp (High $\beta$) |
| :--- | :--- | :--- |
| **Saddle Points** | Can get stuck in the plateau for a long time. | Built-up velocity helps "roll" through the plateau. |
| **Oscillations** | Steps wildly back and forth across a narrow valley. | Vertical steps cancel out; horizontal steps accelerate. |
| **Path to Minima** | Jagged and inefficient. | Smoother and more direct. |

---

## Visualizing the Plateau Escape

In a high-dimensional space, the "plateau" around a saddle point is the biggest enemy. Algorithms with "memory" (weighted averages) are essential because they provide the kinetic energy needed to exit these flat regions where the raw gradient is nearly zero.

# Hyperparameter Tuning in Keras with Keras Tuner — Detailed Summary

## Why Hyperparameter Tuning Matters
Neural networks are extremely flexible, which is both a strength and a weakness. Even for a simple Multilayer Perceptron (MLP), there are many hyperparameters to choose from, including:

- Network architecture (number of layers, neurons per layer)
- Activation functions
- Weight initialization
- Optimizer type
- Learning rate
- Batch size
- Number of training epochs
- Data preprocessing choices

Finding the best combination manually is impractical, so automated hyperparameter tuning is essential.

---

## Traditional Approach: Scikit-Learn Wrappers
One option is to wrap Keras models as Scikit-Learn estimators using **SciKeras**:

- `KerasRegressor`
- `KerasClassifier`

These can then be tuned using:
- `GridSearchCV`
- `RandomizedSearchCV`

While effective, this approach is limited and less integrated with modern deep learning workflows.

---

## Preferred Approach: Keras Tuner
**Keras Tuner** is a dedicated hyperparameter tuning library for Keras that offers:

- Multiple tuning strategies
- High flexibility
- Native Keras integration
- Excellent TensorBoard support
