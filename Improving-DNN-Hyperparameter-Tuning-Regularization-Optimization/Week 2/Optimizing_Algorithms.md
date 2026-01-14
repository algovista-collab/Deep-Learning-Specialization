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
