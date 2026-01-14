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
* **Typical mini-batch sizes:** $64, 128, 256, 512$. 
* **Note:** It is common to use powers of $2$ to take advantage of how CPU/GPU memory is accessed.
* **Constraint:** Ensure that the mini-batch fits in your CPU/GPU memory (RAM).

---
