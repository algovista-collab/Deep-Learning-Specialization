# Deep Learning: Hyperparameter Tuning and Batch Normalization

## 1. Hyperparameter Priority & The Adam Optimizer
In practice, hyperparameters are not created equal. Some require significantly more attention during tuning than others.

* **Most Important:** Learning rate ($\alpha$).
* **Second Tier:** Momentum ($\beta$), number of hidden units, mini-batch size.
* **Third Tier:** Number of layers, learning rate decay.

### Adam Optimizer Default Settings
For the **Adam Optimizer**, the following values are widely accepted as defaults and are rarely tuned:
* $\beta_1 = 0.9$ (Momentum)
* $\beta_2 = 0.999$ (RMSProp)
* $\epsilon = 10^{-8}$ (Stability constant)

---

## 2. Tuning Strategies
### Random Sampling vs. Grid Search
Instead of using a fixed grid, **random sampling** is preferred. It ensures a broader exploration of the hyperparameter space, especially when some hyperparameters are more influential than others.



### Coarse-to-Fine Technique
1.  **Coarse Stage:** Sample random points across the entire 2D (or multi-dimensional) space.
2.  **Fine Stage:** Identify a high-performing cluster of points, zoom into that specific area (the "square"), and sample more densely to find the local optimum.

---

## 3. Sampling Scales
Different hyperparameters require different mathematical scales for effective exploration.

### Linear Scale
Used for parameters where the range is narrow and uniform, such as:
* **Hidden units** in a layer (e.g., 50 to 100).
* **Number of layers**.

### Logarithmic Scale
Used for parameters that span several orders of magnitude, like the **learning rate ($\alpha$)**. 
* Sampling uniformly on $[0.0001, 1]$ would waste 90% of resources between $0.1$ and $1.0$.
* **Method:** Sample $r$ uniformly in $[\log_{10}(\text{min}), \log_{10}(\text{max})]$, then set $\alpha = 10^r$.

### Sampling for $\beta$ (Exponentially Weighted Averages)
When tuning $\beta$ (e.g., $0.9$ to $0.999$), small changes near $1$ have a massive impact.
* It is better to sample $1 - \beta$ on a **logarithmic scale**.
* This explores the "weighting window" $1/(1-\beta)$ more efficiently.

---

## 4. Tuning Approaches
The choice of approach depends on your available computational power.

| Approach | Name | Description |
| :--- | :--- | :--- |
| **Panda** | Babysitting | Monitoring one model over days, tweaking parameters as it trains. Used when resources are low. |
| **Caviar** | Parallel | Training many models simultaneously with different settings and picking the best. Used when resources are high. |

---

## 5. Batch Normalization (BN)
Developed by **Sergey Ioffe** and **Christian Szegedy**, BN makes neural networks much more robust and faster to train.

### The Core Idea
Just as we normalize input features to speed up learning, BN normalizes the **pre-activation values ($z$)** of hidden layers.

1.  Calculate the **mean** ($\mu$) and **variance** ($\sigma^2$) of $z$ for a mini-batch.
2.  Normalize: $z_{norm}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$
3.  Scale and Shift: $\tilde{z}^{(i)} = \gamma z_{norm}^{(i)} + \beta$
    * $\gamma$ and $\beta$ are learnable parameters that allow the network to maintain its representational power.



### Internal Covariate Shift
This is the phenomenon where the distribution of a layer's inputs changes as the parameters of the previous layers change. 
* **Batch Norm's Solution:** It ensures that no matter how the previous layers change, the mean and variance of the current layer's inputs remain stable.
* **Result:** It allows each layer of the network to learn more independently of other layers.
