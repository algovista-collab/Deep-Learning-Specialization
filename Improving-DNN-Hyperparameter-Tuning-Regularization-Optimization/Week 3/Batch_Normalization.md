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

## Keras Tuner: Summary of Hyperparameter Search

For complex problems, deep networks have a much higher parameter efficiency than shallow ones: they can model complex functions using exponentially fewer neurons than shallow nets, allowing them to reach much better performance with the same amount of training data.Hyperparameter tuning automates the "trial and error" process of building neural networks.

### 1. The Search Strategies
| Strategy | How it works | Best Use Case |
| :--- | :--- | :--- |
| **Random Search** | Random guesses. | Simple, non-complex problems. |
| **Hyperband** | "Tournament" style; kills bad models early. | Fast exploration of many architectures. |
| **Bayesian Opt.** | Uses math to "predict" the best settings. | Finding the absolute peak performance. |

### 2. Key Terms
* **Oracle:** The algorithm that picks the next trial's values.
* **Trial:** A single "experiment" (one model architecture trained once).
* **Objective:** The metric the tuner is trying to maximize (e.g., `val_accuracy`).

### 3. Tuning Beyond Layers
To tune **Batch Size** or **Preprocessing**, you must subclass `kt.HyperModel` and override the `fit()` method.

### 4. Analysis
Always use **TensorBoard** with the **HPARAMS** tab. It allows you to see the "Parallel Coordinates" view, which reveals which hyperparameters are actually driving your model's success.

# 📏 Sizing Hidden Layers: Rules of Thumb

### 1. Structure
* **Input/Output:** Fixed by the dataset dimensions.
* **Hidden Layers:** Use a **constant width** for all hidden layers (easier to tune).
* **First Layer:** Occasionally, making the very first hidden layer larger than the rest helps capture initial low-level features.

### 2. The "Stretch Pants" Approach
* **Philosophy:** Over-build the capacity, then constrain it with regularization.
* **Avoid Bottlenecks:** Ensure no hidden layer is significantly smaller than the ones before it to prevent permanent data loss.

### 3. Depth over Width
* Increasing the **number of layers** (depth) is usually more effective than increasing the **number of neurons** (width).

# ⚙️ Key Hyperparameter Tuning Tips

### 1. Learning Rate (LR)
* **Impact:** High. Controls how large the "steps" are during Gradient Descent.
* **Finding the Best LR:** Use an **LR Finder** (increase LR exponentially and plot loss). Pick a value slightly before the loss starts to explode.

### 2. Batch Size
* **The GPU Factor:** Large batches utilize hardware better but may hurt generalization.
* **LeCun's Rule:** "Friends don't let friends use batches > 32."
* **Advanced Hack:** Use large batches + **Learning Rate Warmup** (starting small and ramping up) to get the best of both worlds.

### 3. General Heuristics
| Hyperparameter | Best Practice |
| :--- | :--- |
| **Optimizer** | Try **Adam** or **RMSProp** (covered in Ch 11) over basic SGD. |
| **Iterations** | Use **Early Stopping** instead of a fixed number. |
| **Activation** | **ReLU** for hidden layers; **Softmax/Sigmoid** for output. |

> **⚠️ Critical Note:** Hyperparameters are interdependent. If you change the **Batch Size**, you almost always need to retune the **Learning Rate**.

# 📈 Advanced Activation Functions Summary

When building deep networks, choosing the right activation function prevents **Vanishing Gradients** and **Dead Neurons**.

### 1. The ELU Family
| Function | Key Characteristic | Best Use Case |
| :--- | :--- | :--- |
| **ReLU** | Simple, fast, but can "die" if $z < 0$. | Baseline / Simple models. |
| **ELU** | Smooth, allows negative values. | Faster convergence than ReLU. |
| **SELU** | **Self-normalizing** (Mean 0, Std 1). | Very deep MLPs (Dense layers only). |

### 2. The Modern Standard: GELU (Gaussian Error Linear Unit)
* **Formula:** $GELU(z) = z \Phi(z)$ (where $\Phi$ is the Gaussian CDF).
* **The "Wiggle":** It is non-monotonic (it dips slightly below 0 before going up).
* **Why it wins:** It consistently outperforms ReLU/ELU in complex tasks like NLP (Transformers).
* **Trade-off:** More computationally expensive (use the sigmoid approximation $z\sigma(1.702z)$ for speed).

### 3. Implementation Checklist
| Feature | ELU | SELU | GELU |
| :--- | :--- | :--- | :--- |
| **Keras String** | `activation="elu"` | `activation="selu"` | `activation="gelu"` |
| **Initializer** | He Normal | **LeCun Normal** | He Normal |
| **Constraint** | None | Must be a plain MLP | None |

### 🚀 Pro-Tip for Generalization
> "In general: **GELU > ELU > ReLU**. If you are building a very deep MLP and don't want to use Batch Normalization, try **SELU** with **LeCun Normal** initialization."
