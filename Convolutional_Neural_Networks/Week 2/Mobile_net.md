# MobileNet Evolution: v1 to v2

MobileNets are designed to bring high-performance computer vision to resource-constrained devices (phones, IoT) by replacing standard, expensive convolutions with lighter alternatives.

---

## 1. MobileNet v1: The Depthwise Separable Foundation
The core of v1 is the **Depthwise Separable Convolution**, which splits a standard convolution into two distinct steps to save 80-90% of the math.

* **Architecture:** A simple stack of 13 depthwise separable blocks.
* **Block Structure:** 1.  **Depthwise Conv:** Filters spatial info (one filter per channel).
    2.  **Pointwise Conv ($1\times1$):** Mixes channels together.
* **Ending:** Global Average Pooling $\rightarrow$ Fully Connected $\rightarrow$ Softmax.



---

## 2. MobileNet v2: The Inverted Residual & Linear Bottleneck
MobileNet v2 improves accuracy and memory efficiency by introducing the **Inverted Residual Block** (or Bottleneck Block). It uses 17 of these blocks.

### The Three Layers of a v2 Block:
1.  **Expansion Layer ($1\times1$ Conv):** * Expands the number of channels (usually by a factor of 6). 
    * **Purpose:** Allows the network to learn more complex features in a higher-dimensional space before filtering.
2.  **Depthwise Convolution:** * Performs the spatial filtering on the expanded data.
3.  **Projection Layer ($1\times1$ Conv):** * Projects the data back down to a small number of channels. 
    * **Purpose:** Shrinks the data to fit back into memory-efficient "bottlenecks."



### Key Differences in v2:
* **Residual Connections:** Adds "Shortcuts" between the thin bottleneck layers (similar to ResNet) to help gradients flow better.
* **Linear Bottlenecks:** The last $1\times1$ (Projection) layer uses **no activation function** (Linear). This prevents the ReLU function from destroying useful information when the data is compressed.

---

## 3. Comparison Table: v1 vs. v2

| Feature | MobileNet v1 | MobileNet v2 |
| :--- | :--- | :--- |
| **Main Building Block** | Depthwise Separable Conv | **Inverted Residual Block** |
| **Layers per Block** | 2 (Depthwise + Pointwise) | 3 (Expansion + Depthwise + Projection) |
| **Residual Connections**| No | **Yes** (Skip connections) |
| **Complexity** | 13 Layers | 17 Layers |
| **Efficiency Trick** | Simple splitting of math | **Expansion** for learning + **Linearity** for info preservation |

---

## 4. Why it works: The "Expand-Filter-Compress" Cycle
MobileNet v2 is like a lung:
* **Inhale (Expansion):** Take a small amount of data and "expand" it to see it clearly.
* **Process (Depthwise):** Filter the details.
* **Exhale (Projection):** "Squeeze" the most important info back into a tiny package for the next step.

This cycle allows the network to be smarter (higher accuracy) while staying extremely small (lower memory).
