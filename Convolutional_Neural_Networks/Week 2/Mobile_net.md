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

<img width="1801" height="897" alt="image" src="https://github.com/user-attachments/assets/5ae873e5-65dd-45b5-b2d6-034c1a3ea658" />

<img width="992" height="482" alt="image" src="https://github.com/user-attachments/assets/3c12baac-fd3c-4fea-a042-55475e4be4cc" />

# EfficientNet: Smart Scaling for Any Device

EfficientNet moves away from the traditional "trial and error" method of scaling networks. Instead of just adding more layers (Depth) or more channels (Width), it uses a principled approach called **Compound Scaling**.

---

## 1. The Three Dimensions of Scaling
When you want to make a network more accurate, you generally have three "knobs" to turn:

1.  **Depth ($d$):** Adding more layers. Deeper networks capture more complex features but are harder to train (vanishing gradients).
2.  **Width ($w$):** Adding more channels (filters) per layer. Wider networks capture more fine-grained features but saturate quickly in accuracy.
3.  **Resolution ($r$):** Using higher-resolution input images. Larger images provide more detail but significantly increase the math (FLOPs) required.



---

## 2. The Innovation: Compound Scaling
The authors (Tan and Le) discovered that these three dimensions are **interdependent**. If you increase the image resolution, the network needs more layers to increase its "receptive field" and more channels to capture the extra fine-grained patterns.

**The Compound Scaling Rule:**
Instead of scaling $d, w,$ or $r$ individually, EfficientNet scales all three simultaneously using a single **compound coefficient ($\phi$)**:
* $\text{Depth: } d = \alpha^\phi$
* $\text{Width: } w = \beta^\phi$
* $\text{Resolution: } r = \gamma^\phi$

By doing a small initial search to find the best ratios ($\alpha, \beta, \gamma$), you can then scale the entire network up to any size simply by changing $\phi$.



---

## 3. The Baseline: EfficientNet-B0
To make the scaling effective, you need a high-quality starting point. The researchers used **Neural Architecture Search (NAS)** to design a mobile-size baseline called **EfficientNet-B0**.

* **Core Block:** It uses the **MBConv** (Mobile Inverted Bottleneck) from MobileNet v2.
* **Extra Feature:** It includes **Squeeze-and-Excitation (SE)** blocks, which act as a mini-attention mechanism to help the network focus on important features.

---

## 4. Why Use EfficientNet?

| Model | Efficiency | Best Use Case |
| :--- | :--- | :--- |
| **EfficientNet-B0** | Ultra-Lightweight | Low-power IoT, basic mobile apps. |
| **EfficientNet-B3/B4** | Balanced | Modern smartphones, real-time web apps. |
| **EfficientNet-B7** | High Accuracy | Cloud-based processing, medical imaging. |

### Key Takeaway:
EfficientNet allows you to be **computationally responsible**. You can achieve state-of-the-art accuracy with a model that is often **8x smaller** and **6x faster** than traditional models (like ResNet or Inception) because it scales its dimensions in perfect balance.

<img width="940" height="478" alt="image" src="https://github.com/user-attachments/assets/3f1f93bb-cb3a-4e37-b9a9-3104acf5cc07" />

# Open-Source Implementations: The Industry "Cheat Code"

Andrew Ng explains that even for PhDs, replicating a complex paper from scratch is a nightmare. The secret to success in Deep Learning is standing on the shoulders of giants by using open-source code.

---

## 1. Why Re-implementing is a Trap
Research papers are often missing the "secret sauce"—small details that make the model actually work.
* **Hyperparameter Finickiness:** Details like learning rate decay and specific data augmentation are often omitted in papers.
* **The "PhD Test":** Even top researchers struggle to get the same results as a paper just by reading the text.
* **The Solution:** Use the **author’s actual code** from GitHub to guarantee the results match the paper.

---

## 2. The GitHub Workflow
If you want to use a model like ResNet, don't write it. **Clone it.**

### Step-by-Step:
1.  **Search:** Look for the architecture name on GitHub.
2.  **Verify:** Find a repo with a permissive license (like **MIT** or **Apache**).
3.  **Command:** Use the terminal to download it:
    ```bash
    git clone [https://github.com/author/repository-name.git](https://github.com/author/repository-name.git)
    ```



---

## 3. Transfer Learning: The Real Power
The biggest advantage of using open-source code isn't just the code itself—it's the **Pre-trained Weights**.

* **Weeks vs. Minutes:** Someone else spent thousands of dollars and weeks of GPU time training the model on the massive ImageNet dataset.
* **The Download:** When you download their repo, you are often downloading a "trained brain."
* **Your Job:** You only have to do "Transfer Learning," which means taking that pre-trained brain and slightly adjusting it for your specific task.

---

## Comparison: Scratch vs. Open Source

| Feature | Re-implementing from Scratch | Using Open-Source (GitHub) |
| :--- | :--- | :--- |
| **Effort** | Extremely High (Manual coding) | **Low (Clone & Run)** |
| **Reliability** | High risk of bugs | **High (Verified by community)** |
| **Training Time** | Weeks (Training from zero) | **Minutes (Using pre-trained weights)** |
| **Hardware Cost** | High (Requires multiple GPUs) | **Low (Can run on basic hardware)** |

---

### Key Takeaway
If you are starting a new Computer Vision project: **Search GitHub first.** Picking an architecture and downloading a pre-trained version is the fastest way to get a professional-grade AI running.
