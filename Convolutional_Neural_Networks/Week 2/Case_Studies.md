# Classic Neural Network Architectures Summary

This summary covers the evolution of Convolutional Neural Networks (CNNs) from the pioneering LeNet-5 to the massive VGG-16.

---

## 1. LeNet-5 (1998)
Designed by Yann LeCun for handwritten digit recognition (MNIST), it established the core "Conv-Pool-Conv-Pool" pattern used today.



* **Input:** $32 \times 32 \times 1$ (Grayscale).
* **Structure:** Two sets of Convolutional and Average Pooling layers, followed by three Fully Connected (FC) layers.
* **Key Traits:**
    * Used **Average Pooling** (modern nets prefer Max Pooling).
    * Used **Sigmoid/Tanh** nonlinearities (predates ReLU).
    * **Shrinking Dimensions:** As you go deeper, height/width decrease while channels increase ($1 \to 6 \to 16$).
* **Parameters:** ~60,000.

---

## 2. AlexNet (2012)
Created by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, this architecture launched the modern era of Deep Learning by winning the ImageNet challenge.



* **Input:** $227 \times 227 \times 3$ (RGB).
* **Innovations:**
    * **ReLU Activation:** First major use of Rectified Linear Units to speed up training.
    * **Max Pooling:** Used $3 \times 3$ filters with a stride of 2.
    * **Dropout:** Introduced to prevent overfitting in large networks.
* **Legacy:** Convinced the computer vision community that Deep Learning was superior to hand-crafted features.
* **Parameters:** ~60 Million.

---

## 3. VGG-16 (2014)
The VGG network simplified CNN design by using a very uniform and systematic architecture.


* **Core Philosophy:** * **Uniform Convolutions:** Always $3 \times 3$ filters, Stride 1, "Same" padding.
    * **Uniform Pooling:** Always $2 \times 2$ Max Pooling, Stride 2.
* **Structure:**
    * The network is composed of "blocks" where the number of filters doubles after every pooling layer (e.g., $64 \to 128 \to 256 \to 512$).
    * The "16" refers to the 16 layers with weights.
* **Parameters:** ~138 Million (Large and computationally expensive, but highly effective).

---

## Comparison of Architectures

| Feature | LeNet-5 | AlexNet | VGG-16 |
| :--- | :--- | :--- | :--- |
| **Year** | 1998 | 2012 | 2014 |
| **Parameters** | ~60,000 | ~60 Million | ~138 Million |
| **Activation** | Tanh / Sigmoid | ReLU | ReLU |
| **Pooling** | Average | Max | Max |
| **Complexity** | Low | Medium | High (Due to depth) |

## Common Trends Identified
1.  **Spatial Shrinkage:** Height ($n_H$) and Width ($n_W$) decrease as you go deeper.
2.  **Channel Growth:** The number of channels ($n_C$) increases as you go deeper.
3.  **Standard Pattern:** $\text{Conv} \to \text{Pool} \to \dots \to \text{FC} \to \text{Softmax}$.

<img width="1046" height="511" alt="image" src="https://github.com/user-attachments/assets/49d612c7-da57-48ad-bbc7-3819f530d1d6" />

<img width="1768" height="976" alt="image" src="https://github.com/user-attachments/assets/4c5e78f0-d8ed-46ce-8502-59dcdeef112d" />

# Residual Networks (ResNets) Summary

ResNets were introduced to address the **vanishing and exploding gradient** problems that make very deep "plain" networks difficult to train.

---

## 1. The Residual Block
The fundamental building block of a ResNet is the **Residual Block**. In a standard "plain" network, activations flow linearly from one layer to the next. In a ResNet, we introduce a **shortcut** or **skip connection**.



### The Math Behind the Block
In a standard network, the path is:
1. $z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]}$
2. $a^{[l+1]} = g(z^{[l+1]})$
3. $z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]}$
4. $a^{[l+2]} = g(z^{[l+2]})$

In a **Residual Block**, we inject $a^{[l]}$ directly into the calculation before the second ReLU nonlinearity:
$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$$

---

## 2. Plain Networks vs. ResNets
A ResNet is constructed by stacking multiple residual blocks together.

* **Plain Network:** A standard neural network where layers are stacked strictly one after another.
* **ResNet:** A network where every few layers (usually two) include a skip connection that allows information to bypass the main path.

---

## 3. Performance and Training Error
The inventors (Kaiming He et al.) observed a critical difference in how deep networks behave during training:

### Plain Networks
* **Theory:** Deeper networks should theoretically have lower training error.
* **Reality:** As depth increases, training error eventually starts to **increase**. This is because deep plain networks are significantly harder to optimize due to gradient issues.

### ResNets
* **Empirical Result:** Even as the number of layers exceeds 100 (and even up to 1,000), the training error continues to **decrease**.
* **Impact:** Skip connections make it much easier for the optimization algorithm (like Gradient Descent) to learn the identity function, ensuring that adding depth doesn't hurt performance.



---

## Key Takeaways
| Feature | Plain Network | Residual Network (ResNet) |
| :--- | :--- | :--- |
| **Path** | Linear only | Main path + Shortcut (Skip Connection) |
| **Optimization** | Becomes harder with depth | Remains effective even with 100+ layers |
| **Gradient Issues** | High risk of vanishing/exploding | Significantly mitigated by skip connections |
| **Training Error** | Increases after a certain depth | Continues to decrease with more depth |

# Why ResNets Work: Intuition and Implementation

The primary reason ResNets outperform "plain" networks is their ability to maintain performance even as depth increases by making it easy for the network to learn an **identity function**.

---

## 1. The "Identity Function" Intuition
In a standard deep network, it is surprisingly difficult for layers to learn to simply pass an input through without changing it (the identity function). In a ResNet, this becomes the "default" behavior.

### Mathematical Proof
Given a residual block:
$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$$
Expanding $z^{[l+2]}$:
$$a^{[l+2]} = g(W^{[l+2]}a^{[l+1]} + b^{[l+2]} + a^{[l]})$$

If we use **L2 Regularization** (weight decay), the weights $W^{[l+2]}$ and bias $b^{[l+2]}$ will shrink toward zero. If $W^{[l+2]} = 0$ and $b^{[l+2]} = 0$:
$$a^{[l+2]} = g(0 + a^{[l]}) = a^{[l]}$$
*(Assuming $g$ is a ReLU and $a^{[l]}$ is non-negative from a previous ReLU)*.

**Conclusion:** It is very easy for a residual block to learn $a^{[l+2]} = a^{[l]}$. This ensures that adding layers never hurts the training performance, as the network can simply "skip" layers that aren't useful.

---

## 2. Managing Dimensions ($W_s$)
For the addition $z^{[l+2]} + a^{[l]}$ to work, the vectors must have the same dimensions.

* **Same Convolutions:** Most ResNets use "same" padding to keep dimensions consistent across layers, making the shortcut addition straightforward.
* **Dimension Mismatch:** If dimensions differ (e.g., after a pooling layer), a linear projection matrix $W_s$ is used:
    $$a^{[l+2]} = g(z^{[l+2]} + W_s a^{[l]})$$
    * $W_s$ can be a set of **learnable parameters** to resize $a^{[l]}$.
    * $W_s$ can be a **fixed matrix** used for zero-padding.

---

## 3. ResNet Architecture on Images
The transition from a "Plain" network to a "ResNet" involves adding skip connections every two (or more) convolutional layers.



### Key Structural Patterns:
* **Convolution Layers:** Primarily $3 \times 3$ filters with "same" padding.
* **Periodic Downsampling:** Occasional pooling layers or strided convolutions reduce height and width.
* **Channel Doubling:** When spatial dimensions are halved, the number of filters is typically doubled to preserve representational capacity.
* **Final Layers:** Ends with Global Average Pooling followed by a Softmax output.

---

## Summary Comparison

| Concept | Plain Network | Residual Network |
| :--- | :--- | :--- |
| **Learning Identity** | Difficult; requires specific weight values | Easy; happens naturally when weights are small |
| **Depth Limit** | Training error increases with extreme depth | Can scale to 100+ or 1,000+ layers |
| **Connection Type** | Sequential only | Sequential + Skip Connections |
| **Performance** | Performance plateaus then degrades | Performance continues to improve or stays stable |
