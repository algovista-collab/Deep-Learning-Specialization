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
