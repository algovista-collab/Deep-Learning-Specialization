# Introduction to Convolutional Neural Networks (CNNs)

This course introduction explores how Deep Learning is transforming Computer Vision and why specialized architectures like CNNs are necessary for modern applications.

---

## 1. Real-World Applications
Deep learning has enabled significant leaps in several high-impact areas:
* **Autonomous Driving:** Detecting pedestrians and other vehicles to ensure safety.
* **Biometrics:** Highly accurate facial recognition for unlocking phones and doors.
* **Content Discovery:** Powering apps that curate relevant or "attractive" imagery.
* **Computational Art:** Creating new forms of art through algorithms.

---

## 2. Core Computer Vision Tasks
The course focuses on three primary types of problems:

1.  **Image Classification:** Determining the category of an image (e.g., "Cat" vs. "No Cat").
2.  **Object Detection:** Identifying multiple objects within a scene and providing their coordinates (drawing **bounding boxes**).
3.  **Neural Style Transfer:** Merging the *content* of one image with the *artistic style* of another.



---

## 3. The Challenge of High-Dimensional Inputs
A major bottleneck in computer vision is the sheer size of the input data.

| Image Size | Resolution | Input Features ($H \times W \times 3$) |
| :--- | :--- | :--- |
| **Small** | $64 \times 64$ | 12,288 |
| **Large** | $1000 \times 1000$ | 3,000,000 |

### The Parameter Explosion
If we use a standard **Fully Connected Network** on a 1-megapixel image:
* **Input Features ($x$):** 3 million.
* **Hidden Layer:** Assume 1,000 hidden units.
* **Total Parameters ($W^1$):** $1000 \times 3,000,000 = \mathbf{3 \text{ billion parameters}}$.

**Issues with this approach:**
* **Overfitting:** Too many parameters relative to typical dataset sizes.
* **Computation:** Memory and processing requirements for 3 billion parameters are infeasible.

---

## 4. The Path Forward: Convolutions
To process high-resolution images effectively, we must use the **Convolution Operation**. This allows the network to:
* Reduce the total number of parameters.
* Perform operations like **Edge Detection** to extract meaningful features.

<img width="927" height="456" alt="image" src="https://github.com/user-attachments/assets/f9db576c-8489-42d1-96ed-1a02791a080a" />

# The Convolution Operation: Edge Detection

The **Convolution Operation** is a fundamental building block of CNNs. It allows a network to process images and identify structural features like edges (edges are the locations in an image where there is a sharp change in brightness or color intensity), which are later combined to recognize complex objects.

---

## 1. How the Convolution Works
Convolution involves two main components: an **input image** and a **filter** (also called a **kernel**).

* **Input:** A matrix representing pixel intensities (e.g., a $6x6$ grayscale image).
* **Filter:** A smaller matrix (e.g., $3x3$) designed to detect a specific feature.
* **Operation:** The filter "slides" over the image. At each position, an **element-wise product** is calculated and summed to produce a single value in the output matrix.



### Mathematical Dimensions
For an input of size $n \times n$ and a filter of size $f \times f$:
* **Output Size:** $(n - f + 1) \times (n - f + 1)$
* **Example:** A $6 \times 6$ image convolved with a $3 \times 3$ filter results in a $4 \times 4$ output matrix.

---

## 2. Vertical Edge Detection Example
To detect vertical edges, we use a specific filter that highlights transitions from light to dark (or vice-versa).

### The Vertical Filter
$$
\begin{bmatrix} 
1 & 0 & -1 \\ 
1 & 0 & -1 \\ 
1 & 0 & -1 
\end{bmatrix}
$$



### Why it works:
1.  **Uniform Areas:** When the filter is over a solid color (e.g., all 10s or all 0s), the positive and negative sides cancel out, resulting in **0**.
2.  **Edge Areas:** When the filter sits on a transition (e.g., 10s on the left and 0s on the right), the math produces a **large positive or negative value**, signifying a strong edge.

---

## 3. Implementation in Frameworks
In practice, you don't write the nested loops for these multiplications. Deep learning frameworks provide optimized functions:
* **TensorFlow:** `tf.nn.conv2d`
* **Keras:** `Conv2D`
* **Programming Notation:** Usually represented by the asterisk symbol ($*$).

---

# Edge Detection: Beyond Basic Vertical Filters

We can refine edge detection by distinguishing between transition directions and using specialized filters. Ultimately, Deep Learning allows the network to learn these filters automatically.

---

## 1. Positive vs. Negative Edges
The direction of the brightness transition changes the sign of the output:
* **Light to Dark:** Using a vertical filter on a white-to-black transition results in a **positive** value (e.g., $+30$).
* **Dark to Light:** Applying the same filter to a black-to-white transition results in a **negative** value (e.g., $-30$).
* **Insight:** The sign tells the network the "direction" of the edge. Taking the absolute value would identify the edge regardless of direction.



---

## 2. Specialized Hand-Coded Filters
Researchers historically developed specific matrices to make edge detection more robust:

| Filter Type | Matrix Logic | Benefit |
| :--- | :--- | :--- |
| **Sobel Filter** | $\begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}$ | Puts more weight on the central pixel; more robust to noise. |
| **Scharr Filter** | $\begin{bmatrix} 3 & 0 & -3 \\ 10 & 0 & -10 \\ 3 & 0 & -3 \end{bmatrix}$ | Offers even stronger weight for specific edge properties. |



---

## 3. Horizontal Edge Detection
By rotating a vertical filter 90 degrees, we can detect horizontal transitions (top-to-bottom):
* **Filter:** $\begin{bmatrix} 1 & 1 & 1 \\ 0 & 0 & 0 \\ -1 & -1 & -1 \end{bmatrix}$
* **Logic:** Bright pixels on top and dark pixels on the bottom result in a strong positive output.

---

## 4. Learning Filters via Backpropagation
The most powerful idea in CNNs is that we don't need to hand-pick these numbers.
* **Parameters:** Treat the 9 numbers in a $3 \times 3$ filter as **parameters** ($w_1, w_2, \dots, w_9$).
* **Training:** Through backpropagation, the network learns the best values for these parameters based on the data.
* **Versatility:** The network can learn to detect edges at any angle (45°, 70°, etc.) or even complex patterns that humans don't have names for.

---

# Convolutional Networks: Padding

Padding is the process of adding a border of pixels (usually zeros) around the edges of an input image before applying a convolution.

---

## 1. Why do we need Padding?
Without padding, two major issues occur:
1.  **Shrinking Output:** Every time you apply a filter, the image gets smaller. In a deep network with 100 layers, the image would disappear entirely after a few operations.
2.  **Information Loss at Edges:** Pixels at the corners and edges are only "touched" once by the filter, whereas pixels in the middle are overlapped many times. This means the network "throws away" valuable detail from the image boundaries.



---

## 2. The Mathematics of Padding
If we have an $n \times n$ image, an $f \times f$ filter, and a padding amount $p$:
* The padded image becomes $(n + 2p) \times (n + 2p)$.
* **Output Size Formula:** $$(n + 2p - f + 1) \times (n + 2p - f + 1)$$

*Example:* A $6 \times 6$ image with $p=1$ becomes $8 \times 8$. Convolving with a $3 \times 3$ filter results in a $6 \times 6$ output ($8 - 3 + 1 = 6$), preserving the original size.

---

## 3. Valid vs. Same Convolutions
There are two standard conventions for choosing the amount of padding:

| Type | Definition | Padding Amount ($p$) | Output Size |
| :--- | :--- | :--- | :--- |
| **Valid** | No padding | $p = 0$ | $n - f + 1$ |
| **Same** | Output size = Input size | $p = \frac{f - 1}{2}$ | $n$ |

> **Note:** For "Same" convolution to work with a symmetric border, the filter size ($f$) must be **odd**.

---

## 4. Why Use Odd-Sized Filters ($3 \times 3, 5 \times 5$)?
While not a strict rule, computer vision practitioners almost exclusively use odd-numbered filters because:
1.  **Symmetry:** They allow for even padding on all sides (left/right and top/bottom).
2.  **Center Pixel:** They have a specific "central pixel," which is helpful for tracking the position of the filter relative to the image.

---

# Convolutional Networks: Strided Convolutions

Strided convolution is a variation of the convolution operation where the filter "jumps" multiple pixels at a time instead of moving one pixel at a time.

---

## 1. How Strided Convolution Works
The **stride ($s$)** determines the step size of the filter. 
* **Stride = 1:** The filter moves one pixel at a time (standard).
* **Stride = 2:** The filter jumps two pixels over (skipping one position).



This results in a much smaller output matrix, as there are fewer valid positions for the filter to sit.

---

## 2. Calculating Output Dimensions
When using an $n \times n$ image, an $f \times f$ filter, padding $p$, and stride $s$, the output size is calculated as:

$$\lfloor \frac{n + 2p - f}{s} + 1 \rfloor \times \lfloor \frac{n + 2p - f}{s} + 1 \rfloor$$

* **Rounding Down (Floor):** If the fraction is not an integer, we round down ($\lfloor \dots \rfloor$).
* **The "Legal" Rule:** In practice, a convolution is only computed if the filter is contained **entirely** within the image (or padded image). If a stride would cause the filter to hang off the edge, that computation is skipped.

---

## 3. Convolution vs. Cross-Correlation
There is a technical distinction between the math used in deep learning and the math used in signal processing:

* **Mathematical Convolution:** Requires flipping the filter horizontally and vertically (mirroring) before doing the element-wise products.
* **Cross-Correlation:** Skips the flipping and performs the element-wise product directly.

**In Deep Learning:** We skip the flipping step for simplicity. Even though we call it "convolution," it is technically **cross-correlation**.
* **Why it works:** In a neural network, the filter values are learned. If the network needs a flipped filter, it will simply learn those flipped values during training.
* **Benefit:** Omitting the flip simplifies the implementation and doesn't hurt performance.

---

## Summary of Parameters
| Parameter | Symbol | Effect on Output Size |
| :--- | :--- | :--- |
| **Padding** | $p$ | Increases output size |
| **Filter Size** | $f$ | Decreases output size |
| **Stride** | $s$ | Decreases output size (compresses) |

---

# Convolutions over Volumes

Most real-world images aren't just grids of pixels; they are 3D volumes. This lesson explains how to apply filters to multi-channel data.

---

## 1. Convolving with RGB Images
An RGB image is a volume of size **$6 \times 6 \times 3$** (Height $\times$ Width $\times$ Channels).
* **The Rule:** The number of channels in the **filter** must match the number of channels in the **input**.
* **The Filter:** To process an RGB image, you use a **$3 \times 3 \times 3$** filter.
* **The Math:** You perform $27$ multiplications ($3 \times 3 \times 3$) and sum them all together into a single number. 



> **Important:** Even though the input and filter are 3D, if you use **one** filter, the output is a **2D** matrix (e.g., $4 \times 4 \times 1$).

---

## 2. Detecting Specific Features
By adjusting the values in the three layers of the filter, you can target specific colors:
* **Red-Only Edge Detection:** Set the "Red" layer of the filter to edge-detection values ($1, 0, -1$) and set the Green and Blue layers to all zeros.
* **General Edge Detection:** Use the same edge-detection values across all three layers to detect edges regardless of color.

---

## 3. Multiple Filters (The Power of Depth)
In a real neural network, you don't just want to detect one type of edge. You might want to detect vertical edges, horizontal edges, 45° edges, etc., all at once.

1.  **Filter 1 (Vertical):** Produces a $4 \times 4$ output.
2.  **Filter 2 (Horizontal):** Produces a different $4 \times 4$ output.
3.  **Stacking:** You stack these outputs together to create a **volume**.



### Summary of Dimensions:
If you have:
* **Input:** $n \times n \times n_c$
* **Filter:** $f \times f \times n_c$
* **Number of Filters:** $n_c'$
* The third dimension (the "3" in $6 \times 6 \times 3$) is often called **Channels** or **Depth**. 

The **Output** will be: 
$$(n - f + 1) \times (n - f + 1) \times n_c'$$

---
