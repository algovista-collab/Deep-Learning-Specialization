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

<img width="635" height="447" alt="image" src="https://github.com/user-attachments/assets/19cb4c5d-4387-41aa-8479-ee86f105ad19" />

* Perform operations like **Edge Detection** to extract meaningful features.

