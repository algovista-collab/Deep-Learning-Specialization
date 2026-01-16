# Summary: Object Localization Fundamentals

Object localization is the bridge between simple image classification and complex object detection. It involves not only identifying **what** an object is but also **where** it is located within the image.

---

## 1. Problem Hierarchy
In the terminology of computer vision, tasks are categorized by their complexity and the number of objects they handle:

* **Image Classification:** Identify the class (e.g., "car"). Usually one dominant object.
* **Classification with Localization:** Identify the class AND draw a bounding box around it. Usually one dominant object.
* **Object Detection:** Identify and localize multiple objects of different categories within the same image.



---

## 2. Defining the Bounding Box
To tell a computer where an object is, we use four parameters to define a **Bounding Box**:

* **$b_x, b_y$**: The coordinates of the **center point** of the object.
* **$b_h$**: The **height** of the box.
* **$b_w$**: The **width** of the box.

### Coordinate Convention
* The top-left of the image is $(0,0)$.
* The bottom-right of the image is $(1,1)$.
* All parameters ($b_x, b_y, b_h, b_w$) are relative to the image dimensions (values between 0 and 1).

---

## 3. The Target Label Vector ($y$)
In supervised learning for localization, the ground truth label $y$ is a vector containing the class probability, the bounding box, and the class labels.

For a system detecting 3 classes (1: Pedestrian, 2: Car, 3: Motorcycle), the vector $y$ looks like this:

$$y = \begin{bmatrix} p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \end{bmatrix}$$

### Component Definitions:
1.  **$p_c$ (Presence of object):** If an object is present, $p_c = 1$. If the image is background (no object), $p_c = 0$.
2.  **$b_x, b_y, b_h, b_w$**: Bounding box coordinates.
3.  **$c_1, c_2, c_3$**: One-hot representation of the class. (e.g., for a car, $c_1=0, c_2=1, c_3=0$).

---

## 4. Handling "Don't Care" Scenarios
If there is no object in the image ($p_c = 0$), the remaining elements of the vector ($b_x \dots c_3$) are irrelevant. We treat them as **"don't cares"** during training.

* **Example (Object):** $y = [1, 0.5, 0.7, 0.3, 0.4, 0, 1, 0]^T$
* **Example (No Object):** $y = [0, ?, ?, ?, ?, ?, ?, ?]^T$

---

## 5. Loss Function
The loss function is calculated differently depending on the value of $p_c$. Using **Squared Error** for simplicity:

* **If $y_1 = 1$ (Object present):**
    $$L(\hat{y}, y) = \sum_{i=1}^{8} (\hat{y}_i - y_i)^2$$
    *The loss is the sum of squared differences for all 8 components.*

* **If $y_1 = 0$ (No object):**
    $$L(\hat{y}, y) = (\hat{y}_1 - y_1)^2$$
    *The loss only cares about the accuracy of the $p_c$ prediction.*

> **Note:** While squared error is easy to conceptualize, in practice, you might use **logistic regression loss** for $p_c$, **squared error** for the bounding box, and **softmax loss** for the classes.

---

## 6. Key Takeaway
By modifying a standard Convolutional Neural Network (ConvNet) to output a vector of real numbers (regression) rather than just a single class probability, we can teach the network to localize objects with high precision.
