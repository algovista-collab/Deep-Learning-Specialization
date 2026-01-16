# Summary: Introduction to Face Recognition

This section introduces the final week of the Convolutional Neural Networks course, focusing on **Face Recognition** and **Neural Style Transfer**. We distinguish between basic verification and the much more complex task of recognition.

---

## 1. Key Terminology
In computer vision literature, it is important to distinguish between two distinct tasks:

* **Face Verification (1:1):** * **Input:** An image and a name/ID.
    * **Goal:** To determine if the input image matches the claimed identity.
    * **Example:** A smartphone unlocking when it sees the owner's face.
* **Face Recognition (1:K):** * **Input:** An image.
    * **Goal:** To identify who the person is from a database of $K$ people (or "not recognized").
    * **Example:** A camera at an office entrance recognizing an employee among hundreds of staff members.

---

## 2. Why Recognition is Harder
The complexity of the recognition problem scales with the size of the database ($K$):
* If a verification system is **99% accurate**, it has a 1% chance of error per check.
* In a recognition system with **$K=100$ people**, you are essentially performing the verification task 100 times.
* The cumulative chance of making an error increases significantly. To have a reliable recognition system, the underlying verification accuracy often needs to be **99.9% or higher**.

---

## 3. Liveness Detection
A robust face recognition system often includes **Liveness Detection** to prevent "spoofing."
* **Definition:** Ensuring the input is a live human rather than a photograph, video, or mask.
* **Method:** This is typically treated as a supervised learning problem where the model is trained to distinguish between "live human" and "not live human."



---

## 4. Summary Table

| Feature | Face Verification | Face Recognition |
| :--- | :--- | :--- |
| **Comparison Type** | 1:1 | 1:K |
| **Input Requirements** | Image + ID | Image only |
| **Complexity** | Lower | Much higher (scales with $K$) |
| **Core Question** | "Is this person who they say they are?" | "Who is this person?" |

---

## 5. Moving Toward One-Shot Learning
The standard way to build these systems is to first master **Face Verification** as a building block. However, most face recognition applications face the **One-Shot Learning** challenge: being able to recognize a person after seeing only one example of their face.

# Summary: One-Shot Learning for Face Recognition

One-shot learning is the challenge of correctly identifying a person after seeing only one training example. This is a common requirement for facial recognition systems, where an employee database might only contain a single ID photo for each person.

---

## 1. The Problem with Standard Softmax
A traditional Convolutional Neural Network (CNN) that outputs a discrete label via Softmax is usually ineffective for face recognition because:
* **Small Data:** Deep learning models typically require thousands of examples to generalize; one image per person is insufficient to train a robust classifier.
* **Scalability:** If a new person joins the organization, the network would need to be modified (adding a new output unit) and completely retrained, which is not feasible for growing databases.

---

## 2. The Solution: Learning a Similarity Function
Instead of teaching the network to "classify" a specific face, we teach it to learn a **Similarity Function** ($d$).

* **Function Definition:** $d(img1, img2) = \text{degree of difference between images}$.
* **Logic:** * If the images are of the same person $\rightarrow$ $d$ should be a **small** number.
    * If the images are of different people $\rightarrow$ $d$ should be a **large** number.



---

## 3. The Recognition Process (Verification vs. Recognition)
To verify if a new image matches someone in the database:
1.  **Pairwise Comparison:** Compare the new image against every image in the database using the function $d$.
2.  **Thresholding ($\tau$):** Use a hyperparameter threshold, $\tau$.
    * If $d(img1, img2) \leq \tau$, the system predicts they are the **same person**.
    * If $d(img1, img2) > \tau$, the system predicts they are **different persons**.

---

## 4. Why This Solves One-Shot Learning
This approach decouples the "learning" from the "database":
* **Dynamic Databases:** You can add or remove people from your database without ever retraining the underlying neural network. 
* **Generalization:** The network learns the general concept of "what makes faces similar or different," which it can apply to people it has never seen during its initial training phase.

---

## 5. Summary Table: Softmax vs. Similarity Function

| Feature | Standard Softmax Approach | Similarity Function ($d$) |
| :--- | :--- | :--- |
| **Output** | A specific ID label | A distance/difference score |
| **New User Addition** | Requires retraining the model | Simply add a photo to the database |
| **Data Requirement** | Many photos per person | Works with one photo (One-Shot) |
| **Main Advantage** | Simple for fixed categories | Flexible and scalable for identities |

# Summary: Siamese Networks for Face Recognition

The **Siamese Network** is the architectural answer to the one-shot learning problem. Instead of classifying a face into a specific category, it transforms an image into a unique numerical "fingerprint" called an encoding.

---

## 1. What is a Siamese Network?
A Siamese network consists of **two (or more) identical sub-networks**. "Identical" means they share the exact same parameters, weights, and biases.
* **Input:** You feed two different images ($x^{(1)}$ and $x^{(2)}$) into these two identical pipelines.
* **Architecture:** Each pipeline is a standard ConvNet (Convolutional, Pooling, and Fully Connected layers).
* **Output:** Instead of a Softmax classification, the network outputs a **feature vector** (typically 128 numbers).



---

## 2. Encodings ($f(x)$)
The vector of 128 numbers is called an **encoding**. 
* We denote the encoding of an image $x$ as $f(x)$.
* Think of $f(x)$ as a points in a high-dimensional space.
* The network's job is to learn how to map an image to a point in this space such that the location represents the "identity" of the face.

---

## 3. The Distance Function ($d$)
To compare two faces, we calculate the distance between their two encodings. This is typically done using the **Squared Euclidean Norm ($L2$ Norm)**:

$$d(x^{(1)}, x^{(2)}) = ||f(x^{(1)}) - f(x^{(2)})||^2_2$$

### The Learning Goal:
The network is trained with a specific goal for its parameters:
1.  **If $x^{(1)}$ and $x^{(2)}$ are the same person:** The distance $d$ should be **small**.
2.  **If $x^{(1)}$ and $x^{(2)}$ are different people:** The distance $d$ should be **large**.

---

## 4. Why "Siamese"?
The name comes from "Siamese twins" because the two networks are joined at the end by the distance function and share a single set of "DNA" (the weights). This ensures that the way the network perceives a face is consistent, regardless of which "twin" processes the image.



---

## 5. Summary Table: Encoder vs. Classifier

| Feature | Standard CNN Classifier | Siamese Network (Encoder) |
| :--- | :--- | :--- |
| **Last Layer** | Softmax (probabilities) | Fully Connected (Feature Vector) |
| **Output Type** | Predicted Class (e.g., "ID #402") | 128D Embedding ($f(x)$) |
| **Goal** | Minimize classification error | Minimize/Maximize relative distance |
| **Comparison** | Looks at one image at a time | Compares two images' embeddings |

---
