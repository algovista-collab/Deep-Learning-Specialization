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

# Summary: Triplet Loss Function

The **Triplet Loss** is the standard objective function used to train Siamese Networks for face recognition. It teaches the network to output encodings such that images of the same person are "pushed" together in the feature space, while images of different people are "pulled" apart.

---

## 1. Defining the Triplet
To train the network, you look at three images simultaneously:
* **Anchor (A):** A base image of a specific person.
* **Positive (P):** A *different* image of the same person as the Anchor.
* **Negative (N):** An image of a *different* person.



---

## 2. The Loss Equation and the "Margin"
We want the distance between Anchor and Positive $d(A, P)$ to be smaller than the distance between Anchor and Negative $d(A, N)$.

**The Core Constraint:**
$$\|f(A) - f(P)\|^2_2 + \alpha \leq \|f(A) - f(N)\|^2_2$$

* **$\alpha$ (Alpha):** This is called the **Margin**. It prevents the network from taking a "trivial" shortcut (like setting all encodings to zero). It forces the network to ensure that $d(A, N)$ is at least $\alpha$ units larger than $d(A, P)$.

**The Loss Function ($L$):**
$$L(A, P, N) = \max(\|f(A) - f(P)\|^2_2 - \|f(A) - f(N)\|^2_2 + \alpha, 0)$$

---

## 3. Selecting "Hard" Triplets
If you choose $A, P,$ and $N$ randomly, the constraint is often too easy to satisfy (e.g., a random person usually looks nothing like you), and the network won't learn much. To make training effective, you must find **"Hard" Triplets**:
* **Hard Triplets:** Choose $A, P,$ and $N$ such that $d(A, P) \approx d(A, N)$.
* This forces the gradient descent to work harder to push the negative further away and pull the positive closer.

---

## 4. Training Requirements
* **Dataset Structure:** To generate triplets, you need a training set where you have **multiple images of the same person** (e.g., 10 pictures of each of 1,000 different people).
* **One-Shot Application:** Even though you need many photos per person for *training*, once the model is trained, it can successfully perform **one-shot learning** (recognizing a new person from just one photo).
* **Data Volume:** Modern commercial systems (like FaceNet or DeepFace) are often trained on massive datasets (10M to 100M+ images).

---

## 5. Summary Table

| Term | Role in Triplet Loss | Goal |
| :--- | :--- | :--- |
| **Anchor (A)** | Reference point | N/A |
| **Positive (P)** | Same identity as A | Minimize $\|f(A) - f(P)\|^2$ |
| **Negative (N)** | Different identity | Maximize $\|f(A) - f(N)\|^2$ |
| **Margin ($\alpha$)** | Safety gap | Ensure a distinct distance between P and N |

# Summary: Face Recognition as Binary Classification

As an alternative to Triplet Loss, face recognition can be framed as a straightforward **binary classification problem**. In this approach, a Siamese Network is trained to predict whether a pair of images represents the same person ($y=1$) or different people ($y=0$).

---

## 1. The Binary Classification Pipeline
Instead of comparing three images at once, the system processes **pairs** of images through the following steps:
1.  **Twin Encodings:** Both images ($x^{(i)}$ and $x^{(j)}$) are passed through identical Siamese sub-networks to produce high-dimensional encodings ($f(x^{(i)})$ and $f(x^{(j)})$).
2.  **Difference Layer:** A custom layer computes the difference between these two encodings.
3.  **Logistic Regression:** These differences are fed into a logistic regression unit (using a sigmoid function) to output a probability $\hat{y}$.

---

## 2. Computing the Similarity
A common way to calculate the features for the logistic regression unit is the **Element-wise Absolute Difference**:

$$\hat{y} = \sigma \left( \sum_{k=1}^{128} w_k |f(x^{(i)})_k - f(x^{(j)})_k| + b \right)$$

### Alternative: Chi-Square Similarity
Another variation mentioned in the *DeepFace* paper is the **$\chi^2$ (Chi-Square) Similarity**:
$$\frac{(f(x^{(i)})_k - f(x^{(j)})_k)^2}{f(x^{(i)})_k + f(x^{(j)})_k}$$
This formula is often more sensitive to subtle variations in image features like texture and color.



---

## 3. The Deployment Trick: Pre-computing Encodings
In a real-world system (like an office turnstile), you might have thousands of employees in a database. Comparing a new face against every raw image would be too slow.

* **The Hack:** Pre-compute and store the **encodings** ($f(x)$) for everyone in your database.
* **The Benefit:** When a person walks up to the camera, the system only needs to run the CNN **once** for that new person. It then compares that single new encoding against the stored library of 128-bit vectors, which is computationally very "cheap."

---

## 4. Comparison Table: Triplet Loss vs. Binary Classification

| Feature | Triplet Loss | Binary Classification |
| :--- | :--- | :--- |
| **Input** | Triplets (Anchor, Positive, Negative) | Pairs (Image 1, Image 2) |
| **Labels** | N/A (Relative distance) | Binary (0 or 1) |
| **Complexity** | High (Requires "Hard Triplet" mining) | Moderate (Standard supervised learning) |
| **Output** | An embedding space | A similarity probability |

---

## 5. Summary of the Training Set
To train this system, you create a dataset of **pairs**:
* **Positive Pairs ($y=1$):** Two different photos of the same person.
* **Negative Pairs ($y=0$):** Photos of two different people.

# Summary: Visualizing What ConvNets Learn

To build an intuition for Neural Style Transfer, it is helpful to understand what different layers of a Convolutional Neural Network (CNN) are actually "seeing" or detecting. We can visualize this by finding image patches that maximally activate specific hidden units.

---

## 1. The Layer-by-Layer Evolution
As an image passes through a CNN, the features detected become increasingly complex and abstract:

* **Layer 1 (Shallow):** Detects very simple features. If you look at the neurons here, they are mostly activated by **edges** (at various angles) and **solid colors**.
* **Layer 2:** Starts to recognize more complex shapes and patterns. You might see neurons responding to **circles, stripes, or corners**.
* **Layer 3:** Detects more recognizable textures and parts. Examples include **honeycomb patterns, complex textures**, or even parts of objects like **wheels**.

* **Layer 4:** Begins to detect specific object parts. A neuron might activate strongly for **dog legs, bird beaks, or water**.
* **Layer 5 (Deep):** Detects entire high-level concepts or complex objects. Neurons here might respond to **full faces, flowers, or specific species of animals**.


---

## 2. Feature Visualization via Optimization
How do we actually "see" these features?
1.  **Pick a Unit:** Select a specific hidden unit (neuron) in a layer.
2.  **Scan the Dataset:** Pass many images through the network and find the image patches that cause that specific unit to have the highest activation.
3.  **Synthesize:** Alternatively, use gradient ascent (like in *DeepDream*) to generate an image from scratch that perfectly satisfies what that specific neuron is "looking for."

---

## 3. Implications for Style Transfer
Understanding these layers is crucial for Neural Style Transfer because:
* **Style** is often captured in the **shallow to middle layers** (textures, local patterns, color distributions).
* **Content** is often captured in the **deeper layers** (the high-level structure and arrangement of objects).

---

## 4. Summary Table of Hierarchical Features

| Layer Depth | Feature Type | Examples |
| :--- | :--- | :--- |
| **Shallow (L1, L2)** | Low-level | Edges, Colors, Simple Gradients |
| **Middle (L3)** | Mid-level | Textures, Repeating Patterns, Simple Shapes |
| **Deep (L4, L5)** | High-level | Object Parts (eyes, legs), Complex Entities (dogs, cars) |

---
