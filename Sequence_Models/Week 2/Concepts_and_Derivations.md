# 📘 Summary: Natural Language Processing & Word Embeddings

This lecture introduces the shift from traditional word representations to dense, featurized vectors, which has revolutionized the field of NLP.

---

## 1. The Limitation of One-Hot Representation
In previous models, words were represented using a **one-hot vector** ($O_{index}$). For a vocabulary of 10,000 words, a word is a vector of 10,000 zeros with a single `1` at its specific index.

* **The "Island" Problem:** It treats every word as an independent entity with no relationship to others.
* **Zero Similarity:** The inner product (dot product) between any two different one-hot vectors is **0**.
* **Failure to Generalize:** If a model learns that "orange juice" is a common phrase, it cannot automatically infer that "apple juice" is also likely because the vectors for "orange" and "apple" are mathematically as different as "orange" and "king."

---

## 2. Featurized Representation (Word Embeddings)
To solve this, we represent words as **dense vectors** of features (e.g., a 300-dimensional vector denoted as $e_{index}$). Each dimension represents an abstract attribute.

### Conceptual Feature Table
| Feature | Man | Woman | King | Queen | Apple | Orange |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gender** | -1.0 | +1.0 | -0.95 | +0.97 | 0.0 | 0.0 |
| **Royalty** | 0.01 | 0.02 | 0.93 | 0.95 | -0.01 | 0.0 |
| **Age** | 0.03 | 0.02 | 0.7 | 0.7 | 0.0 | 0.0 |
| **Food** | 0.0 | 0.0 | 0.0 | 0.0 | 0.95 | 0.97 |

**Why it works:**
* **Mathematical Similarity:** The vectors for "Apple" and "Orange" will be very close in 300D space.
* **Analogy Detection:** It allows algorithms to understand relationships like: 
    > *Man is to Woman as King is to Queen.*

---

## 3. Visualizing Embeddings (t-SNE)
Because humans cannot visualize 300 dimensions, we use dimensionality reduction algorithms like **t-SNE** to map these points into a 2D plane.



* **Clustering:** Related concepts (animals, fruits, numbers, etc.) naturally group together in the embedding space.
* **The term "Embedding":** We call them embeddings because we are "embedding" a word into a point in a high-dimensional continuous space.

---

## 4. Key Advantages and Ethics
1.  **Generalization:** Allows models to perform well even with relatively small labeled training sets.
2.  **Transfer Learning:** You can learn these embeddings from massive amounts of unlabeled text and apply them to specific tasks.
3.  **Debiasing:** A critical step in the pipeline is identifying and reducing **gender, ethnic, or other biases** that the algorithm might learn from the source text (e.g., ensuring "Doctor" isn't automatically closer to "Man" than "Woman").

---
