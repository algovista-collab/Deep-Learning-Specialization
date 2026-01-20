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

# 🚀 Word Embeddings and Transfer Learning

Using featurized representations (embeddings) instead of one-hot vectors allows NLP models to perform significantly better, especially when labeled training data is scarce.

---

## 1. The Power of Generalization
One-hot vectors fail when a model encounters a word it hasn't seen in the training set. Word embeddings bridge this gap by grouping similar concepts.

* **Training Example:** "Sally Johnson is an **orange farmer**." 
    * The model learns that the phrase "orange farmer" usually precedes a person's name.
* **Test Case (Common):** "Robert Lin is an **apple farmer**."
    * Since the embedding for *apple* is similar to *orange*, the model easily generalizes.
* **Test Case (Rare):** "Robert Lin is a **durian cultivator**."
    * Even if the model has never seen "durian" or "cultivator" in its labeled training set, it knows from large-scale unlabeled text that *durian* ≈ *fruit* and *cultivator* ≈ *farmer*.

---

## 2. Transfer Learning Workflow
Word embeddings are a classic example of transfer learning in deep learning.

1.  **Learn (or Download):** Obtain embeddings from a massive corpus of unlabeled text (e.g., 1–100 billion words from the Internet) or use pre-trained embeddings (Word2Vec, GloVe, etc.).
2.  **Transfer:** Apply these embeddings (e.g., 300-dimensional dense vectors) to a task with a smaller labeled dataset, such as Named Entity Recognition (NER).
3.  **Fine-tune (Optional):** If the new task has a large enough dataset, you can continue to adjust the word embeddings to better fit the specific data.

---

## 3. When to Use Word Embeddings
Transfer learning via embeddings is most effective when:
* **High Utility:** The target task (NER, Text Summarization, Parsing) has a relatively **small** labeled dataset.
* **Low Utility:** The target task (Machine Translation, Language Modeling) already has a **massive** amount of task-specific data.

---

## 4. Embeddings vs. Face Encodings
The concepts of "Embeddings" in NLP and "Encodings" in Face Recognition are mathematically similar but used differently:

| Feature | Face Recognition (Siamese Net) | Word Embeddings (NLP) |
| :--- | :--- | :--- |
| **Input** | Any new/unseen face image. | A fixed vocabulary (e.g., 10,000 words). |
| **Output** | A dynamic vector computed for any image. | A fixed vector ($e_{i}$) learned for each dictionary word. |
| **Goal** | Distinguish between millions of possible faces. | Map specific known words to a shared feature space. |

---

# 📐 Analogy Reasoning & Cosine Similarity

Word embeddings aren't just lists of numbers; they represent a high-dimensional "concept map" where the mathematical distance between points corresponds to real-world relationships.

---

## 1. The Vector Arithmetic of Analogies
The most famous property of word embeddings is their ability to solve analogies (e.g., *Man is to Woman as King is to Queen*) using simple subtraction and addition.

### How it Works:
1. **Isolate the Difference:** Subtracting $e_{woman}$ from $e_{man}$ results in a vector that represents the concept of **"Gender."**
2. **Apply to a New Word:** If you take the vector for $e_{king}$ and subtract that same "Gender" vector, the closest resulting point in the 300D space is $e_{queen}$.

**The Formula:**
To find the answer to "Man is to Woman as King is to ??", the algorithm looks for a word $w$ that maximizes the similarity to:
$$e_{w} \approx e_{king} - e_{man} + e_{woman}$$

---

## 2. Measuring Similarity: Cosine Similarity
To find which word is "closest" to our calculated result, we use **Cosine Similarity**. This measures the cosine of the angle between two vectors.

**The Formula:**
$$\text{Similarity}(u, v) = \frac{u^T v}{||u||_2 ||v||_2}$$

* **1.0 (Angle = 0°):** The vectors point in the exact same direction (highly similar).
* **0 (Angle = 90°):** The vectors are orthogonal (no relationship).
* **-1.0 (Angle = 180°):** The vectors point in opposite directions.

*Note: While Euclidean distance can be used, Cosine Similarity is preferred in NLP because it focuses on the direction (meaning) rather than the magnitude (length) of the vector.*

---

## 3. Types of Analogies Learned
Because these models are trained on massive datasets (like the entire internet), they capture a wide range of relationships automatically:

| Relationship Type | Example |
| :--- | :--- |
| **Gender** | Man $\rightarrow$ Woman :: Boy $\rightarrow$ Girl |
| **Capital Cities** | Ottawa $\rightarrow$ Canada :: Nairobi $\rightarrow$ Kenya |
| **Grammar** | Big $\rightarrow$ Bigger :: Tall $\rightarrow$ Taller |
| **Currency** | Yen $\rightarrow$ Japan :: Ruble $\rightarrow$ Russia |

---

## 4. A Note on Visualization (t-SNE)
It is important to remember that while 2D visualizations (like t-SNE) help humans see clusters, the **parallelogram relationship** (analogies) usually only holds true in the original high-dimensional space (e.g., 300D). t-SNE is a non-linear mapping that often breaks these linear mathematical relationships.

---
