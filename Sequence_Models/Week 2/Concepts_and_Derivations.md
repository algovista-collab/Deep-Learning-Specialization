# Natural Language Processing & Word Embeddings

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

# 🗄️ The Embedding Matrix (E)

To implement word embeddings, we store all learned vectors in a single matrix, denoted as $E$. This matrix serves as a lookup table that maps high-dimensional one-hot vectors to dense, low-dimensional feature vectors.

---

## 1. Structure of the Matrix
If we have a vocabulary of **10,000 words** and want each word to have **300 features**, the dimensions of our matrix $E$ will be:
* **Rows:** 300 (The number of features/dimensions).
* **Columns:** 10,000 (The size of the vocabulary $V$).

Each column $j$ in the matrix represents the embedding vector $e_j$ for the $j^{th}$ word in the vocabulary.



---

## 2. The Retrieval Process (Matrix Multiplication)
Mathematically, extracting the embedding for a specific word is represented as the multiplication of the Embedding Matrix $E$ by a one-hot vector $O$.

### The Math:
Let's say the word **"Orange"** is at index `6257` in our vocabulary:
* $O_{6257}$ is a $(10000 \times 1)$ one-hot vector (all zeros, with a `1` at index 6257).
* $E$ is a $(300 \times 10000)$ matrix.

When you multiply them:
$$E \cdot O_{6257} = e_{6257}$$

Because the one-hot vector has only one non-zero element, this operation "selects" the $6257^{th}$ column of matrix $E$. The result is a **$(300 \times 1)$ dense vector** representing "Orange."

---

## 3. Implementation Note: Lookup vs. Multiplication
While the math is written as a matrix multiplication ($E \cdot O$), in practice, this is highly inefficient because you are multiplying a lot of zeros.

* **Theory:** $E \times \text{One-Hot Vector}$.
* **Practice:** Most deep learning frameworks (like TensorFlow or PyTorch) use a **Lookup Layer**. This simply fetches the column by its index, which is computationally much faster than performing a full matrix multiplication.

---

# 🧠 Learning Word Embeddings: From Language Models to Simplified Contexts

The development of word embeddings started with complex neural networks designed for language modeling and evolved into streamlined algorithms that focus on the relationship between "Context" and "Target" words.

---

## 1. The Neural Language Model (Bengio et al.)
The foundational approach to learning embeddings involves training a neural network to predict the next word in a sequence.

### How it Works:
1. **Input (Context):** A sequence of words (e.g., "I want a glass of orange").
2. **Embedding Lookup:** Each word's one-hot vector $O$ is multiplied by the embedding matrix $E$ to get a dense 300D vector $e$.
3. **Fixed Window:** To handle varying sentence lengths, a fixed window (hyperparameter) is used—for example, only the previous 4 words.
4. **Hidden Layer:** These 4 embedding vectors are concatenated (forming a 1,200D vector) and passed through a hidden layer with parameters $W^{(1)}, b^{(1)}$.
5. **Softmax (Target):** The output layer is a Softmax over the entire 10,000-word vocabulary to predict the probability of the target word (e.g., "juice").



---

## 2. Why Embeddings Emerge
The model learns meaningful embeddings because it is incentivized to group similar words together to minimize prediction error.
* If the model sees both "orange juice" and "apple juice," it realizes that to predict "juice" correctly, the vectors for **orange** and **apple** must be similar.
* This results in a feature-rich embedding matrix $E$ where semantically related words share similar coordinates.

---

## 3. Generalizing the "Context"
Researchers discovered that if the primary goal is to learn the matrix $E$ (and not necessarily to build a perfect language model), the definition of "Context" can be simplified or altered:

| Context Type | Description | Example (Target: "juice") |
| :--- | :--- | :--- |
| **Last $n$ words** | Standard language modeling. | "a glass of orange" |
| **Window around word** | $n$ words to the left and $n$ to the right. | "glass of... to go" |
| **Last 1 word** | Only the immediate predecessor. | "orange" |
| **Nearby 1 word** | A random word within a local window. | "glass" |



---

## 4. Key Takeaway
While complex models (using large windows and hidden layers) are great for language modeling, **simpler models** using a single nearby word as context (like the **Skip-gram** model) are remarkably effective at learning high-quality word embeddings while being much faster to train.

---
