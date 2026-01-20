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

# ⚡ The Word2Vec Skip-gram Model

The Skip-gram model simplifies the task of learning word embeddings by predicting a **Target ($t$)** word from a single **Context ($c$)** word.

---

## 1. Defining the Supervised Learning Problem
Instead of using a sequence of preceding words, Skip-gram randomly selects a "Context" word and then picks a "Target" word within a fixed window (e.g., $\pm 5$ or $\pm 10$ words).

**Example sentence:** *"I want a glass of orange juice to go..."*
If the Context word is **"orange"**, the training pairs might be:
* `(orange, juice)`
* `(orange, glass)`
* `(orange, my)`



---

## 2. Model Architecture
The model is essentially a shallow neural network consisting of an embedding lookup followed by a Softmax layer.

1.  **Input:** One-hot vector $O_c$ for the context word.
2.  **Embedding Lookup:** $e_c = E \cdot O_c$ (extracting the 300D vector).
3.  **Output (Softmax):** Calculates the probability of word $t$ given context $c$:
    $$P(t|c) = \frac{e^{\theta_t^T e_c}}{\sum_{j=1}^{10,000} e^{\theta_j^T e_c}}$$
    * $\theta_t$ is the parameter vector associated with the output word $t$.

---

## 3. The Computation Bottleneck
The primary weakness of the Skip-gram model is the **Softmax denominator**. 

* **The Problem:** Summing over the entire vocabulary (10k, 100k, or 1M words) for every single training step is computationally expensive and slow.
* **The Solution (Hierarchical Softmax):** Instead of a flat search, a tree-based classifier reduces the complexity from $O(V)$ to $O(\log V)$.
    * Common words like "the" or "of" are placed at the top of the tree for faster access.
    * Rare words like "durian" are placed deeper in the tree.

---

## 4. Word Sampling Heuristics
If you sample words strictly by their frequency in a text corpus, the training set will be dominated by "stop words" (e.g., *the, of, a, and*). 
* **Balancing:** Heuristics are used to down-sample frequent words and up-sample less common words. This ensures the model spends enough time learning meaningful embeddings for words like "apple" or "durian" rather than just "the."

---

## 5. Summary of Word2Vec Variants
| Model | Mechanism |
| :--- | :--- |
| **Skip-gram** | Uses one word to predict the surrounding "context" words. |
| **CBOW (Continuous Bag of Words)** | Uses the surrounding context words to predict the "middle" word. |

---

# 📉 Negative Sampling: Scaling Word2Vec

While the original Skip-gram model is effective, the **Softmax** step is a massive computational bottleneck because it requires summing over the entire vocabulary for every training example. **Negative Sampling** solves this.

---

## 1. Redefining the Problem
Instead of asking, *"What is the next word in the sequence?"* we ask a binary question: 
> **"Is this pair of words (Context, Target) a 'positive' pair (found together) or a 'negative' pair (randomly associated)?"**

### How to Generate the Dataset:
1.  **Positive Example:** Pick a context word (e.g., "orange") and a target word within a window (e.g., "juice"). Label = **1**.
2.  **Negative Examples:** Keep the same context word ("orange") and pick $k$ random words from the dictionary (e.g., "king", "book", "the"). Label = **0**.
    * *Note:* Even if a random word (like "of") actually appears near "orange" in the text, we still label it **0** for this specific sampling step.



---

## 2. The Model Architecture
We replace the giant Softmax with **$V$ binary logistic regression classifiers** (where $V$ is your vocabulary size).

* **Model Formula:**
    $$P(y=1 | c, t) = \sigma(\theta_t^T e_c)$$
    * $\sigma$: Sigmoid function.
    * $e_c$: Embedding vector for context word.
    * $\theta_t$: Parameter vector for target word.

### Why it's Faster:
In the original model, you updated **10,000+** parameters every step. With Negative Sampling, for each positive example, you only update $k+1$ binary classifiers (the positive one plus $k$ negative ones).
* For small datasets: $k = 5$ to $20$.
* For large datasets: $k = 2$ to $5$.



---

## 3. Selecting Negative Examples
How do we pick the $k$ random words? We shouldn't just pick them purely by frequency (too many "the's") or purely at random (not representative).

The authors found a "middle ground" heuristic using the **3/4 power of word frequency**:
$$P(w_i) = \frac{f(w_i)^{3/4}}{\sum_{j=1}^{V} f(w_j)^{3/4}}$$
* **$f(w_i)$:** The observed frequency of the word.
* **Result:** This increases the probability of sampling rare words and decreases the probability of sampling extremely common words compared to the raw distribution.

---

## 4. Summary Table

| Feature | Standard Softmax | Negative Sampling |
| :--- | :--- | :--- |
| **Output Type** | Multi-class (1-of-V) | Binary (Is it a pair?) |
| **Computation** | $O(V)$ — Expensive | $O(k)$ — Very Cheap |
| **Efficiency** | Slow on large vocabularies | Highly scalable |

---

# 🌐 The GloVe Algorithm (Global Vectors)

The GloVe algorithm is based on the idea that the relationship between words can be captured by examining the **global co-occurrence counts** across the entire text corpus.

---

## 1. The Co-occurrence Matrix ($X$)
GloVe starts by explicitly counting how often words appear near each other.
* Let **$X_{ij}$** be the number of times word $i$ (target) appears in the context of word $j$ (context).
* If we define "context" as any word within $\pm 10$ words, then $X_{ij} = X_{ji}$ (the relationship is symmetric).

---

## 2. The Optimization Objective
The goal is to learn vectors $\theta_i$ and $e_j$ such that their dot product predicts the log of their co-occurrence frequency.

**The Loss Function:**
$$\text{Minimize} \sum_{i=1}^{V} \sum_{j=1}^{V} f(X_{ij}) (\theta_i^T e_j - \log X_{ij})^2$$

* **$\theta_i^T e_j$**: The relationship between the two words.
* **$\log X_{ij}$**: The ground truth (how often they actually appear together).
* **$f(X_{ij})$**: A weighting function used to handle two issues:
    1. It equals $0$ if $X_{ij}=0$ (to avoid $\log 0$).
    2. It ensures "stop words" (the, a, is) don't dominate the learning, while still giving rare words (like "durian") enough weight to be meaningful.

---

## 3. Symmetry in GloVe
Unlike Word2Vec where $\theta$ and $e$ play different roles (context vs. target), in GloVe, they are **mathematically symmetric**. 
* Because they play the same role, the final embedding for a word $w$ is often calculated as the average:
$$e_w^{(final)} = \frac{e_w + \theta_w}{2}$$

---

## 4. The "Interpretability" Problem
While we often motivate embeddings using human concepts like **Gender**, **Age**, or **Royalty**, the actual dimensions learned by algorithms like GloVe are rarely that clean.



* **Linear Transformations:** Because of the nature of linear algebra, the algorithm might learn a set of features that are "rotated" or "sheared" versions of human concepts. 
* **Result:** Dimension 1 might be 30% Gender + 20% Royalty + 50% Age. You cannot simply look at one number in a 300D vector and know what it "means."
* **The Good News:** Even if individual dimensions aren't interpretable to humans, the **parallelogram relationship** (analogies) still works perfectly for the computer.



---

## 5. Comparison: Word2Vec vs. GloVe

| Feature | Word2Vec (Skip-gram) | GloVe |
| :--- | :--- | :--- |
| **Philosophy** | Local context prediction | Global co-occurrence counts |
| **Mechanism** | Neural Network / Softmax | Matrix Factorization / Least Squares |
| **Speed** | Fast for small windows | Efficient on large corpora counts |

---

# ⚖️ Debiasing Word Embeddings

Machine learning models are only as good as the data they are trained on. Because word embeddings are learned from text written by humans (the internet, books, etc.), they can inherit and even amplify societal biases regarding gender, race, and age.

---

## 1. The Problem: Learned Bias
Researchers discovered that word embeddings can pick up harmful analogies from the training corpus.
* **Positive Analogy:** *Man is to Computer Programmer as Woman is to Homemaker.*
* **Negative Analogy:** *Father is to Doctor as Mother is to Nurse.*

These biases aren't just conceptual; they affect real-world applications like resume screening or automated loan approvals.

---

## 2. The Bolukbasi et al. Debiasing Process
To fix this, we follow a three-step mathematical process to "neutralize" the embeddings.

### Step 1: Identify the Bias Direction
First, we find the specific axis in the 300D space that represents the bias (e.g., Gender).
* We subtract vector pairs: $e_{man} - e_{woman}$, $e_{male} - e_{female}$, etc.
* We average these differences to find the **Bias Direction** and its **Orthogonal (Non-Bias) Direction**.

### Step 2: Neutralize
For words that are not inherently gendered (like "Doctor," "Programmer," or "Poet"), we project them onto the orthogonal axis to remove the bias component.
* This ensures that the mathematical distance from "Doctor" to "Man" is exactly the same as the distance from "Doctor" to "Woman."

### Step 3: Equalize
For words that *do* have a biological gender (like "Grandmother" and "Grandfather"), we ensure they are equidistant from the neutral words. 
* If "Babysitter" has been neutralized, we want both "Grandmother" and "Grandfather" to be at the exact same distance from "Babysitter" so that the model doesn't favor one over the other.

---

## 3. Summary of Steps

| Step | Action | Goal |
| :--- | :--- | :--- |
| **Identify** | Average the difference of gendered pairs. | Define the "Bias Axis." |
| **Neutralize** | Project non-gendered words to the zero-bias axis. | Remove bias from jobs/traits. |
| **Equalize** | Move gendered pairs to be equidistant from neutral words. | Ensure symmetric relationships. |

---

## 4. Why This Matters
If you don't debias your embeddings:
1. **Search results** might be biased (e.g., searching "brilliant physicist" only shows male results).
2. **Hiring tools** might automatically penalize resumes that contain words associated with a specific gender or ethnicity.
3. **Language models** (like ChatGPT) might generate stereotypical or offensive content.

---

# ⚖️ Debiasing Word Embeddings

Word embeddings often inherit human biases present in the training text (e.g., "Man is to Computer Programmer as Woman is to Homemaker"). To prevent AI from making biased decisions in hiring, loans, or criminal justice, we use a formal debiasing process.

---

## 1. The Three-Step Debiasing Algorithm
Based on the work of **Bolukbasi et al.**, we can adjust the vector space to eliminate undesirable correlations while preserving legitimate semantic meanings.

### Step 1: Identify the Bias Direction
We isolate the specific axis in the 300D space that represents the bias.
* **Process:** Take pairs of gender-definitional words (e.g., $e_{he} - e_{she}$, $e_{male} - e_{female}$) and find their average difference.
* **Math:** In practice, researchers use **Singular Value Decomposition (SVD)** to find a "Bias Subspace" and a "Non-Bias Subspace" (the remaining 299 dimensions).



### Step 2: Neutralize (For Non-Definitional Words)
For words where gender is not part of the definition (e.g., "Doctor," "Babysitter," "Programmer"), we project the vector onto the non-bias subspace.
* **Goal:** This "zeroes out" the component of the word that points toward the bias axis. After this, "Doctor" is no longer mathematically "closer" to "Man" than to "Woman."

### Step 3: Equalize (For Definitional Pairs)
For words that *should* stay gendered (e.g., "Grandmother" vs. "Grandfather"), we ensure they are equidistant from the neutral axis.
* **Goal:** If "Grandmother" is closer to "Babysitter" than "Grandfather" is, the model still has a latent bias. Equalization moves both words so they have the exact same similarity to all neutralized words.



---

## 2. Definitional vs. Neutral Words
A key challenge is deciding which words to neutralize.
* **Definitional Words:** "Boy," "Girl," "Fraternity," "Niece." These have biological or definitional gender and are kept as-is (but equalized).
* **Neutral Words:** "Doctor," "Janitor," "Brilliant," "Kind." These should have zero bias.
* **Heuristic:** The authors trained a **linear classifier** to automatically categorize words into these two groups, finding that the vast majority of words in English are (and should be) gender-neutral.

---

## 3. Real-World Impact
Debiasing is not just a theoretical exercise; it has concrete consequences for social equity.

| Application | Bias Risk | Debiasing Benefit |
| :--- | :--- | :--- |
| **Job Search** | Ranking male resumes higher for "Engineer" roles. | Ensures gender-neutral candidate discovery. |
| **Loan Apps** | Associating low-income neighborhoods with high risk. | Promotes socioeconomic fairness in lending. |
| **Criminal Justice** | Biased sentencing guidelines based on ethnicity. | Reduces systemic discrimination in legal AI. |

---

## 4. Current Limitations
While these methods are effective, debiasing is an **active area of research**. Current algorithms can reduce explicit bias (like analogies), but subtle, "latent" biases can sometimes remain hidden in the high-dimensional relationships between words.

---
