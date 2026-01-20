# Sequence-to-Sequence Models

This summary covers the foundational concepts of **Sequence-to-Sequence (Seq2Seq)** models as applied to machine translation and image captioning.

---

## 1. Sequence-to-Sequence (Seq2Seq) Basics
The Seq2Seq architecture maps an input sequence to an output sequence, even when their lengths differ (e.g., $x_1 \dots x_5$ to $y_1 \dots y_6$). It consists of two primary components:

* **The Encoder:** An RNN (typically GRU or LSTM) that processes the input sequence (e.g., a French sentence) one word at a time. It compresses the information into a single representative **feature vector**.
* **The Decoder:** A second RNN that takes the encoder's output vector as its starting point and generates the output sequence (e.g., an English translation) one word at a time until it reaches an **End of Sequence (EOS)** token.



## 2. Image Captioning (Image-to-Sequence)
A similar architecture is used for computer vision. The primary difference is the type of encoder used:

* **The Encoder (CNN):** A pre-trained **Convolutional Neural Network** (like AlexNet) processes an image. By removing the final Softmax layer, the model provides a high-dimensional feature vector (e.g., 4,096 dimensions) representing the image.
* **The Decoder (RNN):** This vector is fed into an RNN, which generates a descriptive caption (e.g., "A cat sitting on a chair") word by word.



## 3. Key Research & Foundations
The transcript references several seminal papers and researchers:
* **Machine Translation:** Sutskever et al., and Cho et al.
* **Image Captioning:** Mao et al., Vinyals et al., and Karpathy & Li.

## 4. Key Distinction: Search vs. Synthesis
Unlike basic language models that might randomly sample words to generate creative text, Seq2Seq applications like translation require finding the **most likely** sequence.

# Summary: Machine Translation as a Conditional Language Model

This lesson explains how machine translation adapts traditional language models to find the most likely translation using an encoder-decoder framework.

---

## 1. Defining the Conditional Language Model
Machine translation can be viewed as building a **Conditional Language Model**. While a standard language model estimates the probability of a sentence $P(y_1, \dots, y_{T_y})$, a machine translation model estimates the probability of an output sequence **conditioned** on an input sequence $x$:

$$P(y_1, \dots, y_{T_y} \mid x_1, \dots, x_{T_x})$$

### Comparison of Architectures
* **Language Model:** Starts with a vector of zeros ($a_0 = \vec{0}$) to generate text.
* **Seq2Seq Model:** Replaces the zero-vector with an **Encoder Network** (Green) that computes a representation of the input sentence $x$. This representation is then passed to the **Decoder Network** (Purple).



---

## 2. The Goal: Finding the Most Likely Sentence
In creative text synthesis, you might sample outputs at random. However, in translation, the objective is to find the specific sequence of words $y$ that maximizes the conditional probability:

$$\arg \max_{y} P(y \mid x)$$

Randomly sampling from the distribution might lead to "okay" translations that are wordy or slightly inaccurate (e.g., *"Jane is going to be visiting Africa..."*) rather than the most succinct and accurate one (*"Jane is visiting Africa..."*).

---

## 3. Why Greedy Search Fails
**Greedy Search** is an algorithm that picks the single most likely word at each step and moves on.

* **The Problem:** Picking the most likely *first* word doesn't guarantee the most likely *entire* sentence. 
* **Example:** "Jane is visiting" might be the best overall translation. However, if "is going" is more common in English, a greedy algorithm might pick "going" as the third word. Once that choice is made, the model is stuck with a less optimal, more verbose sentence.
* **Computational Limitation:** With a vocabulary ($V$) of 10,000 words, a 10-word sentence has $10,000^{10}$ possible combinations. This space is too large to check every possible sentence (exhaustive search).

---

## 4. The Solution: Approximate Search Algorithms
Because we cannot enumerate every possible translation, we use **Approximate Search Algorithms**. These methods do not guarantee finding the absolute maximum probability, but they are highly effective at finding a "good enough" maximum in a reasonable amount of time.

# Summary: The Beam Search Algorithm

Beam Search is the standard algorithm for finding the most likely output sequence in tasks like machine translation and speech recognition. Unlike Greedy Search, it explores multiple possibilities simultaneously.

---

## 1. The Core Concept: Beam Width ($B$)
The algorithm is defined by a parameter $B$, called the **beam width**. 
* **Greedy Search:** Effectively a Beam Search with $B=1$. It only keeps the single most likely word at each step.
* **Beam Search ($B=3$):** Keeps track of the top 3 most likely partial sequences at every step of the process.



---

## 2. Step-by-Step Execution
Using the example: *"Jane visite l'Afrique en Septembre"* $\rightarrow$ *"Jane visits Africa in September"*

### Step 1: Picking the first word ($y_1$)
1. Run the French sentence through the **Encoder**.
2. The **Decoder** outputs a softmax distribution over the entire vocabulary (e.g., 10,000 words).
3. Instead of picking just one word, Beam Search selects the **top $B$ words** (e.g., "in", "Jane", "September") and stores them in memory.

### Step 2: Picking the second word ($y_2$)
1. For **each** of the $B$ candidates, the model evaluates the probability of every possible second word in the vocabulary.
2. It calculates the **joint probability** of the first two words:
   $$P(y_1, y_2 \mid x) = P(y_1 \mid x) \times P(y_2 \mid x, y_1)$$
3. If the vocabulary is 10,000, the model considers $B \times 10,000$ (e.g., 30,000) total combinations.
4. It then selects the **new top $B$** overall pairs. 
   * *Note: This may result in some of the original first words being dropped entirely if their subsequent pairs are not among the top 3.*

### Step 3: Iteration
1. This process repeats for $y_3, y_4$, and so on.
2. At each step, you instantiate $B$ copies of the network to efficiently evaluate the potential next words for your $B$ candidates.
3. The process continues until the model generates an **<EOS>** (End of Sequence) token.



---

## 3. Why It Is Effective
* **Memory vs. Accuracy:** It is a middle ground between Greedy Search (fast but often inaccurate) and Breadth-First Search (accurate but computationally impossible).
* **Global Optimization:** By keeping several "hypotheses" alive, it can recover from a word that seemed likely at the start but leads to a poor sentence later.

> **Summary:** Beam Search doesn't guarantee finding the absolute maximum probability, but it significantly improves the quality of translations by exploring a much wider, yet manageable, search space.

---
> **Crucial Difference:** You don't want a random translation; you want the *optimal* translation. This requires algorithms like **Beam Search** rather than simple random sampling.

---

# Summary: Beam Search Refinements & Length Normalization

This lesson covers critical adjustments to the basic Beam Search algorithm to handle numerical stability, sentence length bias, and performance trade-offs.

---

## 1. Numerical Stability: Using Log-Probabilities
In practice, multiplying many probabilities (all $< 1$) leads to **numerical underflow**, where the resulting number becomes too small for a computer to store accurately.

* **The Solution:** Instead of maximizing the product of probabilities, we maximize the **sum of their logs**:
    $$\arg \max \sum_{t=1}^{T_y} \log P(y^{\langle t \rangle} \mid x, y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle})$$
* Because the $log$ function is strictly increasing, maximizing the sum of logs is mathematically equivalent to maximizing the original product.

---

## 2. Length Normalization
Standard Seq2Seq models have a **bias toward short sentences**. This is because every additional word added to a sequence involves multiplying by another probability $< 1$ (or adding a negative log-probability), which naturally decreases the total score.

### The Normalization Formula
To penalize longer sentences less severely, we divide the score by the number of words $T_y$, often raised to a hyperparameter $\alpha$:

$$\text{Score} = \frac{1}{T_y^\alpha} \sum_{t=1}^{T_y} \log P(y^{\langle t \rangle} \mid x, \dots)$$

* **$\alpha = 0$:** No normalization (original biased version).
* **$\alpha = 1$:** Full normalization (averaging the log-probability).
* **$\alpha \approx 0.7$:** A common heuristic "sweet spot" used in many production systems.



---

## 3. Implementation Details

### How to Pick the Best Sentence
1.  Run Beam Search for a fixed number of steps (e.g., 30).
2.  Keep track of all candidate sentences that reached an **<EOS>** token during the process.
3.  Calculate the **normalized log-likelihood objective** for each.
4.  Pick the single sentence with the highest normalized score.

---

## 4. Choosing Beam Width ($B$)
The choice of $B$ is a trade-off between computational cost and accuracy:

| Beam Width ($B$) | Pros | Cons | Usage |
| :--- | :--- | :--- | :--- |
| **Small** (e.g., 1–3) | Very fast, low memory. | Often misses the optimal translation. | Mobile/Real-time apps. |
| **Medium** (e.g., 10–100) | Great balance of speed/quality. | Higher computational cost. | Standard production systems. |
| **Large** (e.g., 1000+) | Highest possible accuracy. | Very slow, high memory. | Research / "Squeezing" performance. |

### Key Insight
The gain from increasing $B$ usually shows **diminishing returns**. Moving from $B=1$ to $B=10$ often gives a huge boost, while moving from $B=1000$ to $B=3000$ may result in only a marginal improvement.

---

## 5. Beam Search vs. Exact Search
* **Exact Search (BFS/DFS):** Guaranteed to find the absolute maximum probability but is computationally impossible due to the exponential search space.
* **Beam Search:** An **approximate search algorithm**. It runs much faster but is not guaranteed to find the absolute maximum.

# Summary: Error Analysis in Beam Search

Beam Search is an approximate (heuristic) search algorithm. When it produces a poor result, it is crucial to determine if the failure lies in the **Search Algorithm** or the **RNN Model**.

---

## 1. The Diagnostic Test
To perform error analysis, take a human-provided "gold" translation ($y^*$) and the machine-generated translation ($\hat{y}$). Compute the probability of both according to your RNN:

1.  **Calculate** $P(y^* \mid x)$
2.  **Calculate** $P(\hat{y} \mid x)$

> **Note:** If you are using length normalization, compare the **normalized log-likelihood scores** instead of raw probabilities.

---

## 2. Attributing the Error

### Case 1: $P(y^* \mid x) > P(\hat{y} \mid x)$
* **Meaning:** The RNN actually preferred the correct human translation, but Beam Search failed to find it.
* **Culprit:** **Beam Search.** * **Solution:** Increasing the beam width ($B$) is likely to help.

### Case 2: $P(y^* \mid x) \leq P(\hat{y} \mid x)$
* **Meaning:** The RNN mistakenly assigned a higher probability to a bad translation. Beam Search did its job correctly by finding the most "likely" (per the model) result.
* **Culprit:** **RNN Model.** * **Solution:** Spend time on model architecture, training data, or regularization.

---

## 3. Systematic Error Analysis
Go through your development set and create a table to track the source of errors.

| Example | $P(y^* \mid x)$ | $P(\hat{y} \mid x)$ | At Fault |
| :--- | :--- | :--- | :--- |
| 1 | $2 \times 10^{-10}$ | $1 \times 10^{-10}$ | **Beam Search** |
| 2 | $1 \times 10^{-10}$ | $5 \times 10^{-10}$ | **RNN** |
| 3 | $0.5 \times 10^{-10}$ | $2 \times 10^{-10}$ | **RNN** |



## 4. Conclusion
* If **Beam Search** is responsible for a large fraction of errors, focus on the search algorithm (increase $B$).
* If the **RNN** is responsible for most errors, focus on the neural network (data, architecture, or hyperparameter tuning).

---

# Summary: The BLEU Score (Bilingual Evaluation Understudy)

In tasks like Machine Translation or Image Captioning, multiple different outputs can be considered "correct." The BLEU score provides a way to evaluate these outputs automatically.

---

## 1. The Core Concept
The **BLEU (Bilingual Evaluation Understudy)** score measures how close a machine-generated translation is to human-provided references.
* It serves as a substitute for human evaluators.
* It produces a single real-number metric (0 to 1) to compare model performance.

---

## 2. Modified N-Gram Precision ($P_n$)
To prevent a model from "cheating" (e.g., outputting "the the the the"), BLEU uses **Clipped Precision**.

### The Logic:
1.  **Count:** Count the number of times an n-gram (word or sequence of words) appears in the machine output.
2.  **Clip:** Cap that count at the maximum number of times that n-gram appears in any **single** reference sentence.
3.  **Calculate:** Divide the sum of clipped counts by the total number of n-grams in the machine output.



---

## 3. Calculating the Final Score
The final BLEU score combines precisions for different n-gram lengths (typically $n=1$ to $4$) and adds a penalty for length.

### The Combined Formula:
$$\text{BLEU} = \text{BP} \times \exp\left(\frac{1}{N} \sum_{n=1}^{N} \log p_n\right)$$

### The Brevity Penalty (BP)
Machine models can get high precision scores by outputting very short, safe sentences. The **Brevity Penalty** reduces the score of translations that are shorter than the human references.
* **$BP = 1$**: If machine output length $>$ human reference length.
* **$BP < 1$**: If machine output length $\leq$ human reference length.

---

## 4. Summary Table

| Feature | Details |
| :--- | :--- |
| **Use Cases** | Machine Translation, Image Captioning. |
| **Avoided In** | Speech Recognition (where there is usually only one "correct" transcript). |
| **Unigrams ($P_1$)** | Measures "coverage" (is the vocabulary correct?). |
| **Bigrams/Trigrams** | Measures "fluency" (do the words flow together correctly?). |
| **Primary Benefit** | Allows for rapid, automated testing of model improvements. |

---
