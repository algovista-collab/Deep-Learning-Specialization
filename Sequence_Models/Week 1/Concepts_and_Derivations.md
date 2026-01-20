# Introduction to Sequence Models

Sequence models, particularly **Recurrent Neural Networks (RNNs)**, are designed to handle data where the order of information matters. These models are used across various domains where inputs ($X$) and outputs ($Y$) can be sequences of varying lengths.

---

## 🚀 Key Applications

| Application | Input ($X$) | Output ($Y$) | Sequence Relationship |
| :--- | :--- | :--- | :--- |
| **Speech Recognition** | Audio Clip | Text Transcript | Many-to-Many |
| **Music Generation** | Integer/Nothing | Music Sequence | One-to-Many |
| **Sentiment Analysis** | Text Sentence | Star Rating | Many-to-One |
| **DNA Analysis** | DNA Sequence | Protein Labels | Many-to-Many |
| **Machine Translation** | English Text | French Text | Many-to-Many (Diff. Length) |
| **Video Recognition** | Video Frames | Activity Type | Many-to-One |
| **NER** | Sentence | Entity Labels | Many-to-Many (Same Length) |

---

## 🛠 Model Flexibility

Sequence models are highly versatile because they address different structural requirements:

* **Variable Lengths:** Unlike standard Feedforward Neural Networks, sequence models can handle $X$ and $Y$ having different lengths (e.g., a 10-word English sentence translating to a 12-word French sentence).
* **Temporal Dependency:** They process data points (audio, words, frames) in a way that preserves the context of what happened in previous time steps.
* **Supervised Learning:** These problems are solved using labeled datasets where $(X, Y)$ pairs are provided to the network for training.

---

# Sequence Models: Notation and Representation

This section introduces the mathematical notation for sequence data and how text is transformed into a format that a neural network can process.

---

## 🔢 Mathematical Notation

Using the example sentence: *"Harry Potter and Hermione Granger invented a new spell."*

### Sequence Indices
* **$x^{<t>}$**: The $t$-th element in the input sequence (e.g., $x^{<1>}$ is "Harry").
* **$y^{<t>}$**: The $t$-th element in the output sequence (e.g., $y^{<1>}$ is $1$ if it's a person's name, $0$ otherwise).
* **$T_x$**: Total length of the input sequence ($T_x = 9$ in this example).
* **$T_y$**: Total length of the output sequence.

### Training Set Notation
For multiple training examples $i$:
* **$x^{(i)<t>}$**: The $t$-th element of the $i$-th training example.
* **$T_x^{(i)}$**: The length of the input sequence for the $i$-th training example (lengths can vary across examples).

---

## 🔤 Word Representation

To feed text into a model, words must be converted into numbers. This is done via a **Vocabulary** (or Dictionary).

### 1. The Vocabulary (V)
A list of all unique words the model "knows." 
* **Size:** Common sizes range from 30,000 to 100,000 words (denoted as $|V|$).
* **Selection:** Usually chosen by taking the most frequent words in the training set.

### 2. One-Hot Vectors
Each word is represented as a high-dimensional vector of size $|V|$.
* The vector contains all **0s**, except for a **1** at the index corresponding to that word in the dictionary.
* **Example:** If "Harry" is the 4,075th word in the dictionary, $x^{<1>}$ is a 10,000-node vector with a `1` at position 4075.



### 3. Handling Unknown Words
If the model encounters a word not in its dictionary, it uses a special token:
* **`<UNK>`**: Represents "Unknown Word" to ensure the model can still process the sequence.

---

## 🎯 Task Goal: Named Entity Recognition (NER)
The objective is to learn a mapping $X \rightarrow Y$ where:
* **Input ($X$):** A sequence of one-hot vectors.
* **Output ($Y$):** A sequence of labels indicating if a word is a specific entity (e.g., a Person).

---

# Building a Recurrent Neural Network (RNN)

Standard neural networks (Vanilla NNs) are poorly suited for sequence tasks like Named Entity Recognition (NER) for two main reasons:
1. **Variable Lengths:** Inputs and outputs can have different lengths in different examples.
2. **No Feature Sharing:** A standard NN doesn't share features learned at one position (e.g., "Harry" at the start) with other positions.

---

## 🏗️ RNN Architecture
An RNN processes a sequence one step at a time, maintaining a "hidden state" (activation) that carries information from previous time steps to the current one.

<img width="1025" height="553" alt="image" src="https://github.com/user-attachments/assets/9b319c06-5aed-4627-af27-d14d1c019049" />

### Key Components
* **$x^{<t>}$**: Input at time step $t$.
* **$a^{<t>}$**: The hidden state (activation) at time $t$. It is computed using $x^{<t>}$ and the previous state $a^{<t-1>}$.
* **$y^{<t>}$**: The prediction at time $t$.
* **$a^{<0>}$**: The initial state, typically a vector of zeros.

### Parameter Sharing
Unlike standard NNs, an RNN uses the **same parameters** at every time step:
* **$W_{ax}$**: Weights for the input-to-hidden connection.
* **$W_{aa}$**: Weights for the hidden-to-hidden connection.
* **$W_{ya}$**: Weights for the hidden-to-output connection.

---

## 🧪 Forward Propagation Equations

For each time step $t$:
1.  **Hidden State:** $a^{<t>} = g_1(W_{aa} a^{<t-1>} + W_{ax} x^{<t>} + b_a)$
    * *Note: $g_1$ is usually `tanh` or `ReLU`.*
2.  **Output:** $\hat{y}^{<t>} = g_2(W_{ya} a^{<t>} + b_y)$
    * *Note: $g_2$ is usually `sigmoid` (binary) or `softmax`.*

### Simplified Notation
To make the math cleaner, we can stack the weight matrices into a single matrix $W_a$:
$$a^{<t>} = g(W_a [a^{<t-1>}, x^{<t>}] + b_a)$$
Where:
* $W_a$ is $[W_{aa} | W_{ax}]$ (horizontal stacking).
* $[a^{<t-1>}, x^{<t>}]$ is the vertical concatenation of the two vectors.

---

## ⚠️ Limitations of Basic RNNs
The standard RNN described here is **unidirectional**. 
* **The Problem:** It only uses information from the *past* to make a prediction. 
* **Example:** In "Teddy Roosevelt was...", the model needs to see the word "Roosevelt" (future) to know if "Teddy" is a person or a toy.
* **The Fix:** This is addressed later using **Bidirectional RNNs (BRNNs)**.

---

# Backpropagation Through Time (BPTT)

In Recurrent Neural Networks, backpropagation is referred to as **Backpropagation Through Time** because the gradient calculations move backward from the final time step to the beginning of the sequence.

---

## 🔁 The Computation Graph

To understand backprop, we first look at the flow of the forward pass and the resulting loss calculation:

1. **Forward Pass:** Information flows from left to right ($x^{<1>} \to a^{<1>} \to a^{<2>} \dots$).
2. **Parameters:** The same parameters ($W_a, b_a, W_y, b_y$) are reused at every time step.
3. **Loss Calculation:** A loss is calculated at every time step $t$.



### 1. Element-wise Loss
For a single time step $t$, we use the **Cross-Entropy Loss** (standard for binary classification/NER):
$$\mathcal{L}^{<t>}(\hat{y}^{<t>}, y^{<t>}) = -y^{<t>} \log \hat{y}^{<t>} - (1 - y^{<t>}) \log (1 - \hat{y}^{<t>})$$

### 2. Global Loss
The overall loss for the entire sequence is the sum of the losses at each time step:
$$\mathcal{L} = \sum_{t=1}^{T_y} \mathcal{L}^{<t>}(\hat{y}^{<t>}, y^{<t>})$$

---

## ⬅️ The Backward Pass (BPTT)

The "Backpropagation Through Time" process involves following the computation graph in the opposite direction of the arrows:

* **Step 1:** Calculate the derivative of the global loss $\mathcal{L}$ with respect to the outputs $\hat{y}^{<t>}$.
* **Step 2:** Flow the gradients backward through the network to compute derivatives for the hidden states $a^{<t>}$.
* **Step 3:** Because parameters $W$ and $b$ are shared across all time steps, their total gradient is the sum of the gradients calculated at each individual time step.

### Why the name?
It is called "Through Time" because you are scanning from the end of the sequence ($T_x$) back to the beginning ($t=1$), effectively reversing the temporal flow of the forward pass.

---

# RNN Architecture Variations

Recurrent Neural Networks are highly flexible. Depending on the application, the length of the input sequence ($T_x$) does not have to match the length of the output sequence ($T_y$).

---

## 📊 Summary of RNN Types

| Type | Input ($T_x$) | Output ($T_y$) | Example Application |
| :--- | :--- | :--- | :--- |
| **One-to-One** | 1 | 1 | Standard Neural Network (not typically an RNN) |
| **One-to-Many** | 1 | >1 | Music Generation, Image Captioning |
| **Many-to-One** | >1 | 1 | Sentiment Classification (Rating a movie review) |
| **Many-to-Many ($T_x = T_y$)** | >1 | >1 | Named Entity Recognition (NER) |
| **Many-to-Many ($T_x \neq T_y$)** | >1 | >1 | Machine Translation (Encoder-Decoder) |

---

## 🛠️ Architecture Deep Dive

<img width="1840" height="1001" alt="Screenshot 2026-01-20 080231" src="https://github.com/user-attachments/assets/14e3303c-e21e-486b-94a0-188d3bc7aae9" />

### 1. Many-to-One (Sentiment Classification)
The model processes every word in a sentence sequentially, but only produces a prediction after the final word has been "read."
* **Flow:** $x^{<1>}, x^{<2>}, \dots, x^{<T_x>} \to \text{Hidden States} \to \hat{y}$

### 2. One-to-Many (Music Generation)
The model takes a single input (like a genre ID or a starting note) and generates a sequence of outputs.
* **Flow:** $x \to \text{RNN Cell} \to \hat{y}^{<1>}, \hat{y}^{<2>}, \dots$
* **Note:** Often, the output $\hat{y}^{<t>}$ is fed back into the next time step as the input.

### 3. Many-to-Many (Variable Length / Machine Translation)
Since a French sentence and its English translation may have different word counts, a "Syncronous" RNN won't work. Instead, we use an **Encoder-Decoder** architecture:
* **Encoder:** Reads the entire input sequence and compresses it into a vector.
* **Decoder:** Takes that vector and generates the output sequence in the target language.

---

## 🗝️ Core Concepts
* **Andrej Karpathy's Influence:** This classification is popularized by the blog post *"The Unreasonable Effectiveness of Recurrent Neural Networks."*
* **Parameter Sharing:** In all these variations, the internal weights ($W_{aa}$, $W_{ax}$, etc.) remain shared across all time steps.
* **Flexibility:** By choosing where to place the output $\hat{y}$, you can adapt the RNN to almost any sequential data problem.

---

# RNN Language Modeling

A language model is a system that estimates the probability of a sentence or a sequence of words. This is a foundational task for Speech Recognition, Machine Translation, and Content Generation.

---

## 🧐 What does a Language Model do?
A language model calculates the probability of a specific sequence of words $P(y^{<1>}, y^{<2>}, \dots, y^{<T_y>})$.

* **Example (Speech Recognition):**
    * *Sentence A:* "The apple and **pair** salad was delicious."
    * *Sentence B:* "The apple and **pear** salad was delicious."
* Even if they sound identical, the model assigns a much higher probability to **Sentence B**, allowing the system to output the correct text.

---

## 🛠️ Building a Language Model with RNNs

### 1. Preprocessing (Tokenization)
Before training, you must process a large body of text (**corpus**):
* **Vocabulary:** Create a list of all words (e.g., 10,000 words).
* **`<EOS>` (End of Sentence):** An optional token added to the end of sentences so the model learns when to stop.
* **`<UNK>` (Unknown):** A token used to replace words not found in your vocabulary.

### 2. RNN Architecture for Language Modeling
In a language model, the input at time $t$ ($x^{<t>}$) is actually the **actual word** from the previous time step ($y^{<t-1>}$).

<img width="1045" height="576" alt="image" src="https://github.com/user-attachments/assets/16b79847-c6c5-428c-a67c-35fcd7e5b1eb" />

**The Training Flow:**
1. **Time Step 1:** Input $x^{<1>} = \vec{0}$ (and $a^{<0>} = \vec{0}$). The model predicts $\hat{y}^{<1>}$ via a **Softmax** over the entire vocabulary (what is the first word?).
2. **Time Step 2:** Input $x^{<2>} = y^{<1>}$ (the *actual* first word from the training data). The model predicts the probability of the second word given the first.
3. **Time Step $t$:** Input $x^{<t>} = y^{<t-1>}$. The model predicts $\hat{y}^{<t>}$: the probability of the current word given all preceding words.

---

## 🧪 Loss Function

The model uses a **Softmax loss** at each time step $t$:
$$\mathcal{L}(\hat{y}^{<t>}, y^{<t>}) = -\sum_{i} y_i^{<t>} \log \hat{y}_i^{<t>}$$

The **Total Loss** for the sequence is the sum of losses at every time step:
$$\mathcal{L} = \sum_{t} \mathcal{L}^{<t>}(\hat{y}^{<t>}, y^{<t>})$$

---

## 🔮 How it Predicts a Sentence Probability
For a sentence like "Cats average sleep," the model calculates:
$$P(\text{cats, average, sleep}) = P(\text{cats}) \times P(\text{average} \mid \text{cats}) \times P(\text{sleep} \mid \text{cats, average})$$
Each of these conditional probabilities is provided by one of the RNN's Softmax outputs.

---
