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
