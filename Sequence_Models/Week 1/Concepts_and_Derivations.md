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

# Sampling Novel Sequences

After training a language model, you can "sample" from it to generate new sequences. This allows the model to create original text, music, or other data based on the patterns it learned during training.

---

## 🎲 The Sampling Process

To generate a sequence, the model predicts one step at a time, but instead of using a fixed target, it uses its own previous predictions as inputs for the next step.

<img width="1042" height="573" alt="Screenshot 2026-01-20 082453" src="https://github.com/user-attachments/assets/9c335c2f-745c-41f3-84e2-1ae8dbe4a0a1" />

### Step-by-Step Procedure:
1. **Initial Step:** Input $x^{<1>} = \vec{0}$ and $a^{<0>} = \vec{0}$.
2. **First Word:** The Softmax layer outputs a probability distribution. Use `np.random.choice` to sample the first word ($\hat{y}^{<1>}$) from this distribution.
3. **Looping:** * Take the word you just sampled ($\hat{y}^{<1>}$).
    * Pass it as the input for the next time step ($x^{<2>} = \hat{y}^{<1>}$).
    * Generate and sample the next word ($\hat{y}^{<2>}$).
4. **Termination:** Keep going until the model generates an `<EOS>` (End of Sentence) token or until you reach a predefined number of time steps (e.g., 50 words).

---

## 🔠 Word-Level vs. Character-Level Models

While we have focused on word-level models, you can also train an RNN at the character level.

| Feature | Word-Level RNN | Character-Level RNN |
| :--- | :--- | :--- |
| **Vocabulary ($V$)** | Common words (e.g., "apple", "cat") | Individual chars (e.g., "a", "b", "!", " ") |
| **Vocabulary Size** | Large (10,000 – 100,000+) | Small (~26–100 characters) |
| **Unknown Words** | Uses `<UNK>` for rare words | No `<UNK>`; can construct any word |
| **Sequence Length** | Shorter (fewer time steps) | Much longer (many more time steps) |
| **Complexity** | Efficient and easy to train | Computationally expensive |
| **Dependency** | Captures long-range context better | Struggles with long-range context |

<img width="961" height="548" alt="Screenshot 2026-01-20 082640" src="https://github.com/user-attachments/assets/db2ccce6-7f21-497a-9380-25746080dc44" />

---

## 🎭 Examples of Generated Text
When you sample from a trained model, the output reflects the style of the training **corpus**:
* **News-trained:** Generates text that mimics journalistic style (e.g., "The epidemic to be examined...").
* **Shakespeare-trained:** Generates poetic, archaic-sounding text (e.g., "The mortal moon hath her eclipse...").

---

## ⚠️ Challenges in RNNs
Standard RNNs suffer from **Vanishing Gradients**, making it hard for them to remember information from the beginning of a long sentence by the time they reach the end.

---

# The Vanishing Gradient Problem in RNNs

While RNNs are theoretically capable of handling long sequences, they often struggle with **long-term dependencies** in practice due to vanishing gradients.

---

## 🧠 The Problem: Long-Term Dependencies

In languages like English, a word appearing very early in a sentence can dictate a word much later.

* **Example:** * "The **cat**, which ate... [many words] ... **was** full." (Singular)
    * "The **cats**, which ate... [many words] ... **were** full." (Plural)

A standard RNN often forgets whether the subject was singular or plural by the time it reaches the verb because the "signal" from the beginning of the sentence washes out.

---

## 📉 Vanishing vs. Exploding Gradients

An RNN processing a sequence of 1,000 steps is essentially a **1,000-layer deep neural network**. This leads to two major issues during backpropagation:



### 1. Vanishing Gradients (The Hard Problem)
* **What happens:** As the gradient is multiplied by weight matrices repeatedly through many time steps, it can decrease exponentially.
* **The Result:** The model’s weights are not updated effectively to account for information from the distant past. It becomes "biased" toward local influences (nearby words).
* **Difficulty:** This is hard to solve and requires architectural changes like GRUs or LSTMs.

### 2. Exploding Gradients (The Catastrophic Problem)
* **What happens:** Gradients increase exponentially, leading to extremely large weight updates.
* **The Result:** Parameters "blow up," often resulting in `NaN` (Not a Number) errors in your code.
* **The Fix: Gradient Clipping.** If the gradient vector exceeds a certain threshold, you rescale it (clip it) to a maximum value to keep the training stable.

---

## 🛠️ Comparison Summary

| Issue | Detection | Common Solution |
| :--- | :--- | :--- |
| **Exploding Gradients** | `NaN` values, sudden spikes in loss. | **Gradient Clipping** (simple & effective). |
| **Vanishing Gradients** | Model fails to learn long-range patterns. | **Gated Units (GRU/LSTM)** (requires new architecture). |

---

# Gated Recurrent Unit (GRU)

The GRU is a modification to the standard RNN hidden layer that allows the model to selectively remember or forget information. This "memory" capability is what enables the network to handle long-range dependencies, like connecting a subject at the start of a sentence to a verb at the end.

---

## 🧠 The Concept of "The Cell" ($c^{<t>}$)

In a GRU, we introduce a **memory cell** ($c^{<t>}$). 
* For a GRU, $c^{<t>} = a^{<t>}$ (the cell value is the same as the output activation).
* The cell acts as a "storage" for important information (e.g., whether the subject of a sentence is singular or plural).

<img width="699" height="433" alt="Screenshot 2026-01-20 082754" src="https://github.com/user-attachments/assets/ad944b4d-d495-4310-aa86-f1a27734198b" />

---

## 🛠️ The GRU Equations (Simplified)

To decide how to update the memory cell, the GRU uses a **Candidate Value** and an **Update Gate**.

1.  **Candidate Value ($\tilde{c}^{<t>}$):** A potential new value for the cell, calculated using the current input and previous state.
    $$\tilde{c}^{<t>} = \tanh(W_c [c^{<t-1>}, x^{<t>}] + b_c)$$
2.  **Update Gate ($\Gamma_u$):** A value between $0$ and $1$ (using a sigmoid function) that decides whether to update the cell.
    $$\Gamma_u = \sigma(W_u [c^{<t-1>}, x^{<t>}] + b_u)$$
3.  **Final Cell Update ($c^{<t>}$):** This "gates" the information. If $\Gamma_u \approx 1$, we use the candidate. If $\Gamma_u \approx 0$, we keep the old value.
    $$c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + (1 - \Gamma_u) * c^{<t-1>}$$

---

## 🛠️ The Full GRU Unit

The full version of the GRU includes an additional gate called the **Relevance Gate** ($\Gamma_r$), which determines how much of the *previous* state is relevant to calculating the *next* candidate.

### Complete Formulas:
* **Update Gate:** $\Gamma_u = \sigma(W_u [c^{<t-1>}, x^{<t>}] + b_u)$
* **Relevance Gate:** $\Gamma_r = \sigma(W_r [c^{<t-1>}, x^{<t>}] + b_r)$
* **Candidate:** $\tilde{c}^{<t>} = \tanh(W_c [\Gamma_r * c^{<t-1>}, x^{<t>}] + b_c)$
* **Cell State:** $c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + (1 - \Gamma_u) * c^{<t-1>}$

---

## 🌟 Why this solves Vanishing Gradients
Because the update gate $\Gamma_u$ can be **very close to 0**, the network can choose to skip the update entirely ($c^{<t>} \approx c^{<t-1>}$). This creates a "shortcut" for the gradient to flow backwards through time without being multiplied by small numbers, effectively preserving the signal across hundreds of steps.

---

# Long Short-Term Memory (LSTM)

The LSTM is a more powerful and flexible evolution of the RNN, predating the GRU. It is specifically designed to solve the vanishing gradient problem by allowing the network to maintain a "cell state" that can carry information unchanged across many time steps.

---

## 🧠 Key Differences from GRU
Unlike the GRU, where the activation $a^{<t>}$ is the same as the cell state $c^{<t>}$, the LSTM treats them as two separate quantities. It also uses **three gates** instead of two, providing finer control over the memory.

<img width="864" height="482" alt="Screenshot 2026-01-20 082816" src="https://github.com/user-attachments/assets/57f6d121-fa49-40ea-ac1f-629879c561e0" />

---

## 🧪 The LSTM Equations

The behavior of an LSTM at time $t$ is governed by these six equations:

### 1. The Candidate Value
Computes a potential new value to be added to the cell state.
$$\tilde{c}^{<t>} = \tanh(W_c [a^{<t-1>}, x^{<t>}] + b_c)$$

### 2. The Three Gates
All gates use a **sigmoid** ($\sigma$) function, resulting in values between 0 and 1.
* **Update Gate ($\Gamma_u$):** Decides what new information to store.
* **Forget Gate ($\Gamma_f$):** Decides what old information to discard.
* **Output Gate ($\Gamma_o$):** Decides what part of the cell state to output as the activation.

$$\Gamma_u = \sigma(W_u [a^{<t-1>}, x^{<t>}] + b_u)$$
$$\Gamma_f = \sigma(W_f [a^{<t-1>}, x^{<t>}] + b_f)$$
$$\Gamma_o = \sigma(W_o [a^{<t-1>}, x^{<t>}] + b_o)$$

### 3. The State Updates
* **Cell State ($c^{<t>}$):** Combines the old memory (multiplied by the forget gate) and the new candidate (multiplied by the update gate).
  $$c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + \Gamma_f * c^{<t-1>}$$
* **Activation ($a^{<t>}$):** The final output of the unit.
  $$a^{<t>} = \Gamma_o * \tanh(c^{<t>})$$

---

## 🛠️ Important Variations

### Peephole Connections
A common variation where the gate layers ($\Gamma_u, \Gamma_f, \Gamma_o$) don't just look at $a^{<t-1>}$ and $x^{<t>}$, but also "peek" at the previous cell state $c^{<t-1>}$.
* Usually, this is an **element-wise** relationship (the $i$-th component of the cell state affects only the $i$-th component of the gates).

---

## 📊 GRU vs. LSTM: Which one to use?

| Feature | GRU (Gated Recurrent Unit) | LSTM (Long Short-Term Memory) |
| :--- | :--- | :--- |
| **Complexity** | Simpler (2 gates) | More complex (3 gates) |
| **Speed** | Faster to compute | Slower due to more parameters |
| **Scaling** | Easier to build very large networks | Historically more robust |
| **Performance** | Often matches LSTM performance | More flexible; the "proven" default |

### Summary Advice:
* **LSTM** is the historically proven "gold standard." Use it as your default first choice.
* **GRU** is gaining momentum because it is computationally cheaper and can often scale better to massive models while performing just as well.

---

## 🗝️ Why it Works
The "conveyor belt" of the cell state $c$ allows information to flow through the top of the unit with only minor linear interactions. This prevents the gradients from vanishing exponentially, allowing a subject at time $t=1$ to strongly influence a verb at time $t=100$.

---

# Bidirectional RNN (BRNN)

In standard RNNs (including GRUs and LSTMs), the prediction $\hat{y}^{<t>}$ only depends on the current input $x^{<t>}$ and information from the **past** ($x^{<1>}, \dots, x^{<t-1>}$). This is a problem when the "future" context is needed to understand the present.

---

## 🧐 Motivation: The "Teddy" Problem
Consider these two sentences:
1. "He said **Teddy** Roosevelt was a great president."
2. "He said **Teddy** bears are on sale."

If you only look at the first three words ("He said Teddy"), it is impossible to know if "Teddy" is a person's name ($y=1$) or a toy ($y=0$). You need the words "Roosevelt" or "bears" (the future) to decide.

---

## 🏗️ How it Works
A Bidirectional RNN consists of two independent hidden layers that process the sequence in opposite directions:
1. **Forward Pass ($\vec{a}$):** Processes the sequence from $t=1$ to $t=T_x$.
2. **Backward Pass ($\overleftarrow{a}$):** Processes the sequence from $t=T_x$ down to $t=1$.



### The Prediction Equation
At any time step $t$, the output $\hat{y}^{<t>}$ is calculated by combining the information from both directions:
$$\hat{y}^{<t>} = g(W_y [\vec{a}^{<t>}, \overleftarrow{a}^{<t>}] + b_y)$$

This ensures that the model has access to the **entire sequence** when making a prediction for any specific word.

---

## ⚙️ Key Features and blocks
* **Universal Compatibility:** The "cells" used in a BRNN can be standard RNN units, GRUs, or LSTMs.
* **The "Gold Standard":** For most complex NLP tasks, a **Bidirectional LSTM (Bi-LSTM)** is the most common and effective starting point.

---

## ⚖️ Pros and Cons

| Advantages | Disadvantages |
| :--- | :--- |
| Can capture context from both the past and the future. | **Latency:** You must wait for the *entire* sequence to finish before making the first prediction. |
| Significantly improves accuracy in tasks like Named Entity Recognition (NER). | **Not for real-time:** Harder to use in live speech recognition where you need immediate output as someone talks. |

---

## 🗝️ Summary
While a standard RNN/GRU/LSTM is a **unidirectional** "forward-only" model, the Bidirectional RNN adds a second "backward" layer. This provides the network with a complete "global" view of the sequence before it labels any individual part.

---

<img width="1053" height="590" alt="Screenshot 2026-01-20 085052" src="https://github.com/user-attachments/assets/22f21a98-d744-4aec-a70f-c57c8cba578e" />

# Deep RNNs

While a standard RNN is "deep" in the temporal dimension (across time steps), we can also make them "deep" in the vertical dimension by stacking multiple layers of recurrent units on top of one another.

---

## 🏗️ Architecture and Notation

In a Deep RNN, the activation of a hidden layer at a specific time step serves as the input for the layer directly above it.

* **Notation:** $a^{[l]<t>}$
    * $[l]$ denotes the **layer number**.
    * $<t>$ denotes the **time step**.



### The Calculation
For example, to compute the activation for the 2nd layer at the 3rd time step ($a^{[2]<3>}$), the model looks at two inputs:
1. The activation from the same layer at the previous time step ($a^{[2]<2>}$).
2. The activation from the layer below at the current time step ($a^{[1]<3>}$).

**Formula:**
$$a^{[l]<t>} = g(W_a^{[l]} [a^{[l]<t-1>}, a^{[l-1]<t>}] + b_a^{[l]})$$

---

## 📏 Depth in RNNs vs. CNNs

Unlike Convolutional Neural Networks (CNNs), which might have 100+ layers, Deep RNNs typically only have a few layers (e.g., **3 layers**).

* **Temporal Complexity:** Because an RNN is already unrolled across hundreds or thousands of time steps, it is computationally expensive. Stacking too many layers vertically makes the model extremely difficult to train.
* **Hybrid Approaches:** Often, researchers use a small number of recurrent layers (with horizontal connections) and then stack "standard" dense deep layers (without horizontal connections) on top of the final recurrent layer to make the prediction $\hat{y}$.

---

## 🛠️ Key Variations
* **Unit Types:** The blocks in a Deep RNN can be basic RNN units, **GRUs**, or **LSTMs**.
* **Bidirectional:** You can also create **Deep Bidirectional RNNs**, where each vertical layer contains both a forward and a backward pass.

---

## 🗝️ Summary of the RNN Toolbox
With the conclusion of these topics, you now have the complete set of tools for sequence modeling:
1. **Basic RNN:** For simple sequences.
2. **GRU/LSTM:** To solve the vanishing gradient problem and handle long-term dependencies.
3. **Bidirectional RNN:** To incorporate both past and future context.
4. **Deep RNN:** To learn complex, high-level features of a sequence.

---

<img width="1032" height="582" alt="Screenshot 2026-01-20 090528" src="https://github.com/user-attachments/assets/2747d0d7-92c5-491e-a9b4-50a7d1de5600" />

