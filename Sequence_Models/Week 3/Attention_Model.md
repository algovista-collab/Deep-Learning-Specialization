# The Attention Model Intuition

The Attention Model solves the "bottleneck" problem of standard Encoder-Decoder architectures, allowing neural networks to handle much longer sequences effectively.

---

## 1. The Bottleneck Problem
In a standard Seq2Seq model, the encoder must compress a whole sentence into a single fixed-length vector. 
* **The Limitation:** For long sentences (30-40+ words), the BLEU score drops significantly because the network struggles to memorize everything.
* **The Solution:** Instead of "memorizing," the model "looks" at specific parts of the input sentence as it generates each word, similar to a human translator.



---

## 2. Model Architecture
The Attention Model typically consists of a **Bidirectional RNN** as the encoder and a standard RNN as the decoder.

### The Encoder (Bottom)
* Uses a **Bidirectional RNN** (GRU or LSTM) to compute features for each word $t$.
* The activation $a^{\langle t \rangle}$ contains information from both the past and future of the sentence relative to word $t$.

### The Decoder (Top)
* Instead of a single input from the encoder, the decoder's hidden state $s^{\langle t \rangle}$ receives a **Context Vector ($C$)**.
* **Context Vector ($C$):** A weighted sum of the encoder's activations.
* **Attention Weights ($\alpha^{\langle t, t' \rangle}$):** These weights define how much "attention" the $t$-th output word should pay to the $t'$-th input word.



---

## 3. How Attention "Focuses"
The model dynamically recalculates its focus at every step of the output:
* To generate the 1st English word (**Jane**), the model assigns a high weight $\alpha^{\langle 1, 1 \rangle}$ to the 1st French word (**Jane**).
* To generate the 3rd English word (**Africa**), the model assigns a high weight $\alpha^{\langle 3, 3 \rangle}$ to the 3rd French word (**l'Afrique**).

### Key Intuition:
The Attention Model creates a **local window** of focus. It doesn't need to hold the entire sentence in "memory" at once; it just needs to know where to look in the input features for the specific information it needs next.

---

## 4. Summary Table

| Feature | Standard Seq2Seq | Attention Model |
| :--- | :--- | :--- |
| **Information Flow** | Single fixed-length vector. | Dynamic context vector ($C$). |
| **Sentence Length** | Weakness; score drops on long text. | Strength; maintains score on long text. |
| **Complexity** | Lower (less computation). | Higher (requires computing $\alpha$ weights). |
| **Inspiration** | Memory-based. | Perceptual/Focus-based. |

---

# Summary: Implementing the Attention Model

The Attention Model uses a weighted sum of encoder hidden states to create a dynamic context vector for each step of the decoder.

---

## 1. The Encoder: Bidirectional Features
The encoder is typically a **Bidirectional RNN** (usually an LSTM or GRU). 

* **Features ($a^{\langle t' \rangle}$):** For each time step $t'$ in the input, we concatenate the forward activation $\overrightarrow{a}^{\langle t' \rangle}$ and backward activation $\overleftarrow{a}^{\langle t' \rangle}$.
* **Notation:** $a^{\langle t' \rangle} = (\overrightarrow{a}^{\langle t' \rangle}, \overleftarrow{a}^{\langle t' \rangle})$. This vector represents a "rich" feature set for the word at position $t'$, including context from both its left and right.



---

## 2. The Context Vector ($C^{\langle t \rangle}$)
Instead of a single bottleneck vector, the decoder receives a unique context vector $C$ for every word it generates.

* **Formula:** The context vector is a weighted sum of the encoder activations:
  $$C^{\langle t \rangle} = \sum_{t'} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}$$
* **Attention Weight ($\alpha^{\langle t, t' \rangle}$):** This represents the amount of "attention" the $t$-th output word pays to the $t'$-th input word.
* **Constraint:** All weights for a specific output step must be non-negative and sum to one: $\sum_{t'} \alpha^{\langle t, t' \rangle} = 1$.

---

## 3. Computing Attention ($\alpha$)
How does the model decide where to look? It uses a small, internal neural network.

### The Softmax Step
To ensure the weights sum to 1, we use a softmax function over the intermediate scores $e$:
$$\alpha^{\langle t, t' \rangle} = \frac{\exp(e^{\langle t, t' \rangle})}{\sum_{t'=1}^{T_x} \exp(e^{\langle t, t' \rangle})}$$

### The Alignment Model ($e$)
To compute the score $e^{\langle t, t' \rangle}$, we use a simple **one-hidden-layer neural network**.
* **Inputs:** 1. $s^{\langle t-1 \rangle}$: The hidden state of the decoder from the *previous* step.
    2. $a^{\langle t' \rangle}$: The features of the input word we are currently scoring.
* **Logic:** We don't manually program the rules for translation; we let gradient descent learn the function that maps $(s^{\langle t-1 \rangle}, a^{\langle t' \rangle})$ to an importance score.



---

## 4. Complexity and Applications
* **Computational Cost:** The algorithm has a **quadratic cost** $O(T_x T_y)$. If the input is 50 words and the output is 50 words, there are 2,500 attention parameters to calculate. For most sentences, this is acceptable.
* **Visualizing Attention:** By plotting $\alpha$, we can see a "map" of what the model was thinking. Usually, the diagonal is bright (word-for-word), but it can show complex patterns for language reordering.
* **Beyond Text:** This is also used in **Image Captioning**, where the model "looks" at specific pixels while generating specific words.



---

## 5. Summary Table

| Component | Variable | Purpose |
| :--- | :--- | :--- |
| **Encoder States** | $a^{\langle t' \rangle}$ | Rich features of the input word (Bidirectional). |
| **Decoder State** | $s^{\langle t-1 \rangle}$ | What has been translated so far. |
| **Attention Weight**| $\alpha^{\langle t, t' \rangle}$ | Relative importance of input $t'$ to output $t$. |
| **Context Vector** | $C^{\langle t \rangle}$ | The "summary" of the input focused on the current word. |

---

# Summary: Speech Recognition & CTC

Speech recognition converts audio (acoustic signals) into text. Modern deep learning has shifted this field from hand-engineered phonemes to end-to-end sequence-to-sequence models.

---

## 1. Data Representation & Pre-processing
Audio data is recorded as minuscule changes in air pressure over time.
* **Raw Audio:** A 1D wave plot of air pressure vs. time.
* **Spectrogram:** A common pre-processing step that maps audio to a 2D image where the horizontal axis is **Time**, the vertical axis is **Frequency**, and intensity represents **Energy**. This mimics how the human ear processes sound.



---

## 2. The End-to-End Shift
* **The Old Way:** Linguists used **Phonemes** (hand-engineered basic units of sound like "de", "e", "ku") as intermediate representations.
* **The New Way:** End-to-end deep learning maps the audio clip $x$ directly to the transcript $y$.
* **Data Requirements:** While academic sets use ~300–3,000 hours, high-performance production systems now require **10,000 to 100,000+ hours** of transcribed audio.

---

## 3. Model Architectures
Two primary approaches are used for sequence-to-sequence speech tasks:
1.  **Attention Models:** Similar to machine translation, the model "looks" at specific time frames of the audio to output characters.
2.  **CTC (Connectionist Temporal Classification):** Used when the input length is much larger than the output length.

---

## 4. Understanding CTC (Connectionist Temporal Classification)
In speech, the number of input time steps (e.g., 1,000 samples for 10 seconds) is much larger than the number of output characters (e.g., 19 characters).

### The CTC Mechanism:
The model is allowed to output repeated characters and a special **Blank Character** (denoted as `_`).
* **The Rule:** Collapse repeated characters that are **not** separated by a blank.
* **Example:**
    * **RNN Output:** `ttt__h_eee___ _qqq__`
    * **Collapsed Result:** `the q`



> **Note:** The "blank" character is distinct from a "space" character. Blanks exist only to help the RNN align the timing, whereas spaces are part of the actual text transcript.

---

## 5. Comparison Table

| Feature | Attention Model | CTC Model |
| :--- | :--- | :--- |
| **Input/Output** | $T_x$ and $T_y$ can be very different. | Usually requires $T_x = T_y$ at the RNN layer. |
| **Logic** | Weighs importance of audio frames. | Collapses redundant outputs using blanks. |
| **Use Case** | General Seq2Seq. | High-speed, real-time speech systems (e.g., Baidu DeepSpeech). |

---

# Summary: Trigger Word Detection

Trigger word detection (also known as keyword spotting) is the technology that allows devices like Amazon Echo ("Alexa"), Google Home ("OK Google"), and Apple Siri ("Hey Siri") to wake up upon hearing a specific phrase.

---

## 1. System Overview
The system takes a continuous stream of audio as input and constantly monitors it for the target phrase.
* **Input (X):** An audio clip, typically pre-processed into **spectrogram features** ($x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \dots$).
* **Model:** Usually an **RNN, GRU, or LSTM** that processes these features sequentially.
* **Output (Y):** A probability that the trigger word has just been completed.



---

## 2. Defining Target Labels ($y$)
A crucial part of building this system is how you label the training data.

* **Ideal Labeling:** Set $y = 0$ for all time steps, and $y = 1$ at the exact moment the trigger word finishes.
* **The Challenge:** This creates a **highly imbalanced dataset** (thousands of 0s for every single 1), making it difficult for the neural network to train effectively.
* **The Heuristic (Hack):** Instead of a single time step of $1$, we set the output to $1$ for a **fixed duration** (several time steps) immediately following the trigger word. This slightly rebalances the ratio of 1s to 0s and makes the signal easier for the RNN to pick up.



---

## 3. Deployment Logic
In a production environment:
1. The device stays in a low-power mode, running a small version of this model.
2. When the RNN outputs a value above a certain **threshold** (e.g., $0.5$), the system "wakes up" and begins full speech recognition processing.

---

## 4. Course Wrap-up
This concludes the primary technical content for Sequence Models. Over the past few weeks, we have covered:
* **Recurrent Neural Networks (RNNs):** Including GRUs and LSTMs for handling long-term dependencies.
* **Word Embeddings:** Learning dense vector representations of language (Word2Vec, GloVe).
* **Attention Mechanism:** Allowing models to focus on specific parts of an input sequence.
* **Audio Applications:** Speech recognition, CTC cost functions, and trigger word systems.

---
