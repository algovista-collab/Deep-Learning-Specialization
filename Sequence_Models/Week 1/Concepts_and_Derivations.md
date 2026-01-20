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

## 🔜 Next Steps
The next logical step is defining the **mathematical notation** used to represent these sequences:
- $x^{<t>}$: Representing the $t$-th element of input sequence $X$.
- $T_x$: The total length of the input sequence.
