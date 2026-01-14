# Machine Learning Strategy: A Comprehensive Guide

## 1. Orthogonalization
Orthogonalization is the strategy of having separate controls for separate problems. A system is easier to tune when you can adjust one knob to fix one specific issue without affecting others.

For a supervised learning system to perform well, you must satisfy four distinct goals in order:
1.  **Fit training set well on cost function:** Aim for Human-Level Performance (HLP).
2.  **Fit dev set well on cost function:** Ensure the model generalizes.
3.  **Fit test set well on cost function:** Ensure the model isn't overfit to the dev set.
4.  **Perform well in the real world:** Ensure the metric/distribution matches user needs.

---

## 2. Setting Up Metrics and Goals
### Single Number Evaluation Metric
Use a single number (like **F1 Score**) rather than multiple metrics (Precision and Recall) to allow for quick comparison between Model A and Model B.

### Optimizing vs. Satisficing Metrics
When you have multiple constraints:
* **Optimizing Metric:** The one you want to maximize (e.g., Accuracy).
* **Satisficing Metric:** Must meet a certain threshold (e.g., Running time $\leq$ 100ms, False Positive Rate $< 1\%$).

### Target Setting
If your evaluation metric or dev set no longer tracks what you actually care about in the real world (e.g., it allows too many "NSFW" images), **change your metric or your dev set.**

---

## 3. Human-Level Performance (HLP)
ML systems typically improve rapidly until they surpass human-level performance, then they slow down as they approach the **Bayes Optimal Error** (the theoretical minimum error possible).



* **Avoidable Bias:** The difference between Training Error and Bayes Error (proxy: HLP).
* **Variance:** The difference between Dev Error and Training Error.

---

## 4. Error Analysis
Before spending months fixing a specific problem, perform a manual audit.

1.  **Manual Examination:** Pick ~100 mislabeled examples from the dev set.
2.  **Categorize:** Create a table with columns for different error types (e.g., Dog pictures, Blurry images, Instagram filters).
3.  **Prioritize:** If "Dog pictures" only account for 5% of errors, fixing that category won't significantly improve your model. Focus on the categories with the highest percentages.

> **Note on Labels:** DL algorithms are robust to **random** errors in training data, but sensitive to **systematic** errors (e.g., consistently labeling white cats as dogs).

---

## 5. Mismatched Data Distributions
In the era of Big Data, we often have a large "General" dataset (Web images) and a small "Target" dataset (Mobile app images).

* **Rule:** Dev and Test sets must come from the **same** distribution (the data you care about in the future).
* **Training-Dev Set:** A subset of the training data distribution not used for training. It helps distinguish between a **variance** problem and a **data mismatch** problem.

### Identifying the Problem:
| Error Source | Comparison |
| :--- | :--- |
| **Avoidable Bias** | Human Level vs. Training Error |
| **Variance** | Training Error vs. Training-Dev Error |
| **Data Mismatch** | Training-Dev Error vs. Dev Error |
| **Overfitting to Dev** | Dev Error vs. Test Error |

---

## 6. Addressing Data Mismatch
1.  Perform error analysis to understand the differences between the Training and Dev/Test sets.
2.  **Artificial Data Synthesis:** Create training data that mimics the dev set (e.g., adding car noise to clean audio for a voice assistant). 
    * *Caution:* Be careful not to synthesize a tiny subset of the possible noise space, or the model may overfit to that specific noise.

---

## 7. End-to-End Deep Learning
This replaces multi-stage pipelines with a single, large neural network.

* **Pros:** Let the data speak (no human bias in feature engineering); simpler production pipeline.
* **Cons:** Requires a **massive** amount of labeled data $(x, y)$; may ignore useful hand-designed components if data is scarce.
* **Use Case:** Excellent for Speech Recognition and Machine Translation; less effective for tasks where the pipeline can be broken into clearer, smaller sub-tasks with limited data.

---
