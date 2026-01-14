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
* **Training-Dev Set:** A subset of the training data distribution not used for training. It helps distinguish between a **variance** problem and a **data mismatch** problem. Addressing the largest performance gap (between human-level and training error) is the most efficient strategy.

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

<img width="1749" height="810" alt="Screenshot 2026-01-12 083451" src="https://github.com/user-attachments/assets/2038f42a-2f9f-4f34-8a4e-b42cf77bd0aa" />

<img width="926" height="465" alt="Screenshot 2026-01-12 121716" src="https://github.com/user-attachments/assets/1510efa0-801c-4aff-b6ef-631ef02e7b15" />

<img width="905" height="503" alt="Screenshot 2026-01-12 124747" src="https://github.com/user-attachments/assets/c4204916-3ffc-48cf-94db-a29ff7301c85" />

<img width="437" height="475" alt="Screenshot 2026-01-12 230710" src="https://github.com/user-attachments/assets/80abab55-f846-4171-a662-ea4a6be7df55" />

<img width="454" height="209" alt="Screenshot 2026-01-12 230720" src="https://github.com/user-attachments/assets/baa04f24-b6db-4bcd-a54f-f7a9c0e7b0cb" />

<img width="842" height="452" alt="Screenshot 2026-01-12 230814" src="https://github.com/user-attachments/assets/e391dc65-b884-42d8-826f-58627c32e31f" />

<img width="842" height="429" alt="Screenshot 2026-01-13 110426" src="https://github.com/user-attachments/assets/b936dd1b-b0a4-4c58-94ab-57b91a7c2c80" />
