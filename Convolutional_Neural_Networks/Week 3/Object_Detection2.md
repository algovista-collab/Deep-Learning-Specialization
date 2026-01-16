# Summary: Non-Max Suppression (NMS)

Non-max suppression is a critical post-processing step in object detection. It ensures that your algorithm detects each object only once by "suppressing" redundant, overlapping bounding boxes.

---

## 1. The Problem: Multiple Detections
In a grid-based system like YOLO, multiple grid cells may all believe they have found the center of the same object. This results in several overlapping bounding boxes for a single car or pedestrian.



---

## 2. How Non-Max Suppression Works
NMS cleans up these multiple detections by following a simple logic: it keeps the most confident box and removes others that overlap too much with it.

### The Step-by-Step Process:
1.  **Probability Filtering:** First, discard all boxes with a confidence score ($P_c$) below a certain threshold (e.g., 0.6).
2.  **Pick the Best:** From the remaining boxes, pick the one with the highest $P_c$. This is your official prediction.
3.  **Remove Overlaps:** Look at all other remaining boxes. If any box has a high **Intersection Over Union (IoU)** with the box you just picked, discard it.
4.  **Repeat:** Repeat the process for the remaining boxes until every box has either been selected as a prediction or discarded.



---

## 3. Detailed Algorithm Pseudocode

* **Input:** A list of predicted bounding boxes with their confidence scores.
* **Step 1:** Discard all boxes where $P_c \leq 0.6$.
* **Step 2 (While Loop):** While there are remaining boxes:
    * Pick the box with the highest $P_c$. Output this as a final prediction.
    * Discard any remaining box in the list that has $IoU \geq 0.5$ with the box just picked.

---

## 4. Multi-Class Detection
If you are detecting multiple classes (e.g., Pedestrian, Car, Motorcycle), you should perform Non-Max Suppression **independently for each class**. 
* Run NMS for all "Car" boxes.
* Run NMS for all "Pedestrian" boxes.
* Run NMS for all "Motorcycle" boxes.

---

## 5. Summary Table

| Term | Role in NMS |
| :--- | :--- |
| **$P_c$ (Confidence)** | Used to rank boxes and decide which one to "keep" first. |
| **Thresholding** | Quickly eliminates low-quality background noise. |
| **IoU (Overlap)** | Determines if a box is a "duplicate" of the best box. |
| **Suppression** | The act of discarding non-maximal (less confident) overlapping boxes. |

# Summary: Anchor Boxes

Anchor boxes are a technique used to allow an object detection algorithm to detect **multiple objects within a single grid cell**. It addresses the limitation of the basic YOLO model where each grid cell can only assign one object.

---

## 1. The Problem: Overlapping Midpoints
In a standard grid system, if the midpoints of two different objects (e.g., a pedestrian and a car) fall into the same grid cell, the network is forced to choose only one to detect.



---

## 2. Defining Anchor Boxes
Instead of predicting one set of parameters per cell, you pre-define a set of shapes called **Anchor Boxes**.
* **Specialization:** Anchor Box 1 might be tall and skinny (ideal for pedestrians), while Anchor Box 2 is wide and short (ideal for cars).
* **Assignment:** During training, an object is assigned to a grid cell **AND** a specific anchor box based on which anchor box has the highest **Intersection over Union (IoU)** with the object's ground-truth bounding box.



---

## 3. Changes to the Target Label ($y$)
The output vector $y$ is expanded to include predictions for every anchor box. If you have 2 anchor boxes and 3 classes, the vector doubles in size.

**The Label Vector Structure:**

$$y = \begin{bmatrix} p_c & b_x & b_y & b_h & b_w & c_1 & c_2 & c_3 & p_c & b_x & b_y & b_h & b_w & c_1 & c_2 & c_3 \end{bmatrix}^T$$

* **New Output Volume:** For a $3 \times 3$ grid with 2 anchors, the output becomes $3 \times 3 \times 16$ (or $3 \times 3 \times 2 \times 8$).

---

## 4. Edge Cases and Limitations
While anchor boxes significantly improve detection, some rare scenarios remain difficult:
* **Same Shape:** Two objects of the same shape (same anchor box) in the same cell.
* **Too Many Objects:** More objects in a cell than available anchor boxes (e.g., 3 objects in a cell with only 2 anchors).

*Note: In practice, using a finer grid (like $19 \times 19$) makes these collisions extremely rare.*

---

## 5. Choosing Anchor Box Shapes
* **Manual Selection:** Choosing 5–10 shapes that represent the objects in your dataset (e.g., tall/skinny, short/wide).
* **Automatic Selection (Advanced):** Using the **K-means clustering algorithm** on your training data to automatically find the most representative bounding box shapes (stereotypes) for your specific dataset.

---

## 6. Summary Comparison

| Feature | Without Anchor Boxes | With Anchor Boxes |
| :--- | :--- | :--- |
| **Objects per cell** | Max 1 | Max $N$ (number of anchors) |
| **Assignment** | Midpoint $\to$ Grid Cell | Midpoint $\to$ Cell + Highest IoU Anchor |
| **Output size** | Grid $\times$ Grid $\times$ 8 | Grid $\times$ Grid $\times$ (Anchors $\times$ 8) |
| **Benefit** | Simple | Handles overlap & allows specialization |

# Summary: The Complete YOLO Algorithm

The YOLO (You Only Look Once) algorithm integrates all the concepts discussed—grid systems, bounding box regression, IoU, non-max suppression, and anchor boxes—into a single, high-performance object detection pipeline.

---

## 1. Constructing the Training Set
For a 3x3 grid with two anchor boxes and three classes (Pedestrian, Car, Motorcycle), each grid cell is represented by a 16-dimensional vector.

**The Target Vector ($y$):**
$$y = \begin{bmatrix} p_c & b_x & b_y & b_h & b_w & c_1 & c_2 & c_3 & p_c & b_x & b_y & b_h & b_w & c_1 & c_2 & c_3 \end{bmatrix}^T$$



### Training Data Assignment:
* **Background Cells:** If no object midpoint falls in a cell, $p_c=0$ for both anchor boxes; other values are "don't cares."
* **Object Cells:** The object is assigned to the specific anchor box that has the **highest IoU** with the object's ground-truth shape.
* **Volume:** In practice, a 19x19 grid with 5 anchors would result in a $19 \times 19 \times (5 \times 8) = 19 \times 19 \times 40$ output volume.

---

## 2. Making Predictions
When a test image is fed into the trained ConvNet, it outputs the full volume (e.g., $3 \times 3 \times 16$). 
* Every cell—even those with no objects—will output values. 
* Where $p_c$ is low, the remaining 14 numbers (bounding box and class predictions) are treated as noise and ignored.
* Where $p_c$ is high, the network provides the specific coordinates and the class of the detected object relative to the grid cell.

---

## 3. Post-Processing: Non-Max Suppression (NMS)
Because the network predicts two boxes for every single grid cell (18 boxes for a 3x3 grid), many redundant detections will exist.

1.  **Probability Thresholding:** Discard all boxes where the predicted $p_c$ is below a set limit (e.g., 0.6).
2.  **Class-Independent NMS:** For each class (Pedestrian, Car, Motorcycle):
    * Run the Non-Max Suppression algorithm independently.
    * This ensures that overlapping detections of a "Car" don't suppress a nearby "Pedestrian."
3.  **Final Output:** The remaining boxes are the final localized detections.



---

## 4. Key Takeaways
* **Speed:** YOLO is exceptionally fast because it requires only one forward pass through the ConvNet to see every object in the image.
* **Specialization:** Anchor boxes allow different parts of the output vector to specialize in different shapes (e.g., tall/skinny vs. short/wide).
* **Robustness:** It is one of the most effective algorithms in modern computer vision, combining global context with local precision.

---

## 5. Summary Table: YOLO Pipeline

| Phase | Key Action |
| :--- | :--- |
| **Training** | Assign objects to Grid Cells + Anchor Boxes based on midpoint and IoU. |
| **Inference** | A single forward pass through a ConvNet to generate the output volume. |
| **Cleanup** | Apply $p_c$ thresholding followed by per-class Non-Max Suppression. |
| **Result** | Optimized bounding boxes with high confidence and correct labels. |

# Summary: Region Proposals (R-CNN)

While YOLO handles detection in a single forward pass, the **Region Proposal** family of algorithms (R-CNN) uses a "propose-then-classify" strategy. Although often slower than YOLO, it has been highly influential in computer vision.

---

## 1. The Core Idea: Selective Search
The primary drawback of simple sliding windows is that the algorithm wastes time classifying empty background regions. R-CNN addresses this by:
* Running a **segmentation algorithm** to find "blobs" or regions that look like they *might* be objects.
* Selecting roughly **2,000 regions** (called region proposals).
* Running a ConvNet only on these 2,000 regions rather than every possible window.



---

## 2. The R-CNN Evolution
Over time, several iterations were developed to solve the speed issues of the original algorithm:

| Algorithm | Key Innovation | Characteristics |
| :--- | :--- | :--- |
| **R-CNN** | Propose regions $\to$ classify one by one. | Very slow; uses external segmentation. |
| **Fast R-CNN** | Propose regions $\to$ classify all via **convolutional implementation**. | Much faster than R-CNN; similar to convolutional sliding windows. |
| **Faster R-CNN** | Uses a **Region Proposal Network (RPN)** to suggest regions. | Fully neural network based; faster than Fast R-CNN but usually slower than YOLO. |



---

## 3. How R-CNN Localizes
Like YOLO, R-CNN does not just rely on the initial proposed box.
* The network outputs a class label (Pedestrian, Car, etc.).
* It also outputs **bounding box parameters ($b_x, b_y, b_h, b_w$)** to refine the initial "blob" into a tight, accurate rectangle around the object.

---

## 4. R-CNN vs. YOLO: Philosophical Differences
* **Two-Step (R-CNN):** First find *where* objects might be, then find *what* they are. This is often more accurate but computationally heavier.
* **One-Step (YOLO):** Find *where* and *what* simultaneously in one pass. This is generally faster and more suited for real-time applications.

> **Perspective:** Many researchers (including Andrew Ng) believe that "one-step" frameworks like YOLO are more promising for the long term because of their end-to-end efficiency.

---

## 5. Summary Table

| Feature | R-CNN Family | YOLO |
| :--- | :--- | :--- |
| **Detection Philosophy** | Region Proposals (Selective) | Grid-based (Global) |
| **Speed** | Slow to Moderate | Fast (Real-time) |
| **Accuracy** | Often higher in localization | High (improving rapidly) |
| **Components** | Multiple stages (RPN + Classifier) | Single pipeline |
