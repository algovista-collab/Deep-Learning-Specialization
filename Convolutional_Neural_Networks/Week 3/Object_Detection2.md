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

$$y = \begin{bmatrix} \text{Anchor Box 1 parameters} \\ \hline \text{Anchor Box 2 parameters} \end{bmatrix} = \begin{bmatrix} p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \\ \hline p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \end{bmatrix}$$

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
