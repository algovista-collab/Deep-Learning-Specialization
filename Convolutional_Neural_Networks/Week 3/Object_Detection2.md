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

> **Next Step:** NMS handles the problem of multiple boxes for one object. But what if one grid cell contains **two different objects**? Would you like to learn how **Anchor Boxes** allow the network to handle overlapping objects of different shapes?
