# Model Comparison: YOLOv8 vs YOLO11

This document compares the performance of YOLOv8 and YOLO11 on the Fire Detection dataset.

## 1. Training Configuration
*   **Dataset**: Combined Indoor (Zenodo) + Outdoor (FlameVision)
# Model Comparison: YOLOv8 vs YOLO11

This document compares the performance of YOLOv8 and YOLO11 on the Fire Detection dataset.

## 1. Training Configuration
*   **Dataset**: Combined Indoor (Zenodo) + Outdoor (FlameVision)
*   **Image Size**: 640x640
*   **Batch Size**: 16
*   **Device**: GPU (NVIDIA)

## 2. Results Comparison

| Metric | YOLOv8 (50 Epochs) | YOLOv8 (30 Epochs) | YOLOv5 (30 Epochs) | RT-DETR (5 Epochs) | YOLO11 (Initial Test) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **mAP@50** | **56.1%** | 54.9% | 53.3% | 44.6% | 54.1% |
| **Precision** | 58.0% | 58.5% | **59.9%** | 50.4% | 57.5% |
| **Recall** | **53.6%** | 51.8% | 48.1% | 45.7% | 50.5% |
| **mAP@50-95** | **32.0%** | 31.1% | 30.0% | 23.8% | 30.8% |

## 3. Observations
*   **YOLOv8 (50 Epochs)**: The absolute best model. Training for longer improved mAP by 1.2% and Recall by 1.8%.
*   **YOLOv8 (30 Epochs)**: Still excellent, but the extra 20 epochs paid off.
*   **YOLOv5**: High precision but lower recall. Good if you hate false alarms, but misses some fires.
*   **RT-DETR**: Achieved 44.6% mAP in just 5 epochs. Shows promise but requires significantly longer training to converge.
*   **YOLO11**: Initial testing showed it is very competitive with v8, but v8 is currently more optimized for this specific dataset setup.

## 4. Conclusion
**YOLOv8 (50 Epochs)** is the final recommended model for deployment. It offers the highest detection rate (Recall) and overall accuracy (mAP).
 However, **YOLOv5** is a strong alternative if minimizing false positives is the priority.
