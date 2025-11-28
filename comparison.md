# Model Comparison: YOLOv8 vs YOLO11

This document compares the performance of YOLOv8 and YOLO11 on the Fire Detection dataset.

## 1. Training Configuration
*   **Dataset**: Combined Indoor (Zenodo) + Outdoor (FlameVision)
*   **Image Size**: 640x640
*   **Batch Size**: 16
*   **Device**: GPU (NVIDIA)

## 2. Results Comparison

| Metric | YOLOv8 (30 Epochs) | YOLOv10 (30 Epochs) | YOLOv5 (30 Epochs) | YOLO11 (1 Epoch Test) |
| :--- | :--- | :--- | :--- | :--- |
| **mAP@50** | **54.9%** | 48.9% | 53.3% | 54.1% |
| **Precision** | 58.5% | 55.1% | **59.9%** | 57.5% |
| **Recall** | **51.8%** | 45.8% | 48.1% | 50.5% |
| **mAP@50-95** | **31.1%** | 27.7% | 30.0% | 30.8% |

## 3. Observations
*   **YOLOv8**: The best performing model overall (55% mAP). Best balance of Precision and Recall.
*   **YOLOv5**: Very competitive (53% mAP) and achieved the highest Precision (59.9%), meaning fewer false alarms.
*   **YOLOv10**: Performed worse than v8 (49% mAP) on this dataset.
*   **YOLO11**: Very close to v8 in initial tests.

## 4. Conclusion
**YOLOv8** is the recommended model for deployment due to its superior Recall and mAP. However, **YOLOv5** is a strong alternative if minimizing false positives is the priority.

