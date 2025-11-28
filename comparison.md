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
| **mAP@50** | **54.9%** | 48.9% | *Running* | 54.1% |
| **Precision** | **58.5%** | 55.1% | *Running* | 57.5% |
| **Recall** | **51.8%** | 45.8% | *Running* | 50.5% |
| **mAP@50-95** | **31.1%** | 27.7% | *Running* | 30.8% |

## 3. Observations
*   **YOLOv8**: Currently the best performing model (55% mAP). Fast and stable.
*   **YOLOv10**: Performed worse than v8 (49% mAP) on this dataset.
*   **YOLOv5**: Training in progress. Known for high stability and speed.
*   **YOLO11**: Very close to v8 in initial tests.

## 4. Conclusion
**YOLOv8** remains the recommended model, pending final YOLOv5 results.

