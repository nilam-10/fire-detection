# Model Comparison: YOLOv8 vs YOLO11

This document compares the performance of YOLOv8 and YOLO11 on the Fire Detection dataset.

## 1. Training Configuration
*   **Dataset**: Combined Indoor (Zenodo) + Outdoor (FlameVision)
*   **Image Size**: 640x640
*   **Batch Size**: 16
*   **Device**: GPU (NVIDIA)

## 2. Results Comparison

| Metric | YOLOv8 (30 Epochs) | YOLOv10 (30 Epochs) | RT-DETR (5 Epochs) | YOLO11 (1 Epoch Test) |
| :--- | :--- | :--- | :--- | :--- |
| **mAP@50** | **54.9%** | 48.9% | 44.6% | 54.1% |
| **Precision** | **58.5%** | 55.1% | 50.4% | 57.5% |
| **Recall** | **51.8%** | 45.8% | 45.7% | 50.5% |
| **mAP@50-95** | **31.1%** | 27.7% | 23.8% | 30.8% |

## 3. Observations
*   **YOLOv8**: The clear winner. It achieved the highest accuracy (55% mAP) and is very stable.
*   **YOLOv10**: Surprisingly performed worse than v8 (49% mAP) on this specific dataset, despite being a newer architecture.
*   **RT-DETR**: Showed promise (44% mAP) in just 5 epochs but requires much longer training to converge.
*   **YOLO11**: Initial tests showed it was very close to v8, but v8 edged it out slightly.

## 4. Conclusion
**YOLOv8** is the recommended model for deployment. It offers the best balance of accuracy and speed for this Fire Detection task.

