# Dataset Documentation

## Overview
This project uses a **combined dataset** specifically curated for Fire Detection. It merges indoor and outdoor fire scenarios to ensure the model generalizes well across different environments.

## Data Sources
1.  **Indoor Dataset**: Sourced from Zenodo (Indoor Fire & Smoke). Focuses on domestic and industrial indoor fire scenarios.
2.  **Outdoor Dataset**: Sourced from FlameVision (Aerial/Forest Fire). Focuses on wildfires and outdoor smoke.

## Dataset Structure
The dataset is organized in the standard YOLO format:

```
data/yolo_dataset/
├── train/
│   ├── images/  (6650 images)
│   └── labels/  (6650 txt files)
├── val/
│   ├── images/  (1650 images)
│   └── labels/  (1650 txt files)
└── test/
    ├── images/  (1200 images)
    └── labels/  (1200 txt files)
```

## Statistics
*   **Total Images**: 9,500
    *   **Indoor Scenarios**: 4,997 images
    *   **Outdoor (Wildfire)**: 4,503 images
*   **Training Set**: 6,650 (70%)
*   **Validation Set**: 1,650 (17%)
*   **Testing Set**: 1,200 (13%)
*   **Duplicate Check**: 0 duplicates found (verified via MD5 hashing).

## Classes
*   **0**: Fire

## Preprocessing
*   All images are resized to **640x640** during training.
*   Labels are normalized to YOLO format (`class x_center y_center width height`).
