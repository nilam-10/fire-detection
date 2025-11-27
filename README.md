# Fire Detection System (YOLOv8)

This project implements a real-time fire detection system using **YOLOv8** (You Only Look Once) and **PyTorch**. It is designed to detect fire in both indoor and outdoor (forest/aerial) environments using a combined dataset.

## 📂 Project Structure

- `data/`: Contains the raw and processed datasets (excluded from git).
- `prepare_data.py`: Script to extract, convert, and merge datasets into YOLO format.
- `train_yolo.py`: Script to train the YOLOv8 model.
- `detect_yolo.py`: Real-time inference on local video/webcam.
- `detect_yolo_youtube.py`: Inference on YouTube videos.
- `config.yaml`: Configuration for data paths and parameters.

## 🚀 Setup

1.  **Install Dependencies**:
    ```bash
    pip install ultralytics torch torchvision opencv-python pyyaml yt-dlp
    ```

2.  **Prepare Datasets**:
    *   Place your dataset zip files in `data/`.
    *   Update `config.yaml` with the correct paths to your zip files.
    *   Run the data preparation script:
        ```bash
        python prepare_data.py
        ```
    *   This will create a `data/yolo_dataset` folder with unified images and labels.

## 🏋️ Training

To train the YOLOv8 model on the combined dataset:

```bash
python train_yolo.py
```

*   **Model**: YOLOv8 Nano (`yolov8n.pt`) - optimized for speed.
*   **Epochs**: 30 (default).
*   **Output**: The best model will be saved to `runs/detect/yolo_fire_det/weights/best.pt`.

## 🔍 Inference (Detection)

### 1. Local Video or Webcam
```bash
# For Webcam
python detect_yolo.py 0

# For Video File
python detect_yolo.py "path/to/video.mp4"
```

### 2. YouTube Video
```bash
python detect_yolo_youtube.py "https://youtube.com/shorts/..."
```
The output video with bounding boxes will be saved in `output_videos/`.

## 📊 Results
The model achieves approximately **55% mAP@50** on the challenging combined dataset, capable of detecting fire in various lighting conditions and environments.

## 📝 Notes
*   The `data/` folder is git-ignored because it contains large image datasets (>1GB).
*   The `runs/` folder containing trained weights is also ignored. You must re-train the model locally to generate the weights.
