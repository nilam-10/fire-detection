# Fire Detection System (YOLOv8)

This project implements a real-time fire detection system using **YOLOv8** (You Only Look Once) and **PyTorch**. It is designed to detect fire in both indoor and outdoor (forest/aerial) environments using a combined dataset.

## 📂 Project Structure

- `data/`: Directory for datasets (excluded from git due to size).
- `extract_data.py`: Script to extract the raw zip files.
- `prepare_data.py`: Script to convert and merge datasets into YOLO format.
- `train_yolo.py`: Script to train the YOLOv8 model.
- `detect_yolo.py`: Real-time inference on local video/webcam.
- `detect_yolo_youtube.py`: Inference on YouTube videos.
- `config.yaml`: Configuration for data paths and parameters.

## 🚀 Setup

1.  **Install Dependencies**:
    ```bash
    pip install ultralytics torch torchvision opencv-python pyyaml yt-dlp
    ```

2.  **Download Datasets**:
    *   **Indoor Fire Smoke Dataset**: Place `Indoor Fire Smoke.zip` in the `data/` folder.
    *   **FlameVision Dataset**: Place `FlameVision  A new dataset for wildfire classification and detection using aerial imagery.zip` in the `data/` folder.
    *   *Note: These files are too large (>100MB) to be hosted on GitHub directly.*

3.  **Prepare Data**:
    *   Run the extraction script:
        ```bash
        python extract_data.py
        ```
    *   Run the preparation script to create the YOLO dataset:
        ```bash
        python prepare_data.py
        ```

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
The model achieves approximately **55% mAP@50** on the challenging combined dataset.

## ⚠️ Important Note on Data
The `data/` folder and `runs/` folder are excluded from the repository via `.gitignore` because they contain large files that exceed GitHub's 100MB limit. You must download the datasets and train the model locally to generate the weights.
