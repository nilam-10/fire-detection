from ultralytics import YOLO
import os

def train_yolo():
    # Load YOLOv8 Nano model
    model = YOLO("yolov8n.pt") 

    data_yaml_path = os.path.abspath("data/yolo_dataset/data.yaml")
    
    print(f"Starting YOLOv8 Training...")
    
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        patience=5,
        name="yolo_fire_det",
        device=0,
        verbose=True
    )
    
    print("YOLOv8 Training Complete.")
    print(f"Best model: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_yolo()
