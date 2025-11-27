from ultralytics import YOLO
import os

def train_yolo11():
    # Load YOLO11 Nano model
    # Ultralytics names it "yolo11n.pt"
    model = YOLO("yolo11n.pt") 

    data_yaml_path = os.path.abspath("data/yolo_dataset/data.yaml")
    
    print(f"Starting YOLO11 Training...")
    
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        patience=5,
        name="yolo11_fire_det",
        device=0,
        verbose=True
    )
    
    print("YOLO11 Training Complete.")
    print(f"Best model: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_yolo11()
