from ultralytics import RTDETR
import os

def train_rtdetr():
    # Load RT-DETR model (ResNet18 backbone version for speed)
    # rtdetr-l.pt is large, let's try rtdetr-l or x. 
    # Actually ultralytics provides rtdetr-l.pt and rtdetr-x.pt. 
    # They are heavier than YOLOv8n.
    model = RTDETR("rtdetr-l.pt") 

    data_yaml_path = os.path.abspath("data/yolo_dataset/data.yaml")
    
    print(f"Starting RT-DETR Training...")
    
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        batch=8, # RT-DETR is heavy, reduce batch size
        patience=5,
        name="rtdetr_fire_det",
        device=0,
        verbose=True
    )
    
    print("RT-DETR Training Complete.")
    print(f"Best model: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_rtdetr()
