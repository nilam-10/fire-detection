from ultralytics import YOLO

def train_yolov10():
    # Load the model. 
    # Note: If yolov10n.pt is not available in standard ultralytics yet, 
    # it might download it or we might need to use yolov9c.pt
    try:
        model = YOLO('yolov10n.pt') 
    except Exception:
        print("YOLOv10n not found, falling back to YOLOv9c")
        model = YOLO('yolov9c.pt')

    # Train the model
    results = model.train(
        data='data/yolo_dataset/data.yaml',
        epochs=30,
        imgsz=640,
        batch=16,
        name='yolov10_fire_det',
        device=0,
        patience=10,
        exist_ok=True
    )
    
    print("YOLOv10 Training Complete.")

if __name__ == '__main__':
    train_yolov10()
