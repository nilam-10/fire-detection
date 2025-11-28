from ultralytics import YOLO

def train_yolov5():
    # Load the model.
    # yolov5nu.pt is the YOLOv5 Nano model in Ultralytics format
    model = YOLO('yolov5nu.pt')
    
    # Train the model
    results = model.train(
        data='data/yolo_dataset/data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        name='yolov5_fire_det',
        device=0,
        patience=10,
        exist_ok=True
    )
    
    print("YOLOv5 Training Complete.")

if __name__ == '__main__':
    train_yolov5()
