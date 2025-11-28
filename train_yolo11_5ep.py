from ultralytics import YOLO

def train_yolo11():
    # Load the model.
    model = YOLO('yolo11n.pt')
    
    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=5,
        imgsz=640,
        batch=16,
        name='yolov11_5epochs',
        device=0,
        patience=5,
        exist_ok=True
    )
    
    print("YOLO11 (5 Epochs) Training Complete.")

if __name__ == '__main__':
    train_yolo11()
