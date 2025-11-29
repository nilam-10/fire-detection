from ultralytics import RTDETR

def train_rtdetr():
    # Load the model.
    model = RTDETR('rtdetr-l.pt')
    
    # Train the model
    results = model.train(
        data='data/yolo_dataset/data.yaml',
        epochs=5,
        imgsz=640,
        batch=8,
        workers=0, # Fix for Windows
        patience=5,
        name="rtdetr_fire_det",
        device=0,
        exist_ok=True
    )
    
    print("RT-DETR Training Complete.")

if __name__ == '__main__':
    train_rtdetr()
