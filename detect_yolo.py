from ultralytics import YOLO
import cv2
import argparse
import time

def run_inference(source, model_path="runs/detect/yolo_fire_det/weights/best.pt"):
    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using default yolov8n.pt (will not detect fire well).")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(model_path)

    # Open video source
    if source.isdigit():
        source = int(source) # Webcam
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening video source: {source}")
        return

    # Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter('output_videos/yolo_output.mp4', 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps if fps > 0 else 30, 
                          (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Inference
        results = model(frame, verbose=False)
        
        # Visualize
        annotated_frame = results[0].plot()
        
        # FPS
        fps_text = f"FPS: {1.0 / (time.time() - start_time):.1f}"
        cv2.putText(annotated_frame, fps_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 Fire Detection", annotated_frame)
        out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Video file path or webcam index (0)")
    parser.add_argument("--model", default="runs/detect/yolo_fire_det/weights/best.pt", help="Path to trained model")
    args = parser.parse_args()
    
    run_inference(args.source, args.model)
