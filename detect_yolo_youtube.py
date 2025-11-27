from ultralytics import YOLO
import cv2
import argparse
import time
import os
import subprocess
from pathlib import Path

def download_youtube_video(url: str, out_dir: Path = Path("yt_temp")) -> Path:
    """
    Downloads a YouTube video using yt-dlp to a temporary MP4 file.
    Returns the full path to the downloaded file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Output template for yt-dlp
    out_template = str(out_dir / "%(id)s.%(ext)s")
    
    # Command to download best mp4 format
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", out_template,
        "--no-playlist",
        url
    ]
    
    # Check if file already exists in yt_temp (partial match on ID)
    # Extract ID from URL (simple logic)
    import re
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if video_id_match:
        video_id = video_id_match.group(1)
        existing_files = list(out_dir.glob(f"*{video_id}*.mp4"))
        if existing_files:
            print(f"Found existing file for {video_id}: {existing_files[0]}")
            return existing_files[0]
            
    print(f"Downloading {url}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed with error: {e}")
        
    # Get filename
    cmd_get_filename = [
        "yt-dlp",
        "--get-filename",
        "-o", out_template,
        "--no-playlist",
        url
    ]
    result = subprocess.run(cmd_get_filename, capture_output=True, text=True, check=True)
    filename = result.stdout.strip()
    
    if not os.path.exists(filename):
         # Fallback search
         files = list(out_dir.glob("*.mp4"))
         if files:
             return max(files, key=os.path.getmtime)
         else:
             raise RuntimeError("Download failed: file not found.")
             
    return Path(filename)

def run_inference(video_path, model_path="runs/detect/yolo_fire_det/weights/best.pt"):
    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return

    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video source: {video_path}")
        return

    # Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Generate unique output filename
    video_name = Path(video_path).stem
    os.makedirs("output_videos", exist_ok=True)
    out_path = f"output_videos/yolo_{video_name}.mp4"
    
    out = cv2.VideoWriter(out_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps if fps > 0 else 30, 
                          (width, height))
    
    print(f"Processing video... Output will be saved to {out_path}")

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
    print(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="YouTube Video URL")
    parser.add_argument("--model", default="runs/detect/yolo_fire_det/weights/best.pt", help="Path to trained model")
    args = parser.parse_args()
    
    video_path = download_youtube_video(args.url)
    run_inference(video_path, args.model)
