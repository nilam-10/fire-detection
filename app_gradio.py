import gradio as gr
from ultralytics import YOLO
import cv2
import os
import pandas as pd

# --- LOAD MODELS ---
models = {
    "YOLOv8 (Best)": "runs/detect/yolo_fire_det/weights/best.pt",
    "YOLOv5": "runs/detect/yolov5_fire_det/weights/best.pt"
}

def load_model(model_name):
    path = models.get(model_name)
    if path and os.path.exists(path):
        return YOLO(path)
    return None

# --- INFERENCE FUNCTIONS ---
def detect_image(image, model_name, conf):
    model = load_model(model_name)
    if model is None:
        return image
    
    results = model(image, conf=conf)
    return results[0].plot()

def detect_video(video_path, model_name, conf):
    model = load_model(model_name)
    if model is None:
        return video_path
    
    cap = cv2.VideoCapture(video_path)
    output_path = "output_videos/gradio_output.mp4"
    os.makedirs("output_videos", exist_ok=True)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=conf)
        res_plotted = results[0].plot()
        out.write(res_plotted)
        
    cap.release()
    out.release()
    return output_path

# --- COMPARISON DATA ---
def get_comparison_table():
    data = {
        "Model": ["YOLOv8", "YOLOv5", "YOLOv10", "YOLO11", "RT-DETR (5ep)"],
        "mAP@50": ["54.9%", "53.3%", "48.9%", "54.1%", "44.6%"],
        "Precision": ["58.5%", "59.9%", "55.1%", "57.5%", "50.4%"],
        "Recall": ["51.8%", "48.1%", "45.8%", "50.5%", "45.7%"]
    }
    return pd.DataFrame(data)

import yt_dlp

def detect_youtube(url, model_name, conf):
    model = load_model(model_name)
    if model is None:
        return None
    
    # Download video
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': 'yt_temp/gradio_yt.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'force_overwrite': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        video_path = "yt_temp/gradio_yt.mp4"
    except Exception as e:
        return None # Handle error gracefully
        
    # Process video (reuse detect_video logic)
    return detect_video(video_path, model_name, conf)

# --- UI ---
with gr.Blocks(title="Fire Detection System") as demo:
    gr.Markdown("# 🔥 Real-Time Fire Detection System")
    gr.Markdown("### Honours Final Project")
    
    with gr.Tab("🕵️ Image Detection"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="numpy", label="Upload Image")
                model_select_img = gr.Dropdown(choices=list(models.keys()), value="YOLOv8 (Best)", label="Select Model")
                conf_slider_img = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
                btn_img = gr.Button("Detect Fire")
            with gr.Column():
                img_output = gr.Image(type="numpy", label="Result")
        
        btn_img.click(detect_image, inputs=[img_input, model_select_img, conf_slider_img], outputs=img_output)

    with gr.Tab("🎥 Video Detection"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label="Upload Video")
                model_select_vid = gr.Dropdown(choices=list(models.keys()), value="YOLOv8 (Best)", label="Select Model")
                conf_slider_vid = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
                btn_vid = gr.Button("Process Video")
            with gr.Column():
                vid_output = gr.Video(label="Result")
        
        btn_vid.click(detect_video, inputs=[vid_input, model_select_vid, conf_slider_vid], outputs=vid_output)

    with gr.Tab("📺 YouTube Detection"):
        with gr.Row():
            with gr.Column():
                yt_input = gr.Textbox(label="YouTube URL", placeholder="Paste link here...")
                model_select_yt = gr.Dropdown(choices=list(models.keys()), value="YOLOv8 (Best)", label="Select Model")
                conf_slider_yt = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
                btn_yt = gr.Button("Process YouTube Video")
            with gr.Column():
                yt_output = gr.Video(label="Result")
        
        btn_yt.click(detect_youtube, inputs=[yt_input, model_select_yt, conf_slider_yt], outputs=yt_output)

    with gr.Tab("📊 Model Comparison"):
        gr.DataFrame(value=get_comparison_table(), label="Performance Metrics")
        gr.Markdown("""
        **Analysis:**
        *   **YOLOv8** is the overall winner with **54.9% mAP**.
        *   **YOLOv5** has the highest **Precision** (fewest false alarms).
        """)

if __name__ == "__main__":
    demo.launch()
