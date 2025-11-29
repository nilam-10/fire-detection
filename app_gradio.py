import gradio as gr
from ultralytics import YOLO
import cv2
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Fix for "main thread is not in main loop"
import matplotlib.pyplot as plt
import yt_dlp
import numpy as np

# --- CONFIGURATION ---
MODELS = {
    "YOLOv8 (Best - 50 Epochs)": "runs/detect/yolov8_50epochs/weights/best.pt",
    "YOLOv8 (30 Epochs)": "runs/detect/yolo_fire_det/weights/best.pt",
    "YOLOv5": "runs/detect/yolov5_fire_det/weights/best.pt"
}

# --- HELPER FUNCTIONS ---
def load_model(model_name):
    path = MODELS.get(model_name)
    if path and os.path.exists(path):
        return YOLO(path)
    return None

def detect_image(image, model_name, conf):
    model = load_model(model_name)
    if model is None:
        return image
    results = model(image, conf=conf)
    return results[0].plot()

import uuid

def detect_video(video_path, model_name, conf):
    model = load_model(model_name)
    if model is None:
        return video_path
    
    cap = cv2.VideoCapture(video_path)
    # Use unique filename to prevent browser caching
    unique_id = str(uuid.uuid4())[:8]
    output_path = f"output_videos/gradio_output_{unique_id}.mp4"
    os.makedirs("output_videos", exist_ok=True)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Try 'avc1' (H.264) for browser compatibility
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
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

def detect_youtube(url, model_name, conf):
    unique_id = str(uuid.uuid4())[:8]
    temp_path = f"yt_temp/gradio_yt_{unique_id}.%(ext)s"
    final_path = f"yt_temp/gradio_yt_{unique_id}.mp4"
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': temp_path,
        'quiet': True,
        'no_warnings': True,
        'force_overwrite': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return detect_video(final_path, model_name, conf)
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return None

# --- DASHBOARD PLOTS ---
def get_comparison_data():
    data = {
        "Model": ["YOLOv8 (50ep)", "YOLOv8 (30ep)", "YOLOv5", "YOLOv10", "YOLO11", "RT-DETR (5ep)"],
        "mAP@50": [56.1, 54.9, 53.3, 48.9, 54.1, 44.6],
        "Precision": [58.0, 58.5, 59.9, 55.1, 57.5, 50.4],
        "Recall": [53.6, 51.8, 48.1, 45.8, 50.5, 45.7]
    }
    return pd.DataFrame(data)

def plot_metrics():
    df = get_comparison_data()
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics = ["mAP@50", "Precision", "Recall"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(df["Model"], df[metric], color=colors[i])
        ax.set_title(metric)
        ax.set_ylim(0, 70)
        ax.tick_params(axis='x', rotation=45)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}%',
                    ha='center', va='bottom')
            
    plt.tight_layout()
    return fig

def get_training_plots():
    # Return paths to static images if they exist
    plots = []
    base_path = "runs/detect/yolov8_50epochs"
    for fname in ["results.png", "confusion_matrix.png", "F1_curve.png"]:
        p = os.path.join(base_path, fname)
        if os.path.exists(p):
            plots.append((p, fname.replace(".png", "").replace("_", " ")))
    return plots

# --- GRADIO UI ---
with gr.Blocks(title="Fire Detection Dashboard") as demo:
    gr.Markdown(
        """
        # 🔥 Fire Detection System - Professional Dashboard
        **Honours Final Project** | Model: YOLOv8 (50 Epochs)
        """
    )
    
    with gr.Tab("📊 Analytics Dashboard"):
        gr.Markdown("### 🏆 Model Comparison")
        gr.Plot(plot_metrics, label="Performance Metrics")
        
        gr.Markdown("### 📈 Training History (Best Model)")
        with gr.Row():
            # Dynamically load training plots
            plots = get_training_plots()
            for p, label in plots:
                with gr.Column():
                    gr.Image(p, label=label, type="filepath")
                    
        gr.Markdown("### 📋 Detailed Metrics")
        gr.DataFrame(get_comparison_data(), label="Data Table")

    with gr.Tab("🕵️ Live Detection"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                model_select = gr.Dropdown(choices=list(MODELS.keys()), value="YOLOv8 (Best - 50 Epochs)", label="Select Model")
                conf_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Confidence Threshold")
            
            with gr.Column(scale=3):
                with gr.Tab("Image"):
                    with gr.Row():
                        img_in = gr.Image(type="numpy", label="Input")
                        img_out = gr.Image(type="numpy", label="Output")
                    btn_img = gr.Button("Detect Fire", variant="primary")
                    btn_img.click(detect_image, inputs=[img_in, model_select, conf_slider], outputs=img_out)
                    
                with gr.Tab("Video"):
                    with gr.Row():
                        vid_in = gr.Video(label="Input")
                        vid_out = gr.Video(label="Output")
                    btn_vid = gr.Button("Process Video", variant="primary")
                    btn_vid.click(detect_video, inputs=[vid_in, model_select, conf_slider], outputs=vid_out)
                    
                with gr.Tab("YouTube"):
                    with gr.Row():
                        yt_in = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/...")
                        yt_out = gr.Video(label="Output")
                    btn_yt = gr.Button("Process Stream", variant="primary")
                    btn_yt.click(detect_youtube, inputs=[yt_in, model_select, conf_slider], outputs=yt_out)

if __name__ == "__main__":
    demo.launch(share=False)
