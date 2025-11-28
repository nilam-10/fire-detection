import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import pandas as pd
import altair as alt
import yt_dlp

# Page Config
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

# Title
st.title("🔥 Real-Time Fire Detection System")
st.markdown("### Honours Final Project")

# Sidebar - Model Selection
st.sidebar.header("Model Configuration")
model_options = {
    "YOLOv8 (Best)": "runs/detect/yolo_fire_det/weights/best.pt",
    "YOLOv5": "runs/detect/yolov5_fire_det/weights/best.pt",
    "YOLOv10": "runs/detect/yolov10_fire_det/weights/best.pt",
    "YOLO11": "runs/detect/yolo11_fire_det/weights/best.pt"
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
model_path = model_options[selected_model_name]
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# Load Model
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(model_path)
    st.sidebar.success(f"Loaded {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None

# Tabs
tab1, tab2, tab3 = st.tabs(["🕵️ Detection", "📊 Model Comparison", "ℹ️ Project Info"])

# --- TAB 1: DETECTION ---
with tab1:
    st.header("Run Detection")
    
    input_source = st.radio("Select Input Source", ["Upload Video", "Upload Image", "YouTube URL"])
    
    if input_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            # Save temp file
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            # Run inference
            if st.button("Detect Fire"):
                results = model(tfile.name, conf=conf_threshold)
                res_plotted = results[0].plot()
                
                # Display
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption="Original Image", use_column_width=True)
                with col2:
                    st.image(res_plotted, caption="Detected Fire", use_column_width=True, channels="BGR")

    elif input_source == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            if st.button("Process Video"):
                st.info("Processing video... This may take a while.")
                cap = cv2.VideoCapture(video_path)
                
                # Output setup
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                output_path = os.path.join("output_videos", "streamlit_output.mp4")
                os.makedirs("output_videos", exist_ok=True)
                
                # Streamlit video placeholder
                st_frame = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Inference
                    results = model(frame, conf=conf_threshold)
                    res_plotted = results[0].plot()
                    
                    # Display in Streamlit
                    st_frame.image(res_plotted, channels="BGR", use_column_width=True)
                
                cap.release()
                st.success("Processing Complete!")

    elif input_source == "YouTube URL":
        yt_url = st.text_input("Paste YouTube URL")
        if st.button("Process YouTube Video"):
            if yt_url:
                st.info("Downloading video...")
                try:
                    ydl_opts = {
                        'format': 'best[ext=mp4]',
                        'outtmpl': 'yt_temp/streamlit_yt.%(ext)s',
                        'quiet': True,
                        'no_warnings': True
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                    
                    video_path = "yt_temp/streamlit_yt.mp4"
                    st.success("Download complete. Processing...")
                    
                    cap = cv2.VideoCapture(video_path)
                    st_frame = st.empty()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        results = model(frame, conf=conf_threshold)
                        res_plotted = results[0].plot()
                        st_frame.image(res_plotted, channels="BGR", use_column_width=True)
                    
                    cap.release()
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 2: COMPARISON ---
with tab2:
    st.header("Model Performance Comparison")
    
    # Data
    data = {
        "Model": ["YOLOv8", "YOLOv5", "YOLOv10", "YOLO11", "RT-DETR (5ep)"],
        "mAP@50": [54.9, 53.3, 48.9, 54.1, 44.6],
        "Precision": [58.5, 59.9, 55.1, 57.5, 50.4],
        "Recall": [51.8, 48.1, 45.8, 50.5, 45.7]
    }
    df = pd.DataFrame(data)
    
    # Table
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    # Charts
    st.subheader("Metric Visualization")
    
    # Melt for chart
    df_melt = df.melt('Model', var_name='Metric', value_name='Score')
    
    chart = alt.Chart(df_melt).mark_bar().encode(
        x=alt.X('Model', axis=None),
        y=alt.Y('Score', scale=alt.Scale(domain=[0, 70])),
        color='Model',
        column='Metric',
        tooltip=['Model', 'Metric', 'Score']
    ).properties(width=150)
    
    st.altair_chart(chart)
    
    st.markdown("""
    **Analysis:**
    *   **YOLOv8** is the overall winner with **54.9% mAP**.
    *   **YOLOv5** has the highest **Precision** (fewest false alarms).
    *   **YOLOv10** performed unexpectedly lower on this dataset.
    """)

# --- TAB 3: INFO ---
with tab3:
    st.header("Project Documentation")
    with open("README.md", "r") as f:
        st.markdown(f.read())
