import streamlit as st
import tempfile
import os
import cv2
from pathlib import Path
from inference import EmotionVideoProcessor

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("🎭 Video Emotion Detection")
st.write("Upload a short video. YOLOv8 detects faces, a custom CNN classifies each face's emotion.")

APP_ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = str(APP_ROOT / "models/face_detector/best.pt")
CNN_WEIGHTS = str(APP_ROOT / "models/emotion_classifier/best_emotion_model.pt")

@st.cache_resource
def load_processor():
    return EmotionVideoProcessor(
        yolo_weights_path=YOLO_WEIGHTS,
        cnn_weights_path=CNN_WEIGHTS,
        face_confidence=0.40,
        emotion_confidence_threshold=45.0
    )

processor = load_processor()
st.success(f"Models loaded. Running on: {processor.device}")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name

    st.video(input_path)

    if st.button("Run Emotion Detection"):
        with st.spinner("Processing video frame by frame..."):
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_path = input_path.replace(".mp4", "_output.mp4")
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            frame_count = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
                processed_frame, detections = processor.process_frame(frame)
                writer.write(processed_frame)
                frame_count += 1

            cap.release()
            writer.release()

        st.success(f"Done! Processed {frame_count} frames.")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("Download annotated video", f, file_name="emotion_output.mp4")

    os.unlink(input_path)
