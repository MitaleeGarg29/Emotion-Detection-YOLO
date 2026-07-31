
import csv
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from inference import EmotionVideoProcessor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_ROOT = PROJECT_ROOT / "api"
UPLOAD_DIR = API_ROOT / "uploads"
OUTPUT_DIR = API_ROOT / "outputs"

YOLO_WEIGHTS_PATH = (
    PROJECT_ROOT / "models" / "face_detector" /
    "yolov8n_face" / "weights" / "best.pt"
)

CNN_WEIGHTS_PATH = (
    PROJECT_ROOT / "models" / "emotion_classifier" /
    "best_emotion_model.pt"
)

for folder in [UPLOAD_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the two trained models once when the API starts.
    They are then reused for every video request.
    """

    app.state.processor = EmotionVideoProcessor(
        yolo_weights_path=str(YOLO_WEIGHTS_PATH),
        cnn_weights_path=str(CNN_WEIGHTS_PATH),
        face_confidence=0.40,
        emotion_confidence_threshold=45.0
    )

    print("YOLO face detector and CNN emotion classifier loaded.")
    yield
    print("API stopped.")


app = FastAPI(
    title="Video Emotion Detection API",
    description=(
        "Upload a video. The API detects faces using YOLOv8 and "
        "classifies facial emotions using a trained CNN."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "Video Emotion Detection API is running.",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
def health():
    processor = app.state.processor

    return {
        "status": "healthy",
        "device": str(processor.device),
        "face_detection_model": "YOLOv8n fine-tuned on WIDER FACE",
        "emotion_classification_model": "Custom CNN trained on FER-2013"
    }


@app.post("/process-video")
async def process_video(video: UploadFile = File(...)):
    """
    Upload a video, process every frame, and return URLs for:
    - annotated emotion-detection video
    - frame-level CSV analytics report
    """

    allowed_extensions = {".mp4", ".avi", ".mov"}
    suffix = Path(video.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only .mp4, .avi, and .mov video files are supported."
        )

    # Each request has an isolated job folder
    job_id = str(uuid.uuid4())
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id

    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    input_video_path = job_upload_dir / f"input{suffix}"
    output_video_path = job_output_dir / "emotion_detection_output.mp4"
    report_path = job_output_dir / "frame_level_emotion_report.csv"

    # Save uploaded video
    with open(input_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cap = cv2.VideoCapture(str(input_video_path))

    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be opened as a video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    processor = app.state.processor
    report_rows = []

    frame_number = 0
    start_time = time.time()

    while True:
        success, frame = cap.read()

        if not success:
            break

        processed_frame, detections = processor.process_frame(frame)
        writer.write(processed_frame)

        timestamp_seconds = round(frame_number / fps, 3)

        # Record all detected faces in the frame
        if detections:
            for detection in detections:
                report_rows.append({
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "faces_detected_in_frame": len(detections),
                    **detection
                })
        else:
            report_rows.append({
                "frame_number": frame_number,
                "timestamp_seconds": timestamp_seconds,
                "faces_detected_in_frame": 0,
                "face_id": None,
                "bounding_box": None,
                "yolo_face_confidence_percent": None,
                "displayed_emotion": "No face detected",
                "raw_emotion": None,
                "cnn_emotion_confidence_percent": None
            })

        frame_number += 1

    cap.release()
    writer.release()

    processing_time_seconds = round(time.time() - start_time, 2)
    processing_fps = round(
        frame_number / processing_time_seconds,
        2
    ) if processing_time_seconds > 0 else 0.0

    # Save machine-readable frame-level report
    csv_columns = [
        "frame_number",
        "timestamp_seconds",
        "faces_detected_in_frame",
        "face_id",
        "bounding_box",
        "yolo_face_confidence_percent",
        "displayed_emotion",
        "raw_emotion",
        "cnn_emotion_confidence_percent"
    ]

    with open(report_path, "w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer_csv.writeheader()
        writer_csv.writerows(report_rows)

    return {
        "message": "Video processed successfully.",
        "job_id": job_id,
        "input_video": video.filename,
        "total_frames_processed": frame_number,
        "input_video_fps": round(fps, 2),
        "pipeline_fps": processing_fps,
        "processing_time_seconds": processing_time_seconds,
        "processed_video_url": f"/download/{job_id}/emotion_detection_output.mp4",
        "analytics_report_url": f"/download/{job_id}/frame_level_emotion_report.csv"
    }


@app.get("/download/{job_id}/{file_name}")
def download_output(job_id: str, file_name: str):
    """
    Download a processed video or its CSV report.
    """

    allowed_files = {
        "emotion_detection_output.mp4",
        "frame_level_emotion_report.csv"
    }

    if file_name not in allowed_files:
        raise HTTPException(status_code=404, detail="File not found.")

    output_path = OUTPUT_DIR / job_id / file_name

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Job output not found.")

    media_type = "video/mp4" if file_name.endswith(".mp4") else "text/csv"

    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=file_name
    )
