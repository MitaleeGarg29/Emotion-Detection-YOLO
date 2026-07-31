
import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.se4 = SEBlock(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.se2(x)

        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))

        return self.fc2(x)


class EmotionVideoProcessor:
    """
    Loads the trained YOLO face detector and CNN emotion classifier.
    Provides functions to process individual video frames.
    """

    def __init__(
        self,
        yolo_weights_path,
        cnn_weights_path,
        face_confidence=0.40,
        emotion_confidence_threshold=45.0
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.face_confidence = face_confidence
        self.emotion_confidence_threshold = emotion_confidence_threshold

        self.emotion_names = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }

        # Load trained YOLO face detector
        self.face_model = YOLO(yolo_weights_path)

        # Load trained CNN emotion classifier
        self.emotion_model = EmotionCNN(num_classes=7).to(self.device)

        checkpoint = torch.load(
            cnn_weights_path,
            map_location=self.device
        )

        self.emotion_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.emotion_model.eval()

        # Same preprocessing used during CNN testing
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict_emotion(self, face_bgr):
        """Predict emotion and confidence from one detected face crop."""

        gray_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (48, 48))

        face_pil = Image.fromarray(gray_face)
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.emotion_model(face_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

            emotion_id = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[emotion_id].item() * 100)

        raw_emotion = self.emotion_names[emotion_id]

        if confidence < self.emotion_confidence_threshold:
            return "Uncertain", confidence, raw_emotion

        return raw_emotion, confidence, raw_emotion

    def process_frame(self, frame):
        """
        Detect faces with YOLO, classify every detected face with the CNN,
        draw annotations, and return frame-level prediction details.
        """

        output_frame = frame.copy()
        detections = []

        yolo_results = self.face_model.predict(
            source=frame,
            conf=self.face_confidence,
            imgsz=640,
            verbose=False
        )

        result = yolo_results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return output_frame, detections

        for face_id, box in enumerate(result.boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            yolo_confidence = float(box.conf[0].cpu().numpy() * 100)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                continue

            display_emotion, emotion_confidence, raw_emotion = self.predict_emotion(
                face_crop
            )

            # Orange means uncertain; green means accepted prediction
            colour = (0, 165, 255) if display_emotion == "Uncertain" else (0, 255, 0)

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), colour, 2)

            label = f"{display_emotion} ({emotion_confidence:.1f}%)"
            cv2.putText(
                output_frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                colour,
                2,
                cv2.LINE_AA
            )

            detections.append({
                "face_id": face_id,
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                "yolo_face_confidence_percent": round(yolo_confidence, 2),
                "displayed_emotion": display_emotion,
                "raw_emotion": raw_emotion,
                "cnn_emotion_confidence_percent": round(emotion_confidence, 2)
            })

        return output_frame, detections
