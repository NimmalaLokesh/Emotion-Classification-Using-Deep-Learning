import os
import cv2
import numpy as np
from keras.models import load_model

# Must match train_video.py order exactly
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

MODEL_PATH = os.path.join("models", "video_model.keras")
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_MODEL = None


def get_video_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Video model not found. Train it first using: python train_video.py --data data/images"
            )
        _MODEL = load_model(MODEL_PATH)
    return _MODEL


def annotate_live_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Detects faces in a live BGR frame, predicts emotion, draws bounding box + label,
    and returns (annotated_frame, latest_prediction_dict).
    """
    model = get_video_model()
    annotated = frame_bgr.copy()
    gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)

    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )

    if len(faces) == 0:
        return annotated, {"emotion": "No face detected", "confidence": 0.0, "probabilities": {}}

    # Batch-predict all detected faces in one forward pass for smooth FPS
    rois = []
    for (x, y, w, h) in faces:
        face_roi = _CLAHE.apply(gray[y : y + h, x : x + w])
        face_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        rois.append(face_resized.astype("float32")[..., np.newaxis])

    batch_input = np.stack(rois, axis=0)
    all_probs = model.predict(batch_input, verbose=0)

    # Track the largest face for the live telemetry panel
    largest_idx = int(np.argmax([w * h for (_, _, w, h) in faces]))
    primary_result = {"emotion": "No face detected", "confidence": 0.0, "probabilities": {}}

    for i, (x, y, w, h) in enumerate(faces):
        probs = all_probs[i]
        idx = int(np.argmax(probs))
        emotion = EMOTIONS[idx]
        conf = float(probs[idx]) * 100.0

        # Draw bounding box and label directly on the live stream
        color = (0, 220, 100) if i == largest_idx else (255, 180, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{emotion.title()} ({conf:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x, max(0, y - th - 10)), (x + tw + 8, y), color, -1)
        cv2.putText(
            annotated,
            label,
            (x + 4, max(th + 2, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        if i == largest_idx:
            primary_result = {
                "emotion": emotion,
                "confidence": conf,
                "probabilities": {EMOTIONS[j]: float(probs[j]) * 100.0 for j in range(len(EMOTIONS))},
            }

    return annotated, primary_result