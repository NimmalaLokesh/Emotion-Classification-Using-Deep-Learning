import os
import numpy as np
from keras.models import load_model
from preprocessing import extract_mfcc

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
MODEL_PATH = os.path.join("models", "audio_model.keras")
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Audio model not found. Train it first using: python train_audio.py --data data/audio"
            )
        _MODEL = load_model(MODEL_PATH)
    return _MODEL


def predict_audio(path: str) -> dict:
    model = _get_model()
    x = extract_mfcc(path)[np.newaxis, ...]
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "emotion": EMOTIONS[idx],
        "confidence": float(probs[idx]) * 100.0,
        "probabilities": {EMOTIONS[i]: float(probs[i]) * 100.0 for i in range(len(EMOTIONS))},
    }