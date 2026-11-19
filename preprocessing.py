import numpy as np
import librosa

AUDIO_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
def extract_mfcc(path, duration=3.0, sample_rate=22050, n_mfcc=40, max_frames=130):
    signal, sr = librosa.load(path, sr=sample_rate, duration=duration)
    needed = int(duration * sample_rate)
    if len(signal) < needed:
        signal = np.pad(signal, (0, needed-len(signal)))
    else:
        signal = signal[:needed]
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0,0),(0,max_frames-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_frames]
    return mfcc.T.astype("float32")

def audio_label_from_ravdess(filename):
    parts = filename.split("-")
    if len(parts) < 3 or parts[2] not in AUDIO_EMOTIONS:
        raise ValueError(f"Unexpected RAVDESS filename: {filename}")
    return AUDIO_EMOTIONS[parts[2]]
