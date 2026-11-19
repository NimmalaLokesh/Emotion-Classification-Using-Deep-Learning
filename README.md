# Emotion Classification Using Deep Learning — Streamlit

This Streamlit application follows the project described in the supplied PPT:
- Audio emotion classification with MFCC features and a CNN-LSTM model.
- Facial-expression classification from webcam snapshots with a CNN model.
- Registration and login stored in SQLite.
- Dashboard, audio upload, and webcam capture screens.

## 1. Requirements

Use Python 3.10 or 3.11. TensorFlow support depends on the Python version and operating system.

Open this folder in VS Code and run in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Add datasets

The PPT does not include datasets or trained weights. Obtain the RAVDESS speech audio dataset and FER2013 dataset separately.

Audio files should be placed in folders like:
`data/audio/Actor_01/*.wav`

Train audio model:
```powershell
python train_audio.py --data data/audio
```
This creates `models/audio_model.keras`.

Place the FER2013 CSV at `data/fer2013.csv`. It must have `emotion` and `pixels` columns, with standard FER2013 labels:
0 angry, 1 disgust, 2 fear, 3 happy, 4 sad, 5 surprise, 6 neutral.

Train video model:
```powershell
python train_video.py --csv data/fer2013.csv
```
This creates `models/video_model.keras`.

## 3. Run Streamlit

```powershell
streamlit run streamlit_app.py
```

Streamlit opens the app in your browser (usually `http://localhost:8501`). Register, log in, then use the sidebar to select Audio Recognition or Live Video Recognition.

## Notes

- Webcam module uses Streamlit's camera snapshot control, not continuous frame-by-frame video streaming.
- The app estimates expression categories; it cannot reliably determine a person's actual internal emotional state.
- Audio and video predictions are separate. This version does not implement multimodal fusion.
- For deployment, use HTTPS, a strong secret/session strategy, CSRF/security review, and an appropriate authentication system. The sample SQLite login is intended for a local academic prototype.
