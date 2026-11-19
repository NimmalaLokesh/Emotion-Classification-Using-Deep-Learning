# 🎭 Emotion Classification Using Deep Learning

<p align="center">
  <b>A Deep Learning-Based Emotion Recognition System Using Audio and Facial Expressions</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Database-SQLite-blue" />
  <img src="https://img.shields.io/badge/Status-Academic%20Project-success" />
</p>

---

## 📌 Overview

**Emotion Classification Using Deep Learning** is a web-based application designed to recognize emotion categories from human speech and facial expressions.

The application combines Deep Learning techniques with computer vision and audio processing to analyze uploaded audio recordings and facial images captured through a webcam.

Users can register, log in, and interact with an intuitive Streamlit dashboard to explore emotion predictions from different input modalities.

---

## ✨ Key Features

- 🎤 **Audio Emotion Recognition** – Predicts emotion categories from uploaded audio recordings.
- 🧠 **CNN-LSTM Architecture** – Uses deep learning to learn emotional patterns from audio features.
- 🎵 **MFCC Feature Extraction** – Extracts meaningful audio characteristics for emotion classification.
- 📷 **Facial Emotion Recognition** – Detects facial-expression categories using a CNN model.
- 📸 **Webcam Integration** – Captures images directly through the browser.
- 🔐 **User Authentication** – Registration and login functionality using SQLite.
- 📊 **Interactive Dashboard** – Simple and user-friendly interface built with Streamlit.
- ⚡ **Real-Time Image Prediction** – Predicts facial-expression categories from captured images.

---

## 🧠 System Architecture

The project consists of two independent emotion recognition modules.

### 🎤 Audio Emotion Recognition

Audio recordings are processed to extract MFCC features, which are passed into a CNN-LSTM deep learning model to classify emotion categories.

**Pipeline:**

`Audio Input → Preprocessing → MFCC Extraction → CNN-LSTM Model → Emotion Prediction`

### 📷 Facial Emotion Recognition

Images captured through a webcam are processed and resized before being passed into a CNN model trained on facial-expression data.

**Pipeline:**

`Webcam Image → Image Preprocessing → CNN Model → Emotion Prediction`

---

## 🎭 Emotion Categories

The application classifies emotions into the following seven categories:

| Label | Emotion |
|---|---|
| 0 | 😠 Angry |
| 1 | 🤢 Disgust |
| 2 | 😨 Fear |
| 3 | 😊 Happy |
| 4 | 😢 Sad |
| 5 | 😲 Surprise |
| 6 | 😐 Neutral |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model development |
| CNN | Facial-expression classification |
| CNN-LSTM | Audio emotion classification |
| Librosa | Audio processing and MFCC extraction |
| NumPy | Numerical operations |
| OpenCV | Image processing |
| Streamlit | Web application interface |
| SQLite | User registration and login database |
| Scikit-learn | Machine learning utilities and evaluation |

---

## 📂 Project Structure

```text
Emotion_Recognition_Streamlit/
│
├── streamlit_app.py
├── train_audio.py
├── train_video.py
├── audio_prediction.py
├── video_prediction.py
│
├── requirements.txt
│
├── models/
│   ├── audio_model.keras
│   └── video_model.keras
│
├── data/
│   ├── audio/
│   │   └── Actor_01/
│   │       ├── audio_file.wav
│   │       └── ...
│   │
│   └── fer2013.csv
│
├── database/
│   └── users.db
│
└── README.md
```

> Dataset files and trained model files may need to be downloaded or generated separately.

---

## 📊 Datasets

### 🎤 Audio Dataset – RAVDESS

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is used for audio emotion recognition.

It contains recordings representing different emotional expressions.

🔗 [Download RAVDESS Dataset](https://zenodo.org/record/1188976)

### 📷 Facial Expression Dataset – FER2013

FER2013 is a facial-expression dataset containing grayscale facial images categorized into seven emotion classes.

🔗 [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Emotion_Recognition_Streamlit.git
```

Navigate to the project directory:

```bash
cd Emotion_Recognition_Streamlit
```

### 2️⃣ Create a Virtual Environment

Python 3.10 or Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
```

### 3️⃣ Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4️⃣ Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 🏋️ Model Training

### 🎤 Train the Audio Emotion Recognition Model

Place the RAVDESS audio dataset inside:

```text
data/audio/
```

Example:

```text
data/audio/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
```

Run the training script:

```powershell
python train_audio.py --data data/audio
```

After training, the model will be saved as:

```text
models/audio_model.keras
```

### 📷 Train the Facial Emotion Recognition Model

Place the FER2013 CSV dataset inside:

```text
data/fer2013.csv
```

The CSV file must contain the following columns:

- `emotion`
- `pixels`

Run:

```powershell
python train_video.py --csv data/fer2013.csv
```

The trained model will be saved as:

```text
models/video_model.keras
```

---

## 🚀 Running the Application

Start the Streamlit application:

```powershell
streamlit run streamlit_app.py
```

The application will open in your browser:

```text
http://localhost:8501
```

### 🖥️ Application Workflow

1. Register a new account.
2. Log in using your credentials.
3. Navigate through the dashboard.
4. Select Audio Emotion Recognition or Live Video Recognition.
5. Upload an audio file or capture an image using your webcam.
6. View the predicted emotion.

---

## 📸 Application Screenshots

> Add screenshots of your application here to showcase the user interface.

### 🔐 Login Page

<!-- Add your login screenshot here -->

### 📊 Dashboard

<!-- Add your dashboard screenshot here -->

### 🎤 Audio Emotion Recognition

<!-- Add your audio prediction screenshot here -->

### 📷 Facial Emotion Recognition

<!-- Add your webcam prediction screenshot here -->

---

## 🔮 Future Enhancements

- 🎥 Continuous real-time video emotion recognition.
- 🔊 Improved audio emotion classification accuracy.
- 🤝 Multimodal emotion recognition combining audio and facial expressions.
- 📈 Emotion confidence scores and visual analytics.
- 🌐 Deployment to a cloud platform.
- 🛡️ Improved authentication and security.
- 📱 Responsive UI for mobile devices.

---

## ⚠️ Disclaimer

This application predicts emotion-related categories based on patterns learned from audio and facial-expression datasets.

These predictions represent **estimated emotional expressions, not verified internal emotional states**. Results may be affected by lighting, recording quality, facial expressions, background noise, individual differences, and model limitations.

This project is developed for educational and research purposes.

---

## 👨‍💻 Author

**Lokesh Nimmala**

🎓 Computer Science Graduate | Python | Machine Learning | Deep Learning

---

<p align="center">
  ⭐ If you find this project interesting, consider starring the repository!
</p>
