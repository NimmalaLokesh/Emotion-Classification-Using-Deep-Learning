import os
import sqlite3
import tempfile
import threading
import time
import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from werkzeug.security import generate_password_hash, check_password_hash
from audio_prediction import predict_audio
from video_prediction import annotate_live_frame, get_video_model

st.set_page_config(page_title="Emotion Recognition", page_icon="🎭", layout="wide")

DB_PATH = os.environ.get("USERS_DB_PATH", "users.db")
os.makedirs("models", exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )"""
        )


def register_user(name: str, email: str, password: str):
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users(name, email, password) VALUES(?, ?, ?)",
                (name.strip(), email.strip().lower(), generate_password_hash(password)),
            )
        return True, "Registration successful. Please log in."
    except sqlite3.IntegrityError:
        return False, "This email is already registered."


def authenticate(email: str, password: str):
    with get_db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE email=?", (email.strip().lower(),)
        ).fetchone()
    if user and check_password_hash(user["password"], password):
        return {"id": user["id"], "name": user["name"]}
    return None


def logout():
    for key in ("logged_in", "user", "page"):
        st.session_state.pop(key, None)
    st.rerun()


init_db()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.title("🎭 Emotion Classification Using Deep Learning")
st.caption("Recognizing emotion patterns from speech audio and live facial expressions")

if not st.session_state.logged_in:
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        st.subheader("User Login")
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Login", use_container_width=True)
        if login_submit:
            user = authenticate(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.session_state.page = "Dashboard"
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    with tab_register:
        st.subheader("Create an account")
        with st.form("register_form"):
            name = st.text_input("Full name", key="register_name")
            reg_email = st.text_input("Email address", key="register_email")
            reg_password = st.text_input(
                "Password (minimum 6 characters)", type="password", key="register_password"
            )
            register_submit = st.form_submit_button("Register", use_container_width=True)
        if register_submit:
            if not name.strip() or not reg_email.strip() or len(reg_password) < 6:
                st.warning("Please enter all details and use a password with at least 6 characters.")
            else:
                ok, message = register_user(name, reg_email, reg_password)
                (st.success if ok else st.error)(message)

    st.info("Register an account or log in to access the audio and live-video modules.")
    st.stop()

with st.sidebar:
    st.success(f"Logged in as {st.session_state.user['name']}")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Audio Recognition", "Live Video Recognition"],
        key="navigation",
    )
    st.divider()
    if st.button("Logout", use_container_width=True):
        logout()

if page == "Dashboard":
    st.header(f"Welcome, {st.session_state.user['name']}!")
    st.write("Choose one of the following modules to estimate emotion.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎙️ Audio Emotion Recognition")
        st.write(
            "Upload a speech recording. Extracts MFCC features and predicts one of eight audio emotion categories."
        )
    with col2:
        st.subheader("📷 Live Video Emotion Recognition")
        st.write(
            "Real-time webcam stream with automatic face tracking and live facial expression prediction."
        )
    st.warning(
        "Predictions are model estimates, not reliable measurements of a person's internal emotional state."
    )

elif page == "Audio Recognition":
    st.header("🎙️ Audio Emotion Recognition")
    st.write("Supported formats: WAV, MP3, OGG and FLAC.")
    audio_file = st.file_uploader("Upload an audio recording", type=["wav", "mp3", "ogg", "flac"])
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Predict Audio Emotion", type="primary"):
            suffix = os.path.splitext(audio_file.name)[1] or ".wav"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                    temp.write(audio_file.getvalue())
                    temp_path = temp.name
                with st.spinner("Analyzing audio..."):
                    result = predict_audio(temp_path)
                st.success(f"Predicted emotion: **{result['emotion'].title()}**")
                st.metric("Model confidence", f"{result['confidence']:.2f}%")
                if result.get("probabilities"):
                    df_probs = pd.DataFrame(
                        list(result["probabilities"].items()), columns=["Emotion", "Confidence (%)"]
                    ).set_index("Emotion")
                    st.bar_chart(df_probs)
            except Exception as exc:
                st.error(str(exc))
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

elif page == "Live Video Recognition":
    st.header("📷 Live Video Emotion Recognition")
    st.write("Click **START** below to begin real-time facial emotion detection.")

    # Preload model before starting WebRTC thread
    try:
        get_video_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    lock = threading.Lock()
    shared_state = {
        "result": {"emotion": "Waiting for video...", "confidence": 0.0, "probabilities": {}},
        "last_annotated": None,
        "frame_count": 0,
    }

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Mirror horizontally so webcam feels natural
        img = cv2.flip(img, 1)

        with lock:
            shared_state["frame_count"] += 1
            count = shared_state["frame_count"]

        # Run inference every 3rd frame to keep real-time FPS smooth
        if count % 3 == 1 or shared_state["last_annotated"] is None:
            annotated, result = annotate_live_frame(img)
            with lock:
                shared_state["result"] = result
                shared_state["last_annotated"] = result
        else:
            # Still draw the latest prediction box/label on intermediate frames
            annotated, _ = annotate_live_frame(img)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    col_video, col_metrics = st.columns([2, 1])

    with col_video:
        ctx = webrtc_streamer(
            key="live-emotion-stream",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_metrics:
        st.subheader("Live Analysis")
        status_placeholder = st.empty()
        chart_placeholder = st.empty()

        # Live telemetry loop while stream is active
        while ctx.state.playing:
            with lock:
                res = dict(shared_state["result"])

            with status_placeholder.container():
                if res["emotion"] == "No face detected":
                    st.warning("No face detected — face the camera with front lighting.")
                elif res["emotion"] == "Waiting for video...":
                    st.info("Initializing camera feed...")
                else:
                    st.success(f"Emotion: **{res['emotion'].title()}**")
                    st.metric("Confidence", f"{res['confidence']:.2f}%")

            if res.get("probabilities"):
                df_probs = pd.DataFrame(
                    list(res["probabilities"].items()),
                    columns=["Emotion", "Confidence (%)"],
                ).set_index("Emotion")
                chart_placeholder.bar_chart(df_probs)

            time.sleep(0.25)

st.divider()
st.caption(
    "CNN/LSTM-based audio classification and real-time CNN facial-expression classification."
)