
import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import os
import tempfile
import numpy as np
from torchvision import transforms, models
from model import FusionLSTMNet
from deepface import DeepFace

st.set_page_config(page_title="AI Deception Detector", layout="wide")

st.markdown("""
    <style>
body, .stApp, .stMarkdown, .stTextInput, .stDataFrame, .stAlert,
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: white !important;
}
section[data-testid="stFileUploader"] span {
    color: orange !important;
}
button[kind="primary"] {
    color: black !important;
    font-size: 14px !important;
    padding: 0.4rem 1rem !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "landing"

def set_bg(gif_file):
    st.markdown(f"""
        <style>
        .stApp {{
            background: url(data:image/gif;base64,{gif_file});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

def load_gif_base64(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def show_landing():
    gif_base64 = load_gif_base64("landing.gif")
    set_bg(gif_base64)
    st.markdown("<h1 style='text-align: center; color: white;'>AI Based Deception Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>Reveal truth through vision and emotion</p>", unsafe_allow_html=True)
    if st.button("Get Started", use_container_width=True):
        st.session_state.page = "analyze"

def show_analysis():
    gif_base64 = load_gif_base64("analysis.gif")
    set_bg(gif_base64)

    @st.cache_resource
    def load_model():
        vision_model = models.resnet18(weights=None)
        vision_model.fc = torch.nn.Identity()
        audio_model = models.resnet18(weights=None)
        audio_model.fc = torch.nn.Identity()
        model = FusionLSTMNet(vision_model, audio_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            state_dict = torch.load("lstmfusionvm.pth", map_location=device)
        except FileNotFoundError:
            st.error("❌ Model file not found. The .pth file is private and not shared in this repository.")
            st.stop()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device

    model, device = load_model()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def extract_middle_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def extract_frames_by_second(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        count, saved = 0, []
        temp_dir = tempfile.mkdtemp()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % fps == 0:
                timestamp = int(count / fps)
                frame_path = os.path.join(temp_dir, f"frame_{timestamp}s.jpg")
                cv2.imwrite(frame_path, frame)
                saved.append((timestamp, frame_path))
            count += 1
        cap.release()
        return saved

    def analyze_emotions(frames):
        results = []
        for ts, path in frames:
            try:
                analysis = DeepFace.analyze(img_path=path, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
            except:
                emotion = "Error"
            results.append((ts, emotion, path))
        return results

    def detect_deception_moments(emotions):
        deception_cues = {'fear', 'surprise', 'disgust'}
        potential_lies = []
        previous_emotion = None
        for ts, emotion, path in emotions:
            if emotion in deception_cues:
                potential_lies.append((ts, emotion, "Suspicious emotion", path))
            elif previous_emotion and emotion != previous_emotion:
                potential_lies.append((ts, emotion, f"Emotion shift from {previous_emotion}", path))
            previous_emotion = emotion
        return potential_lies

    st.markdown("<h3 style='color: white;'>Upload a video for analysis</h3>", unsafe_allow_html=True)
    uploaded_file = st.markdown("<label style='color: white;'>Upload your .mp4 video</label>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["mp4"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.video(tmp_path)
        st.markdown("<p style='color: white;'>✅ Video uploaded successfully. Running analysis...</p>", unsafe_allow_html=True)

        frame = extract_middle_frame(tmp_path)
        if frame is not None:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            dummy_audio = input_tensor.clone()
            with torch.no_grad():
                output = model(input_tensor, dummy_audio)
                probs = F.softmax(output, dim=1)
                confidence, pred_class = torch.max(probs, dim=1)
                label_map = {0: "🟥 Deceptive", 1: "🟩 Truthful"}
                label = label_map[pred_class.item()]
                score = round(confidence.item(), 4)

            st.markdown("<h3 style='color: white;'>🧠 Prediction Result</h3>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style='padding: 1rem; border-radius: 8px; background-color: rgba(0,0,0,0.6); text-align: center; color: white;'>
                    <h2>{label}</h2>
                    <p>Confidence: <strong>{score}</strong></p>
                </div>
            """, unsafe_allow_html=True)

            if st.checkbox("📊 Show emotion timeline analysis"):
                st.markdown("<p style='color: white;'>Analyzing video frame-by-frame using DeepFace...</p>", unsafe_allow_html=True)
                frames = extract_frames_by_second(tmp_path)
                emotions = analyze_emotions(frames)
                deception_events = detect_deception_moments(emotions)

                df = pd.DataFrame([{'Second': ts, 'Emotion': emotion, 'Flag': flag} for ts, emotion, flag, _ in deception_events])
                st.dataframe(df)

                st.markdown("<h3 style='color: white;'>🎞️ Flagged Frame Thumbnails</h3>", unsafe_allow_html=True)
                for ts, emotion, flag, path in deception_events:
                    st.image(path, caption=f"{ts}s - {emotion} ({flag})", width=300)
        else:
            st.error("❌ Could not extract frame from the video.")

# Routing
if st.session_state.page == "landing":
    show_landing()
elif st.session_state.page == "analyze":
    show_analysis()
