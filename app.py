import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow import keras
import tempfile
import os

# ===============================
# 1. Load Model + Preprocessors
# ===============================
@st.cache_resource
def load_model():
    model = keras.models.load_model("gesture_model.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
    classes = np.load("classes.npy", allow_pickle=True)
    return model, scaler, classes

model, scaler, classes = load_model()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===============================
# 2. Streamlit UI
# ===============================
st.set_page_config(page_title="Emergency Gesture Detector", page_icon="🚨")
st.title("🚨 Emergency Gesture Detection")
st.write("Show your hand gesture to the camera. The app will recognize it in real-time.")

start = st.button("▶ Start Camera")

# ===============================
# 3. Camera Feed + Prediction
# ===============================
if start:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        final_gesture = None
        final_conf = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                landmarks = np.array(landmarks).reshape(1, -1)
                landmarks_scaled = scaler.transform(landmarks)

                pred = model.predict(landmarks_scaled, verbose=0)[0]
                pred_class = np.argmax(pred)
                conf = np.max(pred)

                if conf > final_conf:
                    final_conf = conf
                    final_gesture = classes[pred_class]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Overlay prediction
        if final_gesture:
            cv2.putText(frame, f"{final_gesture} ({final_conf:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # Show frame in Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()
