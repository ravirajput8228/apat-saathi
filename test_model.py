import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow import keras
import time

# Load model + scaler + label classes
model = keras.models.load_model("gesture_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")
classes = np.load("classes.npy", allow_pickle=True)

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Track emergency start time
emergency_start = None
alert_triggered = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gestures = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 hand landmarks (x,y)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks_scaled = scaler.transform(landmarks)

            pred = model.predict(landmarks_scaled, verbose=0)[0]
            pred_class = np.argmax(pred)
            gesture = str(classes[pred_class]).lower()

            gestures.append(gesture)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Handle emergency detection
    if "fist" in gestures and "open" in gestures:
        if emergency_start is None:
            emergency_start = time.time()  # start timer
        elapsed = time.time() - emergency_start

        if elapsed >= 5 and not alert_triggered:
            cv2.putText(frame, "🚨 CONTROL ROOM ALERTED 🚨",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
            alert_triggered = True
            print("🚨 CONTROL ROOM ALERTED 🚨")
        else:
            cv2.putText(frame, "⚠️ EMERGENCY DETECTED ⚠️",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
    else:
        emergency_start = None
        alert_triggered = False

    cv2.imshow("Live Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

