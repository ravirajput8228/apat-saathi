import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow import keras

# 1. Load model + scaler + label classes
model = keras.models.load_model("gesture_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")
classes = np.load("classes.npy", allow_pickle=True)

# 2. Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 3. Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb_frame)

    final_gesture = None
    final_conf = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 landmarks (x,y)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])  # (x,y) → 42 numbers

            # Convert to numpy array and scale
            landmarks = np.array(landmarks).reshape(1, -1)
            landmarks_scaled = scaler.transform(landmarks)

            # Predict gesture
            pred = model.predict(landmarks_scaled, verbose=0)[0]
            pred_class = np.argmax(pred)
            conf = np.max(pred)
            gesture = classes[pred_class]

            # ✅ PRIORITY: emergency wins if confidence > 0.6
            if gesture == "emergency" and conf > 0.6:
                final_gesture = "emergency"
                final_conf = conf
                break  # stop checking other hands

            # Otherwise keep highest confidence gesture
            if conf > final_conf:
                final_conf = conf
                final_gesture = gesture

            # Draw landmarks for visualization
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Show only ONE final prediction
    if final_gesture:
        cv2.putText(frame, f"Gesture: {final_gesture} ({final_conf:.2f})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    # Show video
    cv2.imshow("Live Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
