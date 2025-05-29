import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import mediapipe as mp
import time

# Load model
model = tf.keras.models.load_model('C:/Users/janur/OneDrive/Desktop/project sumbot/model/asl_alphabet_model.h5')

# Labels (update if yours differ)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

prev_letter = ""
last_spoken_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Extract ROI
        x1, y1, x2, y2 = 100, 100, 324, 324
        roi = frame[y1:y2, x1:x2]

        # Preprocess ROI
        img = cv2.resize(roi, (64, 64))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img)
        class_id = np.argmax(pred[0])
        letter = labels[class_id]

        # Display prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Letter: {letter}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Speak only if new letter and pause of 1.5 sec
        if letter != prev_letter and time.time() - last_spoken_time > 1.5:
            engine.say(f"The letter is {letter}")
            engine.runAndWait()
            prev_letter = letter
            last_spoken_time = time.time()

    cv2.imshow("ASL to Speech", frame)
    if cv2.waitKey(10) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

