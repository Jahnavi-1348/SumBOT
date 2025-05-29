
import cv2
import mediapipe as mp
import threading
import tkinter as tk
import numpy as np
import tensorflow as tf
import pyttsx3

# Load trained model and class names
model = tf.keras.models.load_model(r"C:/Users/janur/OneDrive/Desktop/project sumbot/model/asl_alphabet_model.h5")
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Speech function
def speak_letter(letter):
    engine = pyttsx3.init()
    engine.setProperty('volume', 0.9)
    engine.say(f"The letter is {letter}")
    engine.runAndWait()

# Hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Control flag
running = False
img_height, img_width = 64, 64


def detect_hand():
    global running
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW ensures compatibility on Windows

    if not cap.isOpened():
        print("Webcam could not be opened.")
        return

    running = True

    while running:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                # Save image of the hand
                hand_img = cv2.resize(img, (img_width, img_height))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # Make prediction
                predictions = model.predict(hand_img, verbose=0)
                predicted_index = np.argmax(predictions)
                predicted_label = class_names[predicted_index]

                # Show label and speak
                cv2.putText(img, predicted_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                speak_letter(predicted_label)

        cv2.imshow("ASL Detection", img)
        if cv2.waitKey(10) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

def start_detection():
    threading.Thread(target=detect_hand).start()

def stop_detection():
    global running
    running = False

# Simple GUI
root = tk.Tk()
root.title("ASL Hand Detection & Speech")

start_btn = tk.Button(root, text="Start Webcam", command=start_detection)
start_btn.pack(pady=10)

stop_btn = tk.Button(root, text="Stop Webcam", command=stop_detection)
stop_btn.pack(pady=10)

root.mainloop()
