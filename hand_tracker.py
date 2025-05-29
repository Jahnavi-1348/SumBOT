import cv2
import mediapipe as mp
import threading
import tkinter as tk

# Hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

# Control flag
running = True

def detect_hand():
    global running
    cap = cv2.VideoCapture(0)

    while running and cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                finger_status = []
                if lm_list:
                    if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                        finger_status.append(1)
                    else:
                        finger_status.append(0)

                    for id in range(1, 5):
                        if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                            finger_status.append(1)
                        else:
                            finger_status.append(0)

                    print("Fingers (T, I, M, R, P):", finger_status)

                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Detection", img)
        if cv2.waitKey(10) & 0xFF == 27:  # ESC key also closes
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
root.title("Hand Detection Control")

start_btn = tk.Button(root, text="Start Webcam", command=start_detection)
start_btn.pack(pady=10)

stop_btn = tk.Button(root, text="Stop Webcam", command=stop_detection)
stop_btn.pack(pady=10)

root.mainloop()
