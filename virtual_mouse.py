# virtual_mouse_server.py
from flask import Flask, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import pyautogui
import math
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- MediaPipe and PyAutoGUI setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

click_down = False
right_click_down = False
scroll_mode = False
scroll_prev_y = 0
last_scroll_time = time.time()
scroll_cooldown = 0.05

def smooth_move(curr_x, curr_y, target_x, target_y, factor=3):
    new_x = int(curr_x + (target_x - curr_x) / factor)
    new_y = int(curr_y + (target_y - curr_y) / factor)
    return new_x, new_y

# --- Run virtual mouse in a separate thread ---
def virtual_mouse_loop():
    global click_down, right_click_down, scroll_mode, scroll_prev_y, last_scroll_time
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]

                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)

                # Mouse movement
                mouse_x = int(index_tip.x * screen_width)
                mouse_y = int(index_tip.y * screen_height)
                curr_mouse_x, curr_mouse_y = pyautogui.position()
                smooth_x, smooth_y = smooth_move(curr_mouse_x, curr_mouse_y, mouse_x, mouse_y)
                pyautogui.moveTo(smooth_x, smooth_y)

                # Distances
                dist_thumb_index = math.hypot(index_x - thumb_x, index_y - thumb_y)
                dist_middle_thumb = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
                dist_index_middle = math.hypot(index_x - middle_x, index_y - middle_y)

                # Left click
                if dist_thumb_index < 45 and not click_down:
                    click_down = True
                    pyautogui.click()
                elif dist_thumb_index >= 45:
                    click_down = False

                # Right click
                if dist_middle_thumb < 45 and not right_click_down:
                    right_click_down = True
                    pyautogui.rightClick()
                elif dist_middle_thumb >= 45:
                    right_click_down = False

                # Scroll
                if dist_index_middle < 70:
                    if not scroll_mode:
                        scroll_mode = True
                        scroll_prev_y = (index_y + middle_y) // 2
                    else:
                        curr_y = (index_y + middle_y) // 2
                        diff = scroll_prev_y - curr_y
                        if abs(diff) > 5 and (time.time() - last_scroll_time) > scroll_cooldown:
                            pyautogui.scroll(int(diff * 2))
                            scroll_prev_y = curr_y
                            last_scroll_time = time.time()
                else:
                    scroll_mode = False

        # Optional: Show camera feed
        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the virtual mouse in a separate thread
threading.Thread(target=virtual_mouse_loop, daemon=True).start()

# --- Flask endpoints ---
@app.route("/start_gesture", methods=["GET"])
def start_gesture():
    return jsonify({"status": "started"})

@app.route("/stop_gesture", methods=["GET"])
def stop_gesture():
    return jsonify({"status": "stopped"})

# Start the Flask + SocketIO server
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)




