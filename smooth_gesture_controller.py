import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
import webbrowser
import subprocess
import pyttsx3
import speech_recognition as sr

class SmoothGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        self.prev_x, self.prev_y = 0, 0

        # Fixes: slower/smoother movement for cursor
        self.smoothing_factor = 7
        self.mouse_sensitivity = 0.7

        self.voice_recognizer = sr.Recognizer()
        self.voice_mode = False
        self.last_voice_command = ""
        self.current_gesture = "No Hand Detected"
        self.is_active = False

    def smooth_mouse_move(self, target_x, target_y):
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = target_x, target_y
        smooth_x = self.prev_x + (target_x - self.prev_x) / self.smoothing_factor
        smooth_y = self.prev_y + (target_y - self.prev_y) / self.smoothing_factor
        pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
        self.prev_x, self.prev_y = smooth_x, smooth_y

    def get_finger_distance(self, point1, point2):
        return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)

    def detect_advanced_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]

        thumb_index_dist = self.get_finger_distance(thumb_tip, index_tip)
        thumb_middle_dist = self.get_finger_distance(thumb_tip, middle_tip)
        index_middle_dist = self.get_finger_distance(index_tip, middle_tip)

        if thumb_index_dist < 0.03:
            return "Left Click"
        elif thumb_middle_dist < 0.03:
            return "Right Click"
        elif index_middle_dist < 0.02 and index_tip.y < middle_tip.y:
            return "Scroll Mode"
        elif thumb_index_dist > 0.1 and index_tip.y < wrist.y:
            return "Cursor Move"
        elif thumb_index_dist < 0.05 and index_tip.y < wrist.y:
            return "Drag Mode"
        return "Hand Detected"

    def process_gestures(self, socketio):
        self.is_active = True
        while self.is_active:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            h, w, _ = frame.shape
            gesture = "No Hand Detected"
            landmarks_data = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.draw_landmarks_with_colors(frame, hand_landmarks)
                    gesture = self.detect_advanced_gesture(hand_landmarks.landmark)
                    self.current_gesture = gesture
                    self.execute_gesture_action(gesture, hand_landmarks.landmark, w, h)
                    landmarks_data = self.prepare_landmarks_data(hand_landmarks.landmark)
            socketio.emit('gesture_data', {
                'gesture': gesture,
                'landmarks': landmarks_data
            })
            time.sleep(0.05)

    def draw_landmarks_with_colors(self, frame, hand_landmarks):
        landmark_colors = {
            4: (0, 0, 255),    # Thumb - Red
            8: (0, 255, 0),    # Index - Green
            12: (255, 0, 0),   # Middle - Blue
            16: (255, 255, 0), # Ring - Cyan
            20: (255, 0, 255)  # Pinky - Magenta
        }
        for idx, landmark in enumerate(hand_landmarks.landmark):
            if idx in landmark_colors:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 8, landmark_colors[idx], -1)
                cv2.putText(frame, str(idx), (x-10, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def prepare_landmarks_data(self, landmarks):
        data = []
        for idx, lm in enumerate(landmarks):
            data.append({
                'x': lm.x,
                'y': lm.y, 
                'z': lm.z,
                'finger': self.get_finger_name(idx)
            })
        return data

    def get_finger_name(self, landmark_index):
        finger_names = {
            4: "Thumb Tip",
            8: "Index Tip",
            12: "Middle Tip",
            16: "Ring Tip",
            20: "Pinky Tip"
        }
        return finger_names.get(landmark_index, f"Point_{landmark_index}")

    def execute_gesture_action(self, gesture, landmarks, frame_width, frame_height):
        index_tip = landmarks[8]
        if gesture == "Cursor Move":
            screen_x = np.interp(index_tip.x, [0.1, 0.9], [0, self.screen_width]) * self.mouse_sensitivity
            screen_y = np.interp(index_tip.y, [0.1, 0.9], [0, self.screen_height]) * self.mouse_sensitivity
            screen_x = max(0, min(self.screen_width, screen_x))
            screen_y = max(0, min(self.screen_height, screen_y))
            self.smooth_mouse_move(screen_x, screen_y)
        elif gesture == "Left Click":
            pyautogui.click()
            time.sleep(0.5)
        elif gesture == "Right Click":
            pyautogui.rightClick()
            time.sleep(0.5)
        elif gesture == "Scroll Mode":
            middle_tip = landmarks[12]
            scroll_amount = (middle_tip.y - index_tip.y) * 20
            pyautogui.scroll(int(scroll_amount))

    def start_voice_recognition(self, socketio):
        self.voice_mode = True
        while self.voice_mode:
            try:
                with sr.Microphone() as source:
                    self.voice_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.voice_recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    command = self.voice_recognizer.recognize_google(audio).lower()
                    self.last_voice_command = command
                    socketio.emit('voice_command', {'command': command})
                    # Listen for hotword, activate gesture mode automatically
                    if "start vimouse" in command or "start camera" in command or "start mouse" in command:
                        self.is_active = True
                    elif "stop camera" in command or "stop mouse" in command:
                        self.is_active = False
                    # More voice actions below
                    self.execute_voice_command(command)
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception as e:
                print(f"Voice recognition error: {e}")
                time.sleep(1)

    def execute_voice_command(self, command):
        command = command.lower()
        if 'open' in command:
            if 'browser' in command:
                webbrowser.open('https://google.com')
            elif 'notepad' in command:
                subprocess.Popen('notepad.exe')
        elif 'volume up' in command:
            pyautogui.press('volumeup')
        elif 'volume down' in command:
            pyautogui.press('volumedown')
        elif 'mute' in command:
            pyautogui.press('volumemute')
