# File: backend/advanced_gesture_controller.py
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from threading import Thread, Lock
import webbrowser
import subprocess
import pyttsx3
import speech_recognition as sr
from flask_socketio import SocketIO, emit
import json

class AdvancedGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        
        # Gesture state variables
        self.is_dragging = False
        self.scroll_mode = False
        self.volume_control = False
        self.brightness_control = False
        self.last_gesture = "None"
        self.lock = Lock()
        
        # Voice assistant
        self.voice_engine = pyttsx3.init()
        self.voice_recognizer = sr.Recognizer()
        self.voice_mode = False
        
    def calculate_distance(self, point1, point2):
        return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)
    
    def map_coordinates(self, x, y, frame_width, frame_height):
        # Smooth coordinate mapping with boundary checks
        screen_x = np.interp(x, [0.1, 0.9], [0, self.screen_width])
        screen_y = np.interp(y, [0.1, 0.9], [0, self.screen_height])
        
        # Clamp values to screen boundaries
        screen_x = max(0, min(self.screen_width, screen_x))
        screen_y = max(0, min(self.screen_height, screen_y))
        
        return int(screen_x), int(screen_y)
    
    def detect_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        # Get key points
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        # Calculate distances
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        thumb_middle_dist = self.calculate_distance(thumb_tip, middle_tip)
        thumb_ring_dist = self.calculate_distance(thumb_tip, ring_tip)
        thumb_pinky_dist = self.calculate_distance(thumb_tip, pinky_tip)
        
        # Gesture recognition logic based on research papers
        if thumb_index_dist < 0.05:
            return "left_click"
        elif thumb_middle_dist < 0.05:
            return "right_click"
        elif thumb_pinky_dist > 0.15:
            return "volume_up"
        elif thumb_pinky_dist < 0.08:
            return "volume_down"
        elif thumb_ring_dist < 0.05:
            return "brightness_control"
        elif index_tip.y < middle_tip.y and abs(index_tip.x - middle_tip.x) < 0.05:
            return "scroll_mode"
        elif index_tip.y < wrist.y:
            return "cursor_move"
        
        return "none"
    
    def execute_gesture_action(self, gesture, landmarks, frame_width, frame_height):
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        if gesture == "cursor_move":
            x, y = self.map_coordinates(index_tip.x, index_tip.y, frame_width, frame_height)
            pyautogui.moveTo(x, y, duration=0.1)
            
        elif gesture == "left_click":
            pyautogui.click()
            time.sleep(0.3)  # Prevent multiple clicks
            
        elif gesture == "right_click":
            pyautogui.rightClick()
            time.sleep(0.3)
            
        elif gesture == "scroll_mode":
            scroll_speed = (middle_tip.y - index_tip.y) * 10
            pyautogui.scroll(int(scroll_speed * 50))
            
        elif gesture == "volume_up":
            pyautogui.press('volumeup')
            
        elif gesture == "volume_down":
            pyautogui.press('volumedown')
            
        elif gesture == "brightness_control":
            # Windows brightness control (requires additional permissions)
            pass
    
    def process_voice_command(self, command):
        command = command.lower()
        
        if 'open browser' in command or 'chrome' in command:
            webbrowser.open('https://www.google.com')
            self.speak("Opening browser")
            
        elif 'search for' in command:
            query = command.replace('search for', '').strip()
            webbrowser.open(f'https://www.google.com/search?q={query}')
            self.speak(f"Searching for {query}")
            
        elif 'notepad' in command:
            subprocess.Popen('notepad.exe')
            self.speak("Opening Notepad")
            
        elif 'calculator' in command:
            subprocess.Popen('calc.exe')
            self.speak("Opening Calculator")
            
        elif 'volume up' in command:
            pyautogui.press('volumeup')
            self.speak("Volume increased")
            
        elif 'volume down' in command:
            pyautogui.press('volumedown')
            self.speak("Volume decreased")
            
        elif 'mute' in command:
            pyautogui.press('volumemute')
            self.speak("Volume muted")
            
        else:
            self.speak("Command not recognized")
    
    def speak(self, text):
        self.voice_engine.say(text)
        self.voice_engine.runAndWait()
    
    def start_gesture_recognition(self, socketio):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            h, w, _ = frame.shape
            current_gesture = "None"
            landmarks_data = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Detect gesture
                    gesture = self.detect_gesture(hand_landmarks)
                    current_gesture = gesture
                    
                    # Execute action
                    self.execute_gesture_action(gesture, hand_landmarks.landmark, w, h)
                    
                    # Prepare landmarks for frontend
                    for lm in hand_landmarks.landmark:
                        landmarks_data.append({'x': lm.x, 'y': lm.y})
            
            # Send data to frontend via WebSocket
            socketio.emit('gesture_data', {
                'gesture': current_gesture,
                'landmarks': landmarks_data
            })
            
            # Add small delay to prevent high CPU usage
            time.sleep(0.03)