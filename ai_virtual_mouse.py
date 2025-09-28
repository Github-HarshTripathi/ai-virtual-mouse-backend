# File: backend/ai_virtual_mouse.py - HIGH ACCURACY VERSION

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import threading
import webbrowser
import subprocess
import pyttsx3
import speech_recognition as sr
from flask_socketio import SocketIO
import json
import queue
from enum import Enum
import logging

# Configure PyAutoGUI for GLOBAL control
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

class Gesture(Enum):
    NO_HAND = "NO_HAND"
    CURSOR_MOVE = "CURSOR_MOVE" 
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    SCROLL_UP = "SCROLL_UP"     
    SCROLL_DOWN = "SCROLL_DOWN" 
    DRAG_START = "DRAG_START"
    DRAG_END = "DRAG_END"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    PEACE_SIGN = "PEACE_SIGN"  # NEW: For easy scroll
    THUMBS_UP = "THUMBS_UP"    # NEW: For special actions

class AIVirtualMouse:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # HIGH ACCURACY MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,  # Higher accuracy
            min_tracking_confidence=0.8,   # Higher accuracy
            model_complexity=1             # Better model for accuracy
        )
        
        # Screen properties
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = None
        
        # Enhanced control parameters
        self.smoothing_factor = 4
        self.sensitivity = 1.0
        self.prev_x, self.prev_y = 0, 0
        self.is_dragging = False
        
        # Gesture timing and validation
        self.last_gesture = Gesture.NO_HAND
        self.gesture_hold_frames = 0
        self.gesture_stability_threshold = 3  # Must hold gesture for 3 frames
        self.last_action_time = 0
        self.action_cooldown = 0.4  # Prevent rapid-fire actions
        
        # Click detection improvements
        self.last_click_time = 0
        self.click_cooldown = 0.3
        self.pinch_threshold = 0.04
        self.pinch_release_threshold = 0.06
        self.is_pinching = False
        
        # Scroll improvements
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.3
        self.scroll_gesture_frames = 0
        
        # MediaPipe processing
        self.last_process_time = time.time()
        self.frame_buffer = []
        self.buffer_size = 3
        
        # State management
        self.is_active = False
        self.voice_mode = False
        self.current_gesture = Gesture.NO_HAND
        self.landmark_history = []
        
        # Voice and threading
        self.voice_engine = None
        self.voice_recognizer = None
        self.lock = threading.Lock()
        
        # Initialize components
        self.initialize_camera()
        self.initialize_voice()

    def initialize_camera(self):
        """Initialize camera with HIGH QUALITY settings"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                for i in range(1, 4):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                        
            if self.cap.isOpened():
                # HIGH QUALITY camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 60)  # Higher FPS for better detection
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Better lighting
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
                self.logger.info("HIGH QUALITY Camera initialized successfully")
            else:
                self.logger.error("Failed to initialize camera")
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")

    def initialize_voice(self):
        """Initialize voice components"""
        try:
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 150)
            self.voice_engine.setProperty('volume', 0.7)
            
            self.voice_recognizer = sr.Recognizer()
            self.voice_recognizer.energy_threshold = 300
            self.voice_recognizer.pause_threshold = 0.8
            self.voice_recognizer.dynamic_energy_threshold = True
            
            self.logger.info("Voice components initialized successfully")
        except Exception as e:
            self.logger.error(f"Voice initialization error: {e}")

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2)

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            # Vector from point2 to point1
            v1 = np.array([point1.x - point2.x, point1.y - point2.y])
            # Vector from point2 to point3  
            v2 = np.array([point3.x - point2.x, point3.y - point2.y])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        except:
            return 0.0

    def get_finger_states_advanced(self, landmarks):
        """ADVANCED finger state detection with angle analysis"""
        try:
            finger_states = {
                'thumb': False,
                'index': False, 
                'middle': False,
                'ring': False,
                'pinky': False
            }
            
            # Thumb - check angle and position
            thumb_angle = self.calculate_angle(landmarks[2], landmarks[3], landmarks[4])
            finger_states['thumb'] = thumb_angle > 120 and landmarks[4].x > landmarks[3].x
            
            # Index finger
            index_angle = self.calculate_angle(landmarks[6], landmarks[7], landmarks[8])
            finger_states['index'] = (landmarks[8].y < landmarks[6].y) and index_angle > 140
            
            # Middle finger  
            middle_angle = self.calculate_angle(landmarks[10], landmarks[11], landmarks[12])
            finger_states['middle'] = (landmarks[12].y < landmarks[10].y) and middle_angle > 140
            
            # Ring finger
            ring_angle = self.calculate_angle(landmarks[14], landmarks[15], landmarks[16])
            finger_states['ring'] = (landmarks[16].y < landmarks[14].y) and ring_angle > 140
            
            # Pinky
            pinky_angle = self.calculate_angle(landmarks[18], landmarks[19], landmarks[20])
            finger_states['pinky'] = (landmarks[20].y < landmarks[18].y) and pinky_angle > 140
            
            return finger_states
            
        except Exception as e:
            self.logger.error(f"Finger state detection error: {e}")
            return {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}

    def detect_gesture_advanced(self, landmarks):
        """SUPER ACCURATE gesture detection with multiple validation layers"""
        if not landmarks:
            return Gesture.NO_HAND

        try:
            # Get finger states
            fingers = self.get_finger_states_advanced(landmarks)
            
            # Key landmark points
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            wrist = landmarks[0]
            
            # Calculate distances for pinch detection
            thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
            thumb_middle_dist = self.calculate_distance(thumb_tip, middle_tip)
            index_middle_dist = self.calculate_distance(index_tip, middle_tip)
            
            current_time = time.time()
            
            # 1. PRECISE PINCH GESTURES
            # Left Click - Thumb + Index pinch
            if thumb_index_dist < self.pinch_threshold:
                if not self.is_pinching and current_time - self.last_click_time > self.click_cooldown:
                    self.is_pinching = True
                    self.last_click_time = current_time
                    return Gesture.LEFT_CLICK
            elif thumb_index_dist > self.pinch_release_threshold:
                self.is_pinching = False
                
            # Right Click - Thumb + Middle pinch  
            if thumb_middle_dist < self.pinch_threshold and not self.is_pinching:
                if current_time - self.last_click_time > self.click_cooldown:
                    self.last_click_time = current_time
                    return Gesture.RIGHT_CLICK
            
            # 2. CURSOR MOVEMENT - Only index finger up
            if (fingers['index'] and 
                not fingers['middle'] and 
                not fingers['ring'] and 
                not fingers['pinky'] and
                index_tip.y < wrist.y - 0.1):  # Must be above wrist
                return Gesture.CURSOR_MOVE
            
            # 3. EASY SCROLL - Peace sign (index + middle)
            if (fingers['index'] and 
                fingers['middle'] and 
                not fingers['ring'] and 
                not fingers['pinky'] and
                index_middle_dist > 0.04):  # Fingers must be separated
                
                if current_time - self.last_scroll_time > self.scroll_cooldown:
                    # Determine scroll direction by finger position
                    avg_finger_y = (index_tip.y + middle_tip.y) / 2
                    wrist_y = wrist.y
                    
                    if avg_finger_y < wrist_y - 0.15:  # Fingers well above wrist
                        self.last_scroll_time = current_time
                        return Gesture.SCROLL_UP
                    elif avg_finger_y < wrist_y - 0.05:  # Fingers slightly above wrist
                        self.last_scroll_time = current_time
                        return Gesture.SCROLL_DOWN
                        
                return Gesture.PEACE_SIGN  # Just showing peace, not scrolling
            
            # 4. DRAG MODE - Three fingers (index + middle + ring)
            if (fingers['index'] and 
                fingers['middle'] and 
                fingers['ring'] and
                not fingers['pinky']):
                if not self.is_dragging:
                    return Gesture.DRAG_START
                else:
                    return Gesture.CURSOR_MOVE
            
            # 5. THUMBS UP - Special recognition
            if (fingers['thumb'] and 
                not fingers['index'] and 
                not fingers['middle'] and 
                not fingers['ring'] and
                not fingers['pinky']):
                return Gesture.THUMBS_UP
                
        except Exception as e:
            self.logger.error(f"Advanced gesture detection error: {e}")
            
        return Gesture.NO_HAND

    def validate_gesture(self, new_gesture):
        """Validate gesture stability to prevent false positives"""
        if new_gesture == self.last_gesture:
            self.gesture_hold_frames += 1
        else:
            self.gesture_hold_frames = 0
            self.last_gesture = new_gesture
        
        # Only return gesture if it's been stable for enough frames
        if self.gesture_hold_frames >= self.gesture_stability_threshold:
            return new_gesture
        else:
            return Gesture.NO_HAND

    def smooth_movement(self, target_x, target_y):
        """Enhanced smoothing"""
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = target_x, target_y
            
        smooth_x = self.prev_x + (target_x - self.prev_x) / self.smoothing_factor
        smooth_y = self.prev_y + (target_y - self.prev_y) / self.smoothing_factor
        
        smooth_x = max(0, min(self.screen_width - 1, smooth_x))
        smooth_y = max(0, min(self.screen_height - 1, smooth_y))
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

    def execute_gesture_precise(self, gesture, landmarks, frame_width, frame_height):
        """PRECISE gesture execution with action validation"""
        current_time = time.time()
        
        # Prevent rapid-fire actions
        if current_time - self.last_action_time < self.action_cooldown:
            if gesture not in [Gesture.CURSOR_MOVE, Gesture.NO_HAND]:
                return
        
        try:
            if gesture == Gesture.CURSOR_MOVE and landmarks:
                index_tip = landmarks[8]
                screen_x = np.interp(index_tip.x, [0.1, 0.9], [0, self.screen_width])
                screen_y = np.interp(index_tip.y, [0.1, 0.9], [0, self.screen_height])
                
                x, y = self.smooth_movement(screen_x * self.sensitivity, 
                                          screen_y * self.sensitivity)
                pyautogui.moveTo(x, y, duration=0)
                
            elif gesture == Gesture.LEFT_CLICK:
                pyautogui.click()
                self.last_action_time = current_time
                self.logger.info("✓ LEFT CLICK executed")
                
            elif gesture == Gesture.RIGHT_CLICK:
                pyautogui.rightClick()
                self.last_action_time = current_time
                self.logger.info("✓ RIGHT CLICK executed")
                
            elif gesture == Gesture.SCROLL_UP:
                pyautogui.scroll(3)
                self.last_action_time = current_time
                self.logger.info("✓ SCROLL UP executed")
                
            elif gesture == Gesture.SCROLL_DOWN:
                pyautogui.scroll(-3)
                self.last_action_time = current_time
                self.logger.info("✓ SCROLL DOWN executed")
                
            elif gesture == Gesture.DRAG_START and not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
                self.last_action_time = current_time
                self.logger.info("✓ DRAG STARTED")
                
            elif gesture == Gesture.NO_HAND and self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
                self.logger.info("✓ DRAG ENDED")
                
            elif gesture == Gesture.THUMBS_UP:
                # Special action - maybe show confirmation
                self.logger.info("👍 THUMBS UP detected!")
                
        except Exception as e:
            self.logger.error(f"Gesture execution error: {e}")

    def process_gestures(self, socketio):
        """HIGH ACCURACY gesture processing"""
        self.is_active = True
        frame_count = 0
        
        self.logger.info("🎯 HIGH ACCURACY Virtual Mouse Control Started!")
        
        while self.is_active:
            try:
                if not self.cap or not self.cap.isOpened():
                    self.initialize_camera()
                    time.sleep(1)
                    continue
                    
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process at consistent rate
                current_time = time.time()
                if current_time - self.last_process_time < 0.025:  # 40 FPS max
                    continue
                    
                try:
                    results = self.hands.process(frame_rgb)
                    self.last_process_time = current_time
                except Exception as mp_error:
                    self.logger.warning(f"MediaPipe error: {mp_error}")
                    continue

                h, w, _ = frame.shape
                detected_gesture = Gesture.NO_HAND
                landmarks_data = []

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Enhanced landmark drawing
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=3)
                        )

                        # ADVANCED gesture detection
                        raw_gesture = self.detect_gesture_advanced(hand_landmarks.landmark)
                        
                        # Validate gesture stability
                        validated_gesture = self.validate_gesture(raw_gesture)
                        detected_gesture = validated_gesture
                        self.current_gesture = validated_gesture

                        # Execute gesture with high precision
                        if validated_gesture != Gesture.NO_HAND:
                            self.execute_gesture_precise(validated_gesture, hand_landmarks.landmark, w, h)

                        # Prepare landmarks data
                        landmarks_data = self.prepare_landmarks_data(hand_landmarks.landmark)

                # Send to frontend less frequently to reduce noise
                if frame_count % 5 == 0:
                    try:
                        socketio.emit('gesture_data', {
                            'gesture': detected_gesture.value,
                            'landmarks': landmarks_data,
                            'frame_count': frame_count,
                            'accuracy': 'HIGH',
                            'stability_frames': self.gesture_hold_frames
                        })
                    except Exception as socket_error:
                        self.logger.warning(f"Socket error: {socket_error}")

                frame_count += 1
                time.sleep(0.025)  # Consistent frame rate

            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                time.sleep(0.1)

        # Cleanup
        if self.cap:
            self.cap.release()

    def prepare_landmarks_data(self, landmarks):
        """Prepare enhanced landmarks data"""
        data = []
        finger_names = ["WRIST", "THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        
        try:
            for idx, lm in enumerate(landmarks):
                finger_type = "JOINT"
                if idx in [4, 8, 12, 16, 20]:
                    finger_idx = [4, 8, 12, 16, 20].index(idx) + 1
                    if finger_idx < len(finger_names):
                        finger_type = finger_names[finger_idx] + "_TIP"
                elif idx == 0:
                    finger_type = "WRIST"
                    
                data.append({
                    'x': float(lm.x),
                    'y': float(lm.y), 
                    'z': float(lm.z),
                    'type': finger_type,
                    'visibility': float(getattr(lm, 'visibility', 1.0)),
                    'id': idx
                })
        except Exception as e:
            self.logger.error(f"Landmarks preparation error: {e}")
            
        return data

    def start_voice_recognition(self, socketio):
        """Voice recognition"""
        self.voice_mode = True
        
        while self.voice_mode:
            try:
                if not self.voice_recognizer:
                    time.sleep(1)
                    continue
                    
                with sr.Microphone() as source:
                    self.voice_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    socketio.emit('voice_status', {'status': 'listening'})
                    
                    audio = self.voice_recognizer.listen(source, timeout=10, phrase_time_limit=5)
                    command = self.voice_recognizer.recognize_google(audio, language='en-US').lower()
                    socketio.emit('voice_command', {'command': command})
                    
                    self.execute_voice_command(command, socketio)
                    
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                socketio.emit('voice_status', {'status': 'not_understood'})
            except sr.RequestError as e:
                socketio.emit('voice_status', {'status': 'error', 'message': str(e)})
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Voice error: {e}")
                time.sleep(1)

    def execute_voice_command(self, command, socketio):
        """Execute voice commands"""
        command = command.lower()
        
        try:
            if any(word in command for word in ['open', 'launch']):
                if 'browser' in command:
                    webbrowser.open('https://google.com')
                    self.speak("Opening browser")
                elif 'notepad' in command:
                    subprocess.Popen('notepad.exe')
                    self.speak("Opening Notepad")
                elif 'calculator' in command:
                    subprocess.Popen('calc.exe')
                    self.speak("Opening Calculator")
                    
            elif 'volume' in command:
                if 'up' in command:
                    pyautogui.press('volumeup')
                    self.speak("Volume up")
                elif 'down' in command:
                    pyautogui.press('volumedown')
                    self.speak("Volume down")
                    
        except Exception as e:
            self.logger.error(f"Voice command error: {e}")

    def speak(self, text):
        """Text-to-speech"""
        try:
            if self.voice_engine:
                def speak_async():
                    try:
                        self.voice_engine.say(text)
                        self.voice_engine.runAndWait()
                    except:
                        pass
                threading.Thread(target=speak_async, daemon=True).start()
        except Exception as e:
            self.logger.error(f"TTS error: {e}")

    def cleanup(self):
        """Cleanup resources"""
        self.is_active = False
        self.voice_mode = False
        
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            if self.voice_engine:
                self.voice_engine.stop()
        except:
            pass

# Global instance
ai_mouse = AIVirtualMouse()
