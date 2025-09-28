import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
import threading

class SimpleGestureController:
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
        
        # Mouse control settings
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 3
        self.sensitivity = 1.5
        
        self.current_gesture = "No Hand Detected"
        self.is_active = False
        
    def smooth_move(self, target_x, target_y):
        """Smooth mouse movement"""
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = target_x, target_y
            
        smooth_x = int(self.prev_x + (target_x - self.prev_x) / self.smoothing)
        smooth_y = int(self.prev_y + (target_y - self.prev_y) / self.smoothing)
        
        pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
        self.prev_x, self.prev_y = smooth_x, smooth_y
    
    def detect_gesture(self, landmarks):
        """Simple gesture detection"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        # Calculate distances
        thumb_index_dist = math.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
        thumb_middle_dist = math.sqrt((middle_tip.x - thumb_tip.x)**2 + (middle_tip.y - thumb_tip.y)**2)
        
        if thumb_index_dist < 0.05:
            return "Left Click"
        elif thumb_middle_dist < 0.05:
            return "Right Click"
        elif index_tip.y < middle_tip.y:
            return "Scroll Mode"
        else:
            return "Cursor Move"
    
    def draw_landmarks_with_colors(self, frame, hand_landmarks):
        """Draw hand landmarks with colors"""
        # Finger tip indices and colors
        tip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]
        
        for idx, landmark in enumerate(hand_landmarks.landmark):
            if idx in tip_indices:
                color_idx = tip_indices.index(idx)
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 8, colors[color_idx], -1)
        
        # Draw hand connections
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
    
    def process_gestures(self, socketio):
        """Main gesture processing loop"""
        self.is_active = True
        self.prev_x, self.prev_y = 0, 0
        
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
                    # Draw landmarks
                    self.draw_landmarks_with_colors(frame, hand_landmarks)
                    
                    # Detect gesture
                    gesture = self.detect_gesture(hand_landmarks.landmark)
                    self.current_gesture = gesture
                    
                    # Execute action
                    if gesture == "Cursor Move":
                        index_tip = hand_landmarks.landmark[8]
                        screen_x = np.interp(index_tip.x, [0.1, 0.9], [0, self.screen_width])
                        screen_y = np.interp(index_tip.y, [0.1, 0.9], [0, self.screen_height])
                        self.smooth_move(screen_x * self.sensitivity, screen_y * self.sensitivity)
                    
                    elif gesture == "Left Click":
                        pyautogui.click()
                        time.sleep(0.5)
                    
                    elif gesture == "Right Click":
                        pyautogui.rightClick()
                        time.sleep(0.5)
                    
                    elif gesture == "Scroll Mode":
                        middle_tip = hand_landmarks.landmark[12]
                        scroll_amount = (middle_tip.y - hand_landmarks.landmark[8].y) * 20
                        pyautogui.scroll(int(scroll_amount))
            
            # Send data to frontend
            socketio.emit('gesture_data', {
                'gesture': gesture,
                'landmarks': landmarks_data
            })
            
            time.sleep(0.05)
        
        # Release camera when stopped
        self.cap.release()

# Global instance
gesture_controller = SimpleGestureController()