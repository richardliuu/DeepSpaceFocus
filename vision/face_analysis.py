import cv2 
import mediapipe as mp
import numpy as np
import logging 
import config 

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE
        )
        self.prev_gaze = None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results 
    
    def calculate_face_neutrality(self, face_landmarks):
        """
        Calculate face neutrality based on landmark positions
        Lower values mean more neutral expression
        """
        # Measure the deviation of mouth corners and eyebrows from neutral position
        # Mouth landmarks
        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]
        
        # Eyebrow landmarks
        left_eyebrow = face_landmarks.landmark[65]
        right_eyebrow = face_landmarks.landmark[295]
        
        # Calculate vertical distances
        mouth_height = abs(left_mouth.y - right_mouth.y)
        eyebrow_height = abs(left_eyebrow.y - right_eyebrow.y)
        
        # Normalize and invert (so higher value = more neutral)
        neutrality = 1 - (mouth_height + eyebrow_height)
        
        # Clamp between 0.5 and 1.0 for better visualization
        return max(0.5, min(1.0, neutrality * 3))

# Might need to self.prev_gaze 
def calculate_eye_gaze_stability(self, face_landmarks, prev_gaze=None):
    """
    Calculate eye gaze stability by tracking eye landmarks
    Higher value means more stable gaze
    """
    # Use standard face mesh eye landmarks instead of iris
    left_eye_center = face_landmarks.landmark[config.LEFT_IRIS_CENTER_APPROX]
    right_eye_center = face_landmarks.landmark[config.RIGHT_IRIS_CENTER_APPROX]
    
    # Eye corner landmarks
    left_eye_left = face_landmarks.landmark[33]
    left_eye_right = face_landmarks.landmark[133]
    right_eye_left = face_landmarks.landmark[362]
    right_eye_right = face_landmarks.landmark[263]
    
    # Calculate normalized eye center positions within the eyes
    left_ratio_x = (left_eye_center.x - left_eye_left.x) / (left_eye_right.x - left_eye_left.x)
    right_ratio_x = (right_eye_center.x - right_eye_left.x) / (right_eye_right.x - right_eye_left.x)
    
    # Average gaze position
    avg_gaze = (left_ratio_x + right_ratio_x) / 2
    
    if prev_gaze is None:
        return 0.9, avg_gaze  # Initial stability is high
    
    # Calculate gaze movement
    gaze_movement = abs(avg_gaze - prev_gaze)
    
    # Normalize (higher value = more stable)
    stability = 1 - min(1.0, gaze_movement * 5)
    
    return stability, avg_gaze

