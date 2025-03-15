import mediapipe as mp 
import numpy as np 
import logging

logger = logging.getLogger(__name__)

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose= mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.prev_head_pos = None

    def process_frame(self, frame):
        results = self.pose.process(frame)

    def calculate_head_stability(self, pose_landmarks, prev_head_pos=None):
        """
        Calculate head stability based on how much the head moves
        Higher value means more stable (less movement)
        """
        if pose_landmarks:
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            if self.prev_head_pos is None:
                return 0.9, (nose.x, nose.y)  # Initial stability is high
            
            # Calculate movement from previous position
            movement = np.sqrt((nose.x - self.prev_head_pos[0])**2 + 
                            (nose.y - self.prev_head_pos[1])**2)
            
            # Normalize movement (higher value = more stable)
            stability = 1 - min(1.0, movement * 50)

            self.prev_head_pos = (nose.x. nose.y)
            
            return stability
        
        return 0.5
