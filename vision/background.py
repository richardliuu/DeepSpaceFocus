import cv2
import mediapipe as mp

class Background:
    def __init__(self):
        self.current_light = current_light

    def calculate_light_changes(frame, prev_light=None):
        """
        Calculate light stability
        Higher value means more stable lighting
        """
        current_light = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        if prev_light is None:
            return 0.9, current_light  # Initial stability is high
        
        # Calculate change in lighting
        light_diff = abs(current_light - prev_light) / 255.0
        
        # Normalize (higher value = more stable)
        stability = 1 - min(1.0, light_diff * 10)
        
        return stability, current_light
