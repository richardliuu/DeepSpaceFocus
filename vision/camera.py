import cv2
import logging 

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera at index {self.camera_index}")
            return False
        return True
    
    def read_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return None
            return frame
        return None
    
    def release(self):
        if self.cap:
            self.cap.release()
        