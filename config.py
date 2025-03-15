import pyaudio

# Audio parameters
CHUNK = 1024 
FORMAT = pyaudio.paInt16 
CHANNELS = 1 
RATE = 16000
AUDIO_WINDOW = 5 # seconds of audio to analyze for patterns

w1, w2, w3, w4, w5, w6, w7 = 0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1

# Standard MediaPipe face mesh eye landmark indices
# Left eye
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]  # Outline landmarks
LEFT_IRIS_CENTER_APPROX = 159  # Center of left eye (approximate)

# Right eye
RIGHT_EYE_INDICES = [362, 263, 386, 385, 384, 398, 386, 374]  # Outline landmarks
RIGHT_IRIS_CENTER_APPROX = 386  # Center of right eye (approximate)

update_interval = 10 

CAMERA_INDEX = 0
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5
