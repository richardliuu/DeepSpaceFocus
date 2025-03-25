import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time

# Debug imports
import traceback
import sys 

from scipy.signal import find_peaks

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

#  ======= TKINTER WINDOW STARTUP ============ 

"""import tkinter as tk
from tkinter import ttk, messagebox, Menu, Label, Entry, StringVar 

root = tk.Tk()
root.geometry("")"""

# Set to 0 to avoid messages 
# 2 for warnings 

# Frame 
update_interval = 10
frame_count = 0

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence = 0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)

# Lists to store metrics data
time_stamps = []
head_stability_values = []
face_neutrality_values = []
eye_gaze_values = []
light_change_values = []
concentration_scores = []

# Weights for concentration score calculation
w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25 

# Reference values for normalization
baseline_movement = 0
baseline_samples = 0
baseline_period = 5  # seconds to establish 

start_time = time.time()

# Set up interactive plotting
plt.ion() 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Primary plot for individual metrics
ax1.set_title("Individual Concentration Metrics")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Metric Value")
head_line, = ax1.plot([], [], 'r-', label="Head Stability")
face_line, = ax1.plot([], [], 'g-', label="Face Neutrality")
eye_line, = ax1.plot([], [], 'b-', label="Eye Gaze Stability")
light_line, = ax1.plot([], [], 'y-', label="Light Stability")
ax1.legend(loc="upper right")

# Secondary plot for overall concentration score
ax2.set_title("Overall Concentration Score")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Concentration Score")
concentration_line, = ax2.plot([], [], 'k-', linewidth=2)
ax2.set_ylim(0, 1)

# Standard MediaPipe face mesh eye landmark indices
# Left eye
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]  # Outline landmarks
LEFT_IRIS_CENTER_APPROX = 159  # Center of left eye (approximate)

# Right eye
RIGHT_EYE_INDICES = [362, 263, 386, 385, 384, 398, 386, 374]  # Outline landmarks
RIGHT_IRIS_CENTER_APPROX = 386  # Center of right eye (approximate)

def calculate_face_neutrality(face_landmarks):
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

def calculate_head_stability(pose_landmarks, prev_head_pos=None):
    """
    Calculate head stability based on how much the head moves
    Higher value means more stable (less movement)
    """
    if pose_landmarks:
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        
        if prev_head_pos is None:
            return 0.9, (nose.x, nose.y)  # Initial stability is high
        
        # Calculate movement from previous position
        movement = np.sqrt((nose.x - prev_head_pos[0])**2 + 
                           (nose.y - prev_head_pos[1])**2)
        
        # Normalize movement (higher value = more stable)
        stability = 1 - min(1.0, movement * 50)
        
        return stability, (nose.x, nose.y)
    
    return 0.5, prev_head_pos if prev_head_pos is not None else (0.5, 0.5)

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

def calculate_eye_gaze_stability(face_landmarks, prev_gaze=None):
    """
    Calculate eye gaze stability by tracking eye landmarks
    Higher value means more stable gaze
    """
    # Use standard face mesh eye landmarks instead of iris
    left_eye_center = face_landmarks.landmark[LEFT_IRIS_CENTER_APPROX]
    right_eye_center = face_landmarks.landmark[RIGHT_IRIS_CENTER_APPROX]
    
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

def calculate_angle(a, b, c):
    """
    Calculate angle between three points
    
    Args:
        a, b, c (list or numpy array): Landmark coordinates [x, y]
    
    Returns:
        float: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    return angle if angle <= 180 else 360 - angle

def analyze_upper_body_posture(landmarks):
    """
    Analyze upper body posture focusing on head, neck, and shoulder alignment
    
    Args:
        landmarks (list): MediaPipe pose landmarks
    
    Returns:
        dict: Upper body posture analysis results
    """
    # Key upper body landmarks
    nose = [
        landmarks[mp.solutions.pose.PoseLandmarks.NOSE.value].x,
        landmarks[mp.solutions.pose.PoseLandmarks.NOSE.value].y
    ]
    left_shoulder = [
        landmarks[mp.solutions.pose.PoseLandmarks.LEFT_SHOULDER.value].x,
        landmarks[mp.solutions.pose.PoseLandmarks.LEFT_SHOULDER.value].y
    ]
    right_shoulder = [
        landmarks[mp.solutions.pose.PoseLandmarks.RIGHT_SHOULDER.value].x,
        landmarks[mp.solutions.pose.PoseLandmarks.RIGHT_SHOULDER.value].y
    ]
    left_ear = [
        landmarks[mp.solutions.pose.PoseLandmarks.LEFT_EAR.value].x,
        landmarks[mp.solutions.pose.PoseLandmarks.LEFT_EAR.value].y
    ]
    right_ear = [
        landmarks[mp.solutions.pose.PoseLandmarks.RIGHT_EAR.value].x,
        landmarks[mp.solutions.pose.PoseLandmarks.RIGHT_EAR.value].y
    ]
    
    # Calculate shoulder alignment and head tilt
    shoulder_alignment = calculate_angle(
        [nose[0], left_shoulder[1]], 
        [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2], 
        [nose[0], right_shoulder[1]]
    )
    
    head_tilt = calculate_angle(left_ear, nose, right_ear)
    
    # Posture analysis
    posture_status = {
        'shoulder_alignment': shoulder_alignment,
        'head_tilt': head_tilt,
        'is_aligned': (
            85 <= shoulder_alignment <= 95 and 
            85 <= head_tilt <= 95
        ),
        'tilting_left': head_tilt < 85,
        'tilting_right': head_tilt > 95,
        'shoulders_rotated': abs(shoulder_alignment - 90) > 10
    }
    
    return posture_status

def visualize_upper_body_posture(frame, results):
    """
    Visualize upper body posture landmarks and analysis
    
    Args:
        frame (numpy.ndarray): Input video frame
        results (mediapipe.solutions.pose.PoseResults): MediaPipe pose detection results
    
    Returns:
        numpy.ndarray: Frame with upper body posture visualization
    """
    # Initialize MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Draw all upper body landmarks with specific colors and sizes
        landmark_drawing_spec = [
            (mp_pose.PoseLandmarks.NOSE, (0, 255, 0), 7),  # Green for nose
            (mp_pose.PoseLandmarks.LEFT_SHOULDER, (255, 0, 0), 10),  # Blue for left shoulder
            (mp_pose.PoseLandmarks.RIGHT_SHOULDER, (0, 0, 255), 10),  # Red for right shoulder
            (mp_pose.PoseLandmarks.LEFT_EAR, (255, 255, 0), 5),  # Yellow for left ear
            (mp_pose.PoseLandmarks.RIGHT_EAR, (255, 255, 0), 5)  # Yellow for right ear
        ]
        
        for landmark_type, color, radius in landmark_drawing_spec:
            landmark = results.pose_landmarks.landmark[landmark_type]
            if landmark.visibility > 0.5:  # Only draw if landmark is sufficiently visible
                landmark_point = (
                    int(landmark.x * frame.shape[1]), 
                    int(landmark.y * frame.shape[0])
                )
                cv2.circle(frame, landmark_point, radius, color, -1)
        
        # Draw line connecting shoulders
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmarks.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmarks.RIGHT_SHOULDER]
        
        left_shoulder_point = (
            int(left_shoulder.x * frame.shape[1]), 
            int(left_shoulder.y * frame.shape[0])
        )
        right_shoulder_point = (
            int(right_shoulder.x * frame.shape[1]), 
            int(right_shoulder.y * frame.shape[0])
        )
        
        # Draw shoulder line
        cv2.line(frame, left_shoulder_point, right_shoulder_point, (0, 255, 255), 3)
        
        # Analyze posture
        posture_analysis = analyze_upper_body_posture(results.pose_landmarks.landmark)
        
        # Prepare text for display
        posture_status = (
            "Aligned" if posture_analysis['is_aligned'] else 
            "Tilting Left" if posture_analysis['tilting_left'] else 
            "Tilting Right" if posture_analysis['tilting_right'] else 
            "Rotated Shoulders" if posture_analysis['shoulders_rotated'] else 
            "Misaligned"
        )
        
        # Display posture status and shoulder alignment angle
        cv2.putText(
            frame, 
            f"Posture: {posture_status}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        cv2.putText(
            frame, 
            f"Shoulder Alignment: {posture_analysis['shoulder_alignment']:.2f}°", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
    
    return frame

# Variables to store previous values
prev_head_pos = None
prev_light = None
prev_gaze = None

# To reduce the computational power required
plt.tight_layout()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    elapsed_time = time.time() - start_time

    if face_results.multi_face_landmarks and pose_results.pose_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        pose_landmarks = pose_results.pose_landmarks

        # Calculate all metrics
        face_neutrality = calculate_face_neutrality(face_landmarks)
        head_stability, prev_head_pos = calculate_head_stability(pose_landmarks, prev_head_pos)
        light_stability, prev_light = calculate_light_changes(frame, prev_light)
        eye_stability, prev_gaze = calculate_eye_gaze_stability(face_landmarks, prev_gaze)

        # Calculate overall concentration score
        concentration_score = (w1 * face_neutrality + 
                              w2 * head_stability +  
                              w3 * eye_stability + 
                              w4 * light_stability)

        # Store data
        time_stamps.append(elapsed_time)
        head_stability_values.append(head_stability)
        face_neutrality_values.append(face_neutrality)
        eye_gaze_values.append(eye_stability)
        light_change_values.append(light_stability)
        concentration_scores.append(concentration_score)

        # Draw face landmarks on the image
        # Marking Forehead and Chin
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        forehead_x, forehead_y = int(forehead.x * frame.shape[1]), int(forehead.y * frame.shape[0])
        chin_x, chin_y = int(chin.x * frame.shape[1]), int(chin.y * frame.shape[0])
        
        # Mark eyes using standard face mesh indices
        left_eye = face_landmarks.landmark[LEFT_IRIS_CENTER_APPROX]
        right_eye = face_landmarks.landmark[RIGHT_IRIS_CENTER_APPROX]
        
        left_eye_x, left_eye_y = int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0])
        right_eye_x, right_eye_y = int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0])
        
        # Draw landmarks
        cv2.circle(frame, (forehead_x, forehead_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (chin_x, chin_y), 5, (0, 0, 255), -1)
        cv2.line(frame, (forehead_x, forehead_y), (chin_x, chin_y), (255, 255, 0), 2)
        
        # Draw eye markers
        cv2.circle(frame, (left_eye_x, left_eye_y), 5, (255, 0, 255), -1)
        cv2.circle(frame, (right_eye_x, right_eye_y), 5, (255, 0, 255), -1)
        
        # Add text with metrics
        cv2.putText(frame, f"Concentration: {concentration_score:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Update plots
        if frame_count >= update_interval:
            head_line.set_xdata(time_stamps)
            head_line.set_ydata(head_stability_values)
            
            face_line.set_xdata(time_stamps)
            face_line.set_ydata(face_neutrality_values)
            
            eye_line.set_xdata(time_stamps)
            eye_line.set_ydata(eye_gaze_values)
            
            light_line.set_xdata(time_stamps)
            light_line.set_ydata(light_change_values)

    # Need to graph both audio level and pattern
    # Could seperate the graphs to make it easier to see from the other metrics 
            
            concentration_line.set_xdata(time_stamps)
            concentration_line.set_ydata(concentration_scores)
            
            # Adjust plot limits
            ax1.set_xlim(0, max(10, elapsed_time))
            ax1.set_ylim(0, 1.1)
              
            plt.draw()
            plt.pause(0.01)

            # Resetting the frame counter 
            frame_count = 0 

    # Display the frame
    cv2.imshow("Concentration Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
