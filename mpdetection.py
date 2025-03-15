import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time
import pyaudio
import audioop
import threading 
import queue
import math
import librosa
import scipy.stats

# Debug imports
import traceback
import sys 

from scipy.signal import find_peaks

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Set to 0 to avoid messages 
# 2 for warnings 

# Frame 
update_interval = 10
frame_count = 0

# Audio parameters
CHUNK = 1024 
FORMAT = pyaudio.paInt16 
CHANNELS = 1 
RATE = 16000
AUDIO_WINDOW = 5 # seconds of audio to analyze for patterns

p = pyaudio.PyAudio()

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# Lists to store metrics data
time_stamps = []
head_stability_values = []
face_neutrality_values = []
eye_gaze_values = []
light_change_values = []
task_engagement_values = []
audio_level_values = []
audio_pattern_values = []
concentration_scores = []

# Weights for concentration score calculation
w1, w2, w3, w4, w5, w6, w7 = 0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1

# Reference values for normalization
baseline_movement = 0
baseline_samples = 0
baseline_period = 5  # seconds to establish 

# Reference values for audio processing
audio_queue = queue.Queue()
audio_level = 0.0 
audio_pattern = 0.0
audio_buffers = []
stop_audio = False

start_time = time.time()

# Set up interactive plotting
plt.ion() 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

# Primary plot for individual metrics
ax1.set_title("Individual Concentration Metrics")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Metric Value")
head_line, = ax1.plot([], [], 'r-', label="Head Stability")
face_line, = ax1.plot([], [], 'g-', label="Face Neutrality")
eye_line, = ax1.plot([], [], 'b-', label="Eye Gaze Stability")
task_line, = ax1.plot([], [], 'c-', label="Task Engagement")
light_line, = ax1.plot([], [], 'y-', label="Light Stability")
ax1.legend(loc="upper right")

# Secondary plot for overall concentration score
ax2.set_title("Overall Concentration Score")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Concentration Score")
concentration_line, = ax2.plot([], [], 'k-', linewidth=2)
ax2.set_ylim(0, 1)

ax3.set_title("Audio Metrics")
ax3.set_xlabel("Time (Seconds)")
ax3.set_ylabel(("Metric Values"))
audio_volume_line, = ax3.plot([], [], 'm-', label="Audio Environment")
audio_pattern_line, = ax3.plot([], [], 'g-', label="Audio Pattern")
ax3.legend(loc="upper right")

def audio_callback(in_data, frame_count, time_info, status):
    # Audio streaming through a call back function
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

def analyze_audio_patterns(audio_buffer, sample_rate):
    """
    Analyzing the audio for patterns that indicate concentration 
    through returning a score between 0-1 (normalized)
    """

    if isinstance(audio_buffer, bytes):
        samples = np.frombuffer(audio_buffer, dtype=np.int16)

    else:
        samples = audio_buffer
    
    # Normalizing
    samples = samples.astype(np.float32) / 32768.0

    # Audio Features
    # Checking for high or sharp sounds
    if len(samples) > 512:
        spec_cent = librosa.feature.spectral_centroid(y=samples, sr=sample_rate)[0]
        spec_cent_norm = np.mean(spec_cent) / 4000 # Normalizing
        spec_score = 1 - min(1.0, spec_cent_norm)
    else:
        spec_score = 0.5 


    # Checking for rhythm regularity through autocorrelation (measures relationship of variables with a lagged value)
    if len(samples) > 1024:
        corr = np.correlate(samples, samples, mode='full')
        corr = corr[len(corr)//2:]

        peaks, _ = find_peaks(corr, height=0.1*np.max(corr))

        # Checking for the peaks of regular rhythm which should be even
        if len(peaks) > 2:
            peak_intervals = np.diff(peaks)
            rhythm_regularity = 1 - np.std(peak_intervals) / np.mean(peak_intervals)
            rhythm_score = max(0, min(1, rhythm_regularity))
        else:
            rhythm_score = 0.5
    else:
        rhythm_score = 0.5

    # Third audio feature
    # Sound consistency 
    amplitude_envelope = np.abs(samples)
    amp_std = np.std(amplitude_envelope)
    consistency_score = 1 - min(1.0, amp_std * 10)

    # Combining the scores
    return 0.3 * spec_score + 0.3 * rhythm_score + 0.4 * consistency_score

def process_audio():
    # Processing audio in the background
    global audio_level, audio_pattern, audio_buffers, stop_audio

    # Debug statement
    print("Audio processing thread started")

    while not stop_audio:
        try:
            data = audio_queue.get(timeout=1)

            # Debug statement 
            print(f"Got audio data: {len(data)} bytes, queue size: {audio_queue.qsize()}")

            # Volume level
            rms = audioop.rms(data, 2)
            if rms > 0:
                decibel = 20 * math.log10(rms)

                # Debug statement
                print(f"Raw RMS: {rms}, Decibel: {decibel}")

                # Normalize for the 0-1 range (avg speech = 40-60db)
                normalized_db = max(0, min(1.0, (decibel-10) / 40))

            else:
                normalized_db = 0

            audio_level = normalized_db

            # Storing audio data to allow for pattern analysis
            audio_buffers.append(data)

            max_buffers = int(RATE * AUDIO_WINDOW / CHUNK)

            if len(audio_buffers) > max_buffers:
                audio_buffers.pop(0)

            if len(audio_buffers) > 5:
                all_samples = np.frombuffer(b''.join(audio_buffers), dtype=np.int16)

                audio_pattern = analyze_audio_patterns(all_samples, RATE)

                # Classifying the audio environment through volume
                if audio_level < 0.2:
                    environment = "Quiet"
                elif audio_level < 0.5:
                    environment = "Moderate"
                else:
                    environment = "Noisy"

                if audio_pattern > 0.7:
                    pattern_type = "Consistent"
                elif audio_pattern > 0.4:
                    pattern_type = "Somewhat variable"
                else:
                    pattern_type = "Unstable"

        except Exception as e:
            print("Thread error:", e)
            traceback.print_exc(file=sys.stdout)
            
            # Debug Statement 
            print(f"Processed audio: level={audio_level:.2f}, pattern={audio_pattern:.2f}")

            audio_queue.task_done()
        except queue.Empty:
            continue

"""Low volume generally better for concentration while silence could also mean AFK"""
def calculate_audio_concentration(audio_level, audio_pattern):
    volume_score = 1.0 - audio_level if audio_level > 0.1 else 0.8 

    # Consistent audio patterns are better for concentration
    pattern_score = audio_pattern
    
    # Combine scores
    return (volume_score * 0.4) + (pattern_score * 0.6)

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

def calculate_task_engagement(face_landmarks, head_stability, eye_stability):
    """
    Estimate task engagement based on head direction and eye gaze
    Higher value means more engaged
    """
    # Use a combination of head stability and eye gaze stability
    # Also consider the forward-facing position of the head
    
    # Nose tip and forehead landmarks
    nose_tip = face_landmarks.landmark[4]
    forehead = face_landmarks.landmark[10]
    
    # Measure how forward-facing the face is
    face_direction = forehead.z - nose_tip.z
    forward_factor = max(0, min(1, 1 - abs(face_direction) * 5))
    
    # Combine factors
    engagement = (head_stability * 0.4 + 
                  eye_stability * 0.4 + 
                  forward_factor * 0.2)
    
    return max(0.3, min(1.0, engagement))

# Variables to store previous values
prev_head_pos = None
prev_light = None
prev_gaze = None



stream = p.open(format=FORMAT, 
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

audio_thread = threading.Thread(target=process_audio)
audio_thread.daemon = True
audio_thread.start()

stream.start_stream() 

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
        task_engagement = calculate_task_engagement(face_landmarks, head_stability, eye_stability)

        # Audio metrics
        audio_concentration = calculate_audio_concentration(audio_level, audio_pattern)

        # Calculate overall concentration score
        concentration_score = (w1 * face_neutrality + 
                              w2 * head_stability + 
                              w3 * task_engagement + 
                              w4 * eye_stability + 
                              w5 * light_stability +
                              w6 * (1 - audio_level) +
                              w7 * audio_pattern)

        # Store data
        time_stamps.append(elapsed_time)
        head_stability_values.append(head_stability)
        face_neutrality_values.append(face_neutrality)
        eye_gaze_values.append(eye_stability)
        light_change_values.append(light_stability)
        task_engagement_values.append(task_engagement)
        audio_level_values.append(1 - audio_level)
        audio_pattern_values.append(audio_pattern)
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
            
            task_line.set_xdata(time_stamps)
            task_line.set_ydata(task_engagement_values)
            
            light_line.set_xdata(time_stamps)
            light_line.set_ydata(light_change_values)

    # Need to graph both audio level and pattern
    # Could seperate the graphs to make it easier to see from the other metrics 

            audio_volume_line.set_xdata(time_stamps)
            audio_volume_line.set_ydata(audio_level_values)
            audio_pattern_line.set_xdata(time_stamps)
            audio_pattern_line.set_ydata(audio_pattern_values)
            
            concentration_line.set_xdata(time_stamps)
            concentration_line.set_ydata(concentration_scores)
            
            # Adjust plot limits
            ax1.set_xlim(0, max(10, elapsed_time))
            ax1.set_ylim(0, 1.1)
            
            ax2.set_xlim(0, max(10, elapsed_time))
            ax3.set_xlim(0, max(10, elapsed_time))
            
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
