import matplotlib
matplotlib.use('TkAgg')

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import Menu, Label, Entry, StringVar
from tkinter import ttk, messagebox

# Global flag for monitoring
run_monitoring = False
monitoring_thread = None

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def calculate_face_neutrality(face_landmarks):
    """
    Calculate face neutrality based on landmark positions
    Lower values mean more neutral expression
    """
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
    # Standard face mesh eye landmark indices
    LEFT_IRIS_CENTER_APPROX = 159
    RIGHT_IRIS_CENTER_APPROX = 386

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

def analyze_upper_body_posture(landmarks):
    """
    Enhanced posture stability analysis
    
    Args:
        landmarks (list): MediaPipe pose landmarks
    
    Returns:
        tuple: (Posture stability score, Neck angle)
    """
    try:
        # Validate landmark visibility
        landmark_visibility = [
            landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].visibility,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].visibility,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value].visibility
        ]
        
        # Check if any critical landmarks are not visible
        if any(vis < 0.5 for vis in landmark_visibility):
            return 0.5, 0  # Return moderate stability if landmarks are unclear
        
        # Key landmarks
        nose = [
            landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].x,
            landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].y
        ]
        left_shoulder = [
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        right_shoulder = [
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ]
        left_ear = [
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].x,
            landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value].y
        ]
        right_ear = [
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value].x,
            landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value].y
        ]
        
        # Calculate shoulder midpoint
        shoulder_midpoint = [
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        ]
        
        # Calculate horizontal levelness of shoulders
        shoulder_level_deviation = abs(left_shoulder[1] - right_shoulder[1])
        
        # Calculate head alignment relative to shoulders
        vertical_alignment_deviation = abs(nose[0] - shoulder_midpoint[0])
        
        # Calculate head tilt
        head_tilt = abs(calculate_angle(left_ear, nose, right_ear) - 90)
        
        def calculate_neck_angle():
            # Calculate the angle between the vertical line and the neck line
            vertical_reference = [shoulder_midpoint[0], shoulder_midpoint[1] - 1]
            
            # Calculate angle using vectors
            neck_vector = [nose[0] - shoulder_midpoint[0], nose[1] - shoulder_midpoint[1]]
            vertical_vector = [vertical_reference[0] - shoulder_midpoint[0], 
                               vertical_reference[1] - shoulder_midpoint[1]]
            
            # Use dot product to find angle
            dot_product = neck_vector[0]*vertical_vector[0] + neck_vector[1]*vertical_vector[1]
            magnitude_neck = np.sqrt(neck_vector[0]**2 + neck_vector[1]**2)
            magnitude_vertical = np.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)
            
            # Prevent division by zero
            if magnitude_neck == 0 or magnitude_vertical == 0:
                return 0
            
            cos_angle = dot_product / (magnitude_neck * magnitude_vertical)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
            
            return angle
        
        def stability_score(deviation, max_deviation):
            # Inverse exponential to give more penalty for larger deviations
            return max(0, 1 - (deviation / max_deviation) ** 2)
        
        # Define maximum acceptable deviations
        MAX_SHOULDER_LEVEL = 0.05  # 5% vertical difference
        MAX_VERTICAL_ALIGNMENT = 0.1  # 10% horizontal offset
        MAX_HEAD_TILT = 10  # 10 degrees
        MAX_NECK_ANGLE = 15  # 15 degrees
        
        # Calculate stability components
        shoulder_level_stability = stability_score(shoulder_level_deviation, MAX_SHOULDER_LEVEL)
        vertical_alignment_stability = stability_score(vertical_alignment_deviation, MAX_VERTICAL_ALIGNMENT)
        head_tilt_stability = stability_score(head_tilt, MAX_HEAD_TILT)
        
        # Calculate neck angle
        neck_angle = calculate_neck_angle()
        neck_stability = stability_score(neck_angle, MAX_NECK_ANGLE)
        
        # Combined posture stability
        posture_stability = (
            shoulder_level_stability * 0.3 +
            vertical_alignment_stability * 0.2 +
            head_tilt_stability * 0.2 +
            neck_stability * 0.3
        )
        
        # Ensure the score is between 0 and 1
        posture_stability = max(0, min(1, posture_stability))
        
        return posture_stability, neck_angle
    
    except Exception as e:
        print(f"Error in posture analysis: {e}")
        return 0.5, 0

def calculate_angle(a, b, c):
    """
    Calculate angle between three points
    Args:
        a, b, c (list or numpy array): Landmark coordinates [x, y]
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    return angle if angle <= 180 else 360 - angle

def detect_hunching(landmarks):
    """
    Detect hunching by analyzing the relative positions of shoulders, spine, and head
    
    Args:
        landmarks (list): MediaPipe pose landmarks
    
    Returns:
        tuple: (hunching_score, diagnostic_details)
    """
    try:
        # Key landmarks for hunching detection
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        
        # Check visibility of critical landmarks
        visibility_check = [
            nose.visibility, 
            left_shoulder.visibility, 
            right_shoulder.visibility,
            left_hip.visibility, 
            right_hip.visibility
        ]
        
        # Reject if any critical landmark is poorly visible
        if any(vis < 0.5 for vis in visibility_check):
            return 0, {"error": "Insufficient landmark visibility"}
        
        # Calculate shoulder and hip midpoints
        shoulder_midpoint = [
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ]
        hip_midpoint = [
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ]
        
        # Calculate vertical alignment
        # Ideal spine should be relatively straight
        spine_angle = np.arctan2(
            shoulder_midpoint[1] - hip_midpoint[1],
            shoulder_midpoint[0] - hip_midpoint[0]
        ) * 180 / np.pi
        
        # Calculate head-shoulder relationship
        # How far forward is the head compared to shoulders
        head_forward_deviation = nose.x - shoulder_midpoint[0]
        head_vertical_deviation = nose.y - shoulder_midpoint[1]
        
        # Calculate shoulder-to-nose distance relative to shoulder width
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        head_offset_ratio = abs(head_forward_deviation) / shoulder_width
        
        # Hunching indicators
        hunching_indicators = {
            "spine_angle": abs(spine_angle),  # Deviation from vertical
            "head_forward_ratio": head_offset_ratio,
            "vertical_head_drop": head_vertical_deviation
        }
        
        # Scoring mechanism
        # Lower scores indicate more hunching
        MAX_ACCEPTABLE_SPINE_ANGLE = 20  # degrees
        MAX_HEAD_FORWARD_RATIO = 0.3  # proportion of shoulder width
        MAX_VERTICAL_DROP = 0.1  # proportion of body height
        
        # Calculate hunching score
        spine_score = 1 - min(1, abs(spine_angle) / MAX_ACCEPTABLE_SPINE_ANGLE)
        head_forward_score = 1 - min(1, head_offset_ratio / MAX_HEAD_FORWARD_RATIO)
        vertical_drop_score = 1 - min(1, abs(head_vertical_deviation) / MAX_VERTICAL_DROP)
        
        # Weighted average of scores
        hunching_score = (
            spine_score * 0.4 + 
            head_forward_score * 0.3 + 
            vertical_drop_score * 0.3
        )
        
        # Invert score so that lower values (more hunching) approach 0
        hunching_score = max(0, min(1, hunching_score))
        
        return hunching_score, hunching_indicators
    
    except Exception as e:
        print(f"Error in hunching detection: {e}")
        return 0, {"error": str(e)}
    
run_monitoring = False

def run_concentration_monitor():
    global run_monitoring, monitoring_stop_event
    monitoring_stop_event = threading.Event()

    # Weights for concentration score calculation
    w1, w2, w3, w4, w5 = 0.3, 0.3, 0.15, 0.05, 0.10 

    # Initialize MediaPipe solutions
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    # Lists to store metrics data
    time_stamps = []
    head_stability_values = []
    face_neutrality_values = []
    eye_gaze_values = []
    light_change_values = []
    posture_values = []
    concentration_scores = []

    # Set up plotting
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
    posture_line, = ax1.plot([], [], 'p-', label="Posture Stability")
    ax1.legend(loc="upper right")

    # Secondary plot for overall concentration score
    ax2.set_title("Overall Concentration Score")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Concentration Score")
    concentration_line, = ax2.plot([], [], 'k-', linewidth=2)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    # Open camera
    cap = cv2.VideoCapture(0)

    # Variables to store previous values
    prev_head_pos = None
    prev_light = None
    prev_gaze = None

    # Tracking variables
    frame_count = 0
    update_interval = 10
    start_time = time.time()

    CONCENTRATION_THRESHOLD = 0.6 
    LOW_CONCENTRATION_WARNING = False 

    while run_monitoring and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed_time = time.time() - start_time

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        if face_results.multi_face_landmarks and pose_results.pose_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            pose_landmarks = pose_results.pose_landmarks

            # Calculate all metrics
            face_neutrality = calculate_face_neutrality(face_landmarks)
            head_stability, prev_head_pos = calculate_head_stability(pose_landmarks, prev_head_pos)
            light_stability, prev_light = calculate_light_changes(frame, prev_light)
            eye_stability, prev_gaze = calculate_eye_gaze_stability(face_landmarks, prev_gaze)
            posture_stability, neck_angle = analyze_upper_body_posture(pose_landmarks.landmark)
            hunching_score, _ = detect_hunching(pose_results.pose_landmarks.landmark)

            # Calculate overall concentration score
            concentration_score = (w1 * face_neutrality + 
                                  w2 * head_stability +  
                                  w3 * eye_stability + 
                                  w4 * light_stability +
                                  w5 * posture_stability +
                                  0.1 * (1 - hunching_score))
            
            if concentration_score < CONCENTRATION_THRESHOLD and not LOW_CONCENTRATION_WARNED:
                # Use Tkinter messagebox for popup
                r.after(0, lambda: messagebox.showwarning(
                    "Concentration Alert", 
                    f"Your concentration has dropped to {concentration_score:.2f}. \n"
                    "Take a short break or adjust your posture."
                ))
                LOW_CONCENTRATION_WARNED = True

            # Reset warning flag if concentration improves
            if concentration_score >= CONCENTRATION_THRESHOLD:
                LOW_CONCENTRATION_WARNED = False

            # Store data
            time_stamps.append(elapsed_time)
            head_stability_values.append(head_stability)
            face_neutrality_values.append(face_neutrality)
            eye_gaze_values.append(eye_stability)
            light_change_values.append(light_stability)
            posture_values.append(posture_stability)
            concentration_scores.append(concentration_score)

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

                posture_line.set_xdata(time_stamps)
                posture_line.set_ydata(posture_values)

                concentration_line.set_xdata(time_stamps)
                concentration_line.set_ydata(concentration_scores)
                
                # Adjust plot limits
                ax1.set_xlim(0, max(10, elapsed_time))
                ax1.set_ylim(0, 1.1)

                ax2.set_xlim(0, max(10, elapsed_time))
                  
                plt.draw()
                plt.pause(0.01)

                # Resetting the frame counter 
                frame_count = 0 

    try:
        while run_monitoring and not monitoring_stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Add a periodic check for stop condition
            if not run_monitoring:
                monitoring_stop_event.set()
                break

            # Existing frame display and key check
            cv2.imshow("Concentration Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in concentration monitoring: {e}")
    
    finally:
        # Comprehensive cleanup
        try:
            cap.release()
            cv2.destroyAllWindows()
            plt.close(fig)
            plt.close('all')  # Ensure all matplotlib windows are closed
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

    print("Concentration monitoring stopped.")

# Tkinter Window Setup
r = tk.Tk()
r.geometry("600x500")
r.title("DeepSpaceFocus")

# Create main container
container = tk.Frame(r)
container.pack(fill="both", expand=True)

# Dictionary to store frames
frames = {}

# Menu Bar Config
menu = Menu(r)
r.config(menu=menu)

# Function to switch frames
def show_frame(frame):
    frame.tkraise()

# HOME PAGE 
home_page = tk.Frame(container)
frames["home"] = home_page
home_page.grid(row=0, column=0, sticky="nsew")

# Home Functions
def start_monitoring():
    global run_monitoring, monitoring_thread
    if not run_monitoring:
        run_monitoring = True
        monitoring_thread = threading.Thread(target=run_concentration_monitor, daemon=True)
        monitoring_thread.start()

def stop_monitoring():
    global run_monitoring, monitoring_thread
    
    run_monitoring = False
    monitoring_stop_event.set()

    # Attempt to gracefully stop
    try:
        if monitoring_thread and monitoring_thread.is_alive():
            monitoring_thread.join(timeout=2)
    except Exception as e:
        print(f"Error stopping monitoring thread: {e}")
    
    # Force close if thread doesn't respond
    cv2.destroyAllWindows()
    plt.close('all')

    # Optional: Reset the event for future use
    monitoring_stop_event.clear()

# Home page widgets using grid
home_title = Label(home_page, text="Welcome to DeepSpaceFocus", font=("Arial", 16, "bold"))
home_title.grid(row=0, column=0, padx=20, pady=20, columnspan=3)

home_info = Label(home_page, text="Your productivity assistant")
home_info.grid(row=1, column=0, padx=20, pady=10, columnspan=3)

home_description = Label(home_page, text="Use the menu above to navigate between features")
home_description.grid(row=2, column=0, padx=20, pady=30, columnspan=3)

monitor_start_button = ttk.Button(home_page, text="Start Monitoring", command=start_monitoring)
monitor_start_button.grid(row=3, column=0, padx=20, pady=10)

monitor_stop_button = ttk.Button(home_page, text="Stop Monitoring", command=stop_monitoring)
monitor_stop_button.grid(row=3, column=1, padx=20, pady=10)

# TIMER PAGE 
timer_page = tk.Frame(container)
frames["timer"] = timer_page
timer_page.grid(row=0, column=0, sticky="nsew")

# Timer page widgets (from previous script)
timer_title = Label(timer_page, text="Break Timer", font=("Arial", 14, "bold"))
timer_title.grid(row=0, column=0, padx=20, pady=20, columnspan=3)

time_unit_label = Label(timer_page, text="Select time unit:")
time_unit_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

time_unit_var = StringVar()
combo_box = ttk.Combobox(timer_page, textvariable=time_unit_var, 
                         values=["Minutes", "Seconds", "Hours"])
combo_box.grid(row=1, column=1, padx=10, pady=10, sticky="w")
combo_box.set("Minutes")

selected_unit_label = Label(timer_page, text="")
selected_unit_label.grid(row=1, column=2, padx=10, pady=10, sticky="w")

time_value_label = Label(timer_page, text="Enter time value:")
time_value_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")

time_entry = Entry(timer_page)
time_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

def select_time(event):
    selected_time = time_unit_var.get()
    selected_unit_label.config(text="Selected: " + selected_time)

combo_box.bind("<<ComboboxSelected>>", select_time)

def start_timer():
    try: 
        time_value = float(time_entry.get())

        time_unit = time_unit_var.get()
        if time_unit == "Minutes":
            seconds = time_value * 60
        elif time_unit == "Hours":
            seconds = time_value * 3600
        else:
            seconds = time_value

        timer_window = tk.Toplevel()
        timer_window.title("Timer")
        timer_window.geometry("400x150")

        countdown_label = Label(timer_window, text="Time Remaining:", font=("Arial", 12))
        countdown_label.pack(pady=10)

        time_left_label = Label(timer_window, text="", font=("Arial", 24))
        time_left_label.pack(pady=10)

        stop_button = tk.Button(timer_window, text="Stop Timer", bg="#F44336", fg="white", 
                               command=timer_window.destroy)
        stop_button.pack(pady=10)

        def update_countdown(remaining):
            if remaining <= 0:
                time_left_label.config(text="Time's up!")
                messagebox.showinfo("Break Timer", "Your break time is over!")
                timer_window.destroy()
                return
            
            mins, secs = divmod(int(remaining), 60)
            hours, mins = divmod(mins, 60)

            if hours > 0:
                time_left_label.config(text=f"{hours:02d}:{mins:02d}:{secs:02d}")
            else:
                time_left_label.config(text=f"{mins:02d}:{secs:02d}")

            timer_window.after(1000, update_countdown, remaining - 1)
        
        update_countdown(seconds)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid number")    

# Start button
start_button = tk.Button(timer_page, text="Start Timer", bg="#4CAF50", fg="white", command=start_timer)
start_button.grid(row=3, column=0, columnspan=3, padx=20, pady=20)    

# HELP PAGE
help_page = tk.Frame(container)
frames["help"] = help_page
help_page.grid(row=0, column=0, sticky="nsew")

# Help page widgets
help_title = Label(help_page, text="Help & Instructions", font=("Arial", 14, "bold"))
help_title.grid(row=0, column=0, padx=20, pady=20, columnspan=2)

help_subtitle = Label(help_page, text="How to use DeepSpaceFocus:", font=("Arial", 12))
help_subtitle.grid(row=1, column=0, padx=20, pady=10, sticky="w", columnspan=2)

help_text1 = Label(help_page, text="1. Use the Timer feature to set break reminders")
help_text1.grid(row=2, column=0, padx=30, pady=5, sticky="w", columnspan=2)

help_text2 = Label(help_page, text="2. Navigate using the menu bar at the top")
help_text2.grid(row=3, column=0, padx=30, pady=5, sticky="w", columnspan=2)

help_text3 = Label(help_page, text="3. Contact support at support@deepspacefocus.com")
help_text3.grid(row=4, column=0, padx=30, pady=5, sticky="w", columnspan=2)

# MENU CONFIGURATION
# Home Menu
home_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Home", menu=home_menu)
home_menu.add_command(label="Home", command=lambda: show_frame(home_page))

# Timer Menu
timer_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Timer", menu=timer_menu)
timer_menu.add_command(label="Set Timer", command=lambda: show_frame(timer_page))

# Help Menu
help_menu = Menu(menu, tearoff=0)
menu.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="Controls", command=lambda: show_frame(help_page))

# Configure grid weights to make frames expandable
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

# Make all frames expand to fill the container
for frame in frames.values():
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

# Show home page initially
show_frame(frames["home"])

r.mainloop()