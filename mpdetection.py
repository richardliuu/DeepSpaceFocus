import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time
import threading

# Debug imports
import traceback
import sys 

from scipy.signal import find_peaks
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

run_monitoring = False

# Set to 0 to avoid messages 
# 2 for warnings 

# Frame 
update_interval = 10
frame_count = 0

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Lists to store metrics data
time_stamps = []
head_stability_values = []
face_neutrality_values = []
eye_gaze_values = []
light_change_values = []
task_engagement_values = []
concentration_scores = []
latest_frame = None

# Weights for concentration score calculation
w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.2, 0.2 

# Reference values for normalization
baseline_movement = 0
baseline_samples = 0
baseline_period = 5  # seconds to establish 

start_time = time.time()

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
task_line, = ax1.plot([], [], 'c-', label="Task Engagement")
light_line, = ax1.plot([], [], 'y-', label="Light Stability")
ax1.legend(loc="upper right")

# Secondary plot for overall concentration score
ax2.set_title("Overall Concentration Score")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Concentration Score")
concentration_line, = ax2.plot([], [], 'k-', linewidth=2)
ax2.set_ylim(0, 1)

# To reduce the computational power required
plt.tight_layout()

def run_concentration_monitor():
    global run_monitoring, latest_frame, prev_head_pos, prev_light, prev_gaze
    run_monitoring = True
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and run_monitoring:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame in RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        elapsed_time = time.time() - start_time

        # If both face and pose data are available, compute metrics and draw indicators
        if face_results.multi_face_landmarks and pose_results.pose_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            pose_landmarks = pose_results.pose_landmarks

            # Calculate metrics
            face_neutrality = calculate_face_neutrality(face_landmarks)
            head_stability, prev_head_pos = calculate_head_stability(pose_landmarks, prev_head_pos)
            light_stability, prev_light = calculate_light_changes(frame, prev_light)
            eye_stability, prev_gaze = calculate_eye_gaze_stability(face_landmarks, prev_gaze)
            task_engagement = calculate_task_engagement(face_landmarks, head_stability, eye_stability)
            concentration_score = (w1 * face_neutrality + w2 * head_stability +
                                   w3 * task_engagement + w4 * eye_stability +
                                   w5 * light_stability)

            # Append data to global lists
            time_stamps.append(elapsed_time)
            head_stability_values.append(head_stability)
            face_neutrality_values.append(face_neutrality)
            eye_gaze_values.append(eye_stability)
            light_change_values.append(light_stability)
            task_engagement_values.append(task_engagement)
            concentration_scores.append(concentration_score)
            
            # Draw face indicators on the frame
            # Marking Forehead and Chin
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            forehead_x = int(forehead.x * frame.shape[1])
            forehead_y = int(forehead.y * frame.shape[0])
            chin_x = int(chin.x * frame.shape[1])
            chin_y = int(chin.y * frame.shape[0])
            cv2.circle(frame, (forehead_x, forehead_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (chin_x, chin_y), 5, (0, 0, 255), -1)
            cv2.line(frame, (forehead_x, forehead_y), (chin_x, chin_y), (255, 255, 0), 2)
            
            # Mark eyes
            left_eye = face_landmarks.landmark[LEFT_IRIS_CENTER_APPROX]
            right_eye = face_landmarks.landmark[RIGHT_IRIS_CENTER_APPROX]
            left_eye_x = int(left_eye.x * frame.shape[1])
            left_eye_y = int(left_eye.y * frame.shape[0])
            right_eye_x = int(right_eye.x * frame.shape[1])
            right_eye_y = int(right_eye.y * frame.shape[0])
            cv2.circle(frame, (left_eye_x, left_eye_y), 5, (255, 0, 255), -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 5, (255, 0, 255), -1)
            
            # Add text with concentration score
            cv2.putText(frame, f"Concentration: {concentration_score:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Update the global frame for display in the main thread
        latest_frame = frame.copy()
        time.sleep(0.1)
        
    cap.release()

def update_plots():
    # Update plots if new data is available
    if time_stamps:
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
        concentration_line.set_xdata(time_stamps)
        concentration_line.set_ydata(concentration_scores)
        
        # Adjust axes limits
        ax1.set_xlim(0, max(10, time_stamps[-1]))
        ax1.set_ylim(0, 1.1)
        ax2.set_xlim(0, max(10, time_stamps[-1]))
        
        fig.canvas.draw_idle()
    
    # If a new video frame is available, display it in an OpenCV window
    if latest_frame is not None:
        cv2.imshow("Concentration Monitoring", latest_frame)
        # Check if user pressed 'q' in the OpenCV window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_monitoring()

    # Schedule next update after 100 ms
    r.after(100, update_plots)

    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

# ============= TKINTER WINDOW =============
import tkinter as tk
from tkinter import Menu, Label, Entry, StringVar
from tkinter import ttk, messagebox

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
    global monitor_threading
    if not run_monitoring:
        monitoring_thread = threading.Thread(target=run_concentration_monitor, daemon=True)
        monitoring_thread.start()

def stop_monitoring():
    global run_monitoring  
    run_monitoring = False
    cv2.destroyAllWindows()

# Home page widgets using grid
home_title = Label(home_page, text="Welcome to DeepSpaceFocus", font=("Arial", 16, "bold"))
home_title.grid(row=0, column=0, padx=20, pady=20, columnspan=3)

home_info = Label(home_page, text="Your productivity assistant")
home_info.grid(row=1, column=0, padx=20, pady=10, columnspan=3)

# Additional home content can be added here
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

# Timer page title
timer_title = Label(timer_page, text="Break Timer", font=("Arial", 14, "bold"))
timer_title.grid(row=0, column=0, padx=20, pady=20, columnspan=3)

# Time unit selection
time_unit_label = Label(timer_page, text="Select time unit:")
time_unit_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

time_unit_var = StringVar()
combo_box = ttk.Combobox(timer_page, textvariable=time_unit_var, 
                         values=["Minutes", "Seconds", "Hours"])
combo_box.grid(row=1, column=1, padx=10, pady=10, sticky="w")
combo_box.set("Minutes")

selected_unit_label = Label(timer_page, text="")
selected_unit_label.grid(row=1, column=2, padx=10, pady=10, sticky="w")

# Time value entry
time_value_label = Label(timer_page, text="Enter time value:")
time_value_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")

time_entry = Entry(timer_page)
time_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

# Function to update the selected time unit label
def select_time(event):
    selected_time = time_unit_var.get()
    selected_unit_label.config(text="Selected: " + selected_time)

combo_box.bind("<<ComboboxSelected>>", select_time)

def start_timer():
    try: 
        time_value = float(time_entry.get())

        time_unit = time_unit_var.get()
        # Fixed the variable name here (was using time_unit_var instead of time_unit)
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
        countdown_label.pack(pady=10)  # Reduced padding

        time_left_label = Label(timer_window, text="", font=("Arial", 24))  # Added font size
        time_left_label.pack(pady=10)

        stop_button = tk.Button(timer_window, text="Stop Timer", bg="#F44336", fg="white", 
                               command=timer_window.destroy)
        stop_button.pack(pady=10)

        def update_countdown(remaining):
            if remaining <= 0:
                time_left_label.config(text="Time's up!")
                # Fixed missing message parameter
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
        
        # Fixed indentation - this call was inside the update_countdown function
        update_countdown(seconds)
    except ValueError:
        # Fixed missing message parameter
        messagebox.showerror("Input Error", "Please enter a valid number")    

# Start button
start_button = tk.Button(timer_page, text="Start Timer", bg="#4CAF50", fg="white", command=start_timer)
start_button.grid(row=3, column=0, columnspan=3, padx=20, pady=20)    

# HELP PAGE
help_page = tk.Frame(container)
frames["help"] = help_page
help_page.grid(row=0, column=0, sticky="nsew")

# Help page widgets using grid
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

# ============= MENU CONFIGURATION =============
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

r.after(100, update_plots)
r.mainloop()