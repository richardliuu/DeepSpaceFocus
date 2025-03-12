import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# Starting with empty lists so they can be filled 
time_stamps = []
head_stability_values = []
face_neutrality_values = []
eye_gaze_values = []
light_change_values = []

start_time = time.time()

plt.ion() 
fig, ax1 = plt.subplots()

# Primary axis (Head Tilt - Red)
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Head Stability", color='r')
line1, = ax1.plot([], [], 'r-', label="Head Stability")
ax1.tick_params(axis='y', labelcolor='r')

# Secondary axis (Eye Blink - Blue)
ax2 = ax1.twinx()
ax2.set_ylabel("Eye Gaze Stability", color='b')
line2, = ax2.plot([], [], 'b-', label="Eye Gaze Stability")
ax2.tick_params(axis='y', labelcolor='b')

def calculate_face_neutrality(face_landmarks):
    return np.random.uniform(0.7, 1.0)

def calculate_head_stability(pose_landmarks):
    if pose_landmarks:
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        return nose.y
    return 0

def calculate_light_changes(frame):
    return np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

def calculate_eye_gaze_stability(face_landmarks):
    left_eye = [face_landmarks.landmark[i] for i in [159, 145]]
    right_eye = [face_landmarks.landmark[i] for i in [386, 374]]
    left_ear = abs(left_eye[0].y - left_eye[1].y) 
    right_ear = abs(right_eye[0].y - right_eye[1].y) 
    return (left_ear + right_ear) / 2 * 100

# Need to modify the metrics that are tracked
# New metrics for the concentration algorithm
# Still need to keep the shape indicators 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe to process it 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    if face_results.multi_face_landmarks and pose_results.pose_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        pose_landmarks = pose_results.pose_landmarks

        face_neutrality = calculate_face_neutrality(face_landmarks)
        head_stability = calculate_head_stability(pose_landmarks)
        eye_gaze_stability = calculate_eye_gaze_stability(face_landmarks)
        light_changes = calculate_light_changes(frame)

        # Marking Forehead and Chin

        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        forehead_x, forehead_y = int(forehead.x * frame.shape[1]), int(forehead.y * frame.shape[0])
        chin_x, chin_y = int(chin.x * frame.shape[1]), int(chin.y * frame.shape[0])
        
        cv2.circle(frame, (forehead_x, forehead_y), 5, (0, 255, 0), -1)
        cv2.circle(frame, (chin_x, chin_y), 5, (0, 0, 255), -1)
        cv2.line(frame, (forehead_x, forehead_y), (chin_x, chin_y), (255, 255, 0), 2) 

        elapsed_time = time.time() - start_time
        time_stamps.append(elapsed_time)
        head_stability_values.append(head_stability)
        face_neutrality_values.append(face_neutrality)
        eye_gaze_values.append(eye_gaze_stability)
        light_change_values.append(light_changes)

        line1.set_xdata(time_stamps)
        line1.set_ydata(head_stability_values)
        ax1.set_xlim(0, max(10, elapsed_time))
        ax1.set_ylim(0, max(head_stability_values) + 0.1)

        line2.set_xdata(time_stamps)
        line2.set_ydata(eye_gaze_values)
        ax2.set_xlim(0, max(10, elapsed_time))
        ax2.set_ylim(0, max(eye_gaze_values) + 5)

        plt.draw()
        plt.pause(0.01)

    cv2.imshow("Concentration Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()


# Calculating Concentration Score 

#Concentration Score = w1(Face Neutrality) + w2(Head Stability) + w3(Task Engagement) + w4(Eye Gaze Stability) + w5(Changes in Light) 

# Equal Weighting (Subject to change)



