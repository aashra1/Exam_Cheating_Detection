import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time

# Load YOLOv5 model
model = YOLO('yolov5s.pt')

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)

# Camera
cap = cv2.VideoCapture(0)

# 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),              
    (0.0, -63.6, -12.5),          
    (-43.3, 32.7, -26.0),         
    (43.3, 32.7, -26.0),          
    (-28.9, -28.9, -24.1),        
    (28.9, -28.9, -24.1)          
])

# Cheating score setup
cheating_score = 0
last_check_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    phone_detected = False

    # --- Phone Detection ---
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        if 'phone' in label.lower():
            phone_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {int(conf*100)}%", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- Head Pose Detection and Face Rectangle ---
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Get all landmark points for this face
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]

            # Calculate bounding box coordinates for rectangle
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            # Draw rectangle around face
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Prepare 2D image points for head pose estimation
            image_points = np.array([
                (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),      # Nose tip
                (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h),  # Chin
                (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h),  # Right eye right corner
                (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),    # Left eye left corner
                (landmarks.landmark[287].x * w, landmarks.landmark[287].y * h),  # Left mouth corner
                (landmarks.landmark[57].x * w, landmarks.landmark[57].y * h)     # Right mouth corner
            ], dtype="double")

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)

            rmat, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rmat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

            pitch, yaw, roll = [angle[0] for angle in euler_angles]

            cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ---- Cheating score logic ----
            now = time.time()
            if now - last_check_time > 1:  # check every second
                last_check_time = now

                if abs(yaw) > 25:
                    cheating_score += 10
                elif abs(yaw) > 15:
                    cheating_score += 5
                else:
                    cheating_score -= 3  # normalize for normal posture

                if pitch < -40:  # looking up too much
                    cheating_score += 8
                elif pitch < -25:
                    cheating_score += 4
                elif pitch < -10:  # normal writing head down
                    cheating_score -= 2
                else:
                    cheating_score += 2

                if phone_detected:
                    cheating_score += 30

                # Clamp score between 0 and 100
                cheating_score = max(0, min(cheating_score, 100))

    # ---- Display score and status ----
    cv2.putText(frame, f"Cheating Score: {cheating_score}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if cheating_score > 60:
        cv2.putText(frame, "CHEATING LIKELY!", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.imshow("Cheating Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
