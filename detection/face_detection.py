import mediapipe as mp
import numpy as np
import cv2
from ultralytics import YOLO

mp_face_mesh = mp.solutions.face_mesh

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),                # Nose tip
    (0.0, -63.6, -12.5),            # Chin
    (-43.3, 32.7, -26.0),           # Left eye left corner
    (43.3, 32.7, -26.0),            # Right eye right corner
    (-28.9, -28.9, -24.1),          # Left mouth corner
    (28.9, -28.9, -24.1)            # Right mouth corner
])

def init_face_mesh():
    yolo_model = YOLO("models/yolov8s-face-lindevs.pt")
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)
    return yolo_model, face_mesh

def get_faces(yolo_model, face_mesh, rgb_frame, frame_width, frame_height):
    results = yolo_model.predict(source=rgb_frame, verbose=False, conf=0.4)
    faces = []

    if not results or not results[0].boxes:
        return faces

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width - 1, x2), min(frame_height - 1, y2)

        face_roi = cv2.cvtColor(rgb_frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(face_roi)

        pitch, yaw, roll = 0, 0, 0
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            image_points = np.array([
                (landmarks.landmark[1].x * (x2 - x1), landmarks.landmark[1].y * (y2 - y1)),     # Nose tip
                (landmarks.landmark[152].x * (x2 - x1), landmarks.landmark[152].y * (y2 - y1)), # Chin
                (landmarks.landmark[263].x * (x2 - x1), landmarks.landmark[263].y * (y2 - y1)), # Right eye right corner
                (landmarks.landmark[33].x * (x2 - x1), landmarks.landmark[33].y * (y2 - y1)),   # Left eye left corner
                (landmarks.landmark[287].x * (x2 - x1), landmarks.landmark[287].y * (y2 - y1)), # Right mouth corner
                (landmarks.landmark[57].x * (x2 - x1), landmarks.landmark[57].y * (y2 - y1))    # Left mouth corner
            ], dtype="double")

            focal_length = x2 - x1
            center = (focal_length / 2, (y2 - y1) / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))
            success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                proj_matrix = np.hstack((rmat, tvec))
                _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(proj_matrix)
                pitch, yaw, roll = [angle[0] for angle in angles]

        faces.append({
            'id': i,
            'bbox': (x1, y1, x2, y2),
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'landmarks': mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None
        })

    return faces
