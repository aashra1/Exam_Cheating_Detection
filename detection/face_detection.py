import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh

# 3D model points for head pose
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),                # Nose tip
    (0.0, -63.6, -12.5),            # Chin
    (-43.3, 32.7, -26.0),           # Left eye left corner
    (43.3, 32.7, -26.0),            # Right eye right corner
    (-28.9, -28.9, -24.1),          # Left Mouth corner
    (28.9, -28.9, -24.1)            # Right Mouth corner
])

def init_face_mesh():
    return mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)

def get_faces(face_mesh, rgb_frame, frame_width, frame_height):
    results = face_mesh.process(rgb_frame)
    faces = []

    if results.multi_face_landmarks:
        for i, landmarks in enumerate(results.multi_face_landmarks):
            image_points = np.array([
                (landmarks.landmark[1].x * frame_width, landmarks.landmark[1].y * frame_height),     # Nose tip
                (landmarks.landmark[152].x * frame_width, landmarks.landmark[152].y * frame_height), # Chin
                (landmarks.landmark[263].x * frame_width, landmarks.landmark[263].y * frame_height), # Right eye
                (landmarks.landmark[33].x * frame_width, landmarks.landmark[33].y * frame_height),   # Left eye
                (landmarks.landmark[287].x * frame_width, landmarks.landmark[287].y * frame_height), # Right mouth
                (landmarks.landmark[57].x * frame_width, landmarks.landmark[57].y * frame_height)    # Left mouth
            ], dtype="double")

            # Camera intrinsics
            focal_length = frame_width
            center = (frame_width / 2, frame_height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs)

            rmat, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rmat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

            pitch, yaw, roll = [angle[0] for angle in euler_angles]

            lm_x = [lm.x for lm in landmarks.landmark]
            lm_y = [lm.y for lm in landmarks.landmark]
            min_x, max_x = int(min(lm_x) * frame_width), int(max(lm_x) * frame_width)
            min_y, max_y = int(min(lm_y) * frame_height), int(max(lm_y) * frame_height)

            faces.append({
                'id': i,
                'landmarks': landmarks,
                'bbox': (min_x, min_y, max_x, max_y),
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll
            })

    return faces
