from ultralytics import YOLO
import torch
import numpy as np
import cv2  # Required for visualization

def init_pose():
    model = YOLO('models/yolov8s-pose.pt')
    if torch.cuda.is_available():
        model.to('cuda')
    return model

def detect_pose_keypoints(pose_model, rgb_frame):
    results = pose_model(rgb_frame)
    if not results or len(results) == 0 or results[0].keypoints is None:
        return []
    keypoints_list = results[0].keypoints.cpu().numpy()  # (num_people, 17, 3)
    return keypoints_list

def hands_near_faces(pose_model, rgb_frame, faces, distance_threshold=50):
    keypoints_list = detect_pose_keypoints(pose_model, rgb_frame)

    hands_near_face = {face['id']: False for face in faces}
    for person_kpts in keypoints_list:
        if len(person_kpts) < 11:
            continue  # not enough keypoints for wrists

        left_wrist = person_kpts[9]   # (x, y, confidence)
        right_wrist = person_kpts[10]

        wrists = []
        if left_wrist[2] > 0.3:
            wrists.append(left_wrist[:2])
        if right_wrist[2] > 0.3:
            wrists.append(right_wrist[:2])

        for face in faces:
            if hands_near_face[face['id']]:
                continue

            x1, y1, x2, y2 = face['bbox']
            face_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            for wrist in wrists:
                dist = np.linalg.norm(np.array(wrist) - face_center)
                if dist < distance_threshold:
                    hands_near_face[face['id']] = True
                    break
    return hands_near_face

# Pose connections for drawing (COCO format)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 12), (8, 10)
]

def draw_pose(frame, keypoints, color=(0, 255, 255)):
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            if keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3:
                pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(frame, pt1, pt2, color, 2)
