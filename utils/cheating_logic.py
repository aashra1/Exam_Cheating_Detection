import time
from collections import defaultdict, deque
import cv2

rolling_window_seconds = 10

cheating_scores = defaultdict(float)
suspicious_timestamps = defaultdict(lambda: deque())

def update_scores(faces, phone_boxes, hands_near_face, now):
    frame_interval = 0.5
    for face in faces:
        face_id = face['id']
        min_x, min_y, max_x, max_y = face['bbox']
        pitch = face['pitch']
        yaw = face['yaw']

        suspicious = False
        if abs(yaw) > 40 or pitch < -30:
            suspicious = True

        if hands_near_face:
            suspicious = True

        for px1, py1, px2, py2 in phone_boxes:
            if px1 < max_x and px2 > min_x and py1 < max_y and py2 > min_y:
                suspicious = True

        # Clean old timestamps
        while suspicious_timestamps[face_id] and now - suspicious_timestamps[face_id][0] > rolling_window_seconds:
            suspicious_timestamps[face_id].popleft()

        if suspicious:
            suspicious_timestamps[face_id].append(now)

        suspicious_frames = len(suspicious_timestamps[face_id])
        total_frames = int(rolling_window_seconds / frame_interval)
        suspicion_ratio = suspicious_frames / total_frames if total_frames > 0 else 0

        increment = suspicion_ratio * 20
        if hands_near_face:
            increment += 5
        if phone_boxes:
            increment += 10

        cheating_scores[face_id] = min(100, cheating_scores[face_id] + increment)
        decay = (1 - suspicion_ratio) * 10
        cheating_scores[face_id] = max(0, cheating_scores[face_id] - decay)

def visualize(frame, faces):
    for face in faces:
        face_id = face['id']
        min_x, min_y, max_x, max_y = face['bbox']
        score = cheating_scores[face_id]
        label = f"Face {face_id} - {int(score)}%"
        color = (0, 0, 255) if score > 85 else (255, 255, 0)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
        cv2.putText(frame, label, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if score > 85:
            cv2.putText(frame, "CHEATING LIKELY!", (min_x, max_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
