import time
from collections import defaultdict, deque
import cv2

rolling_window_seconds = 10
frame_interval = 0.5  # used for decay calculation

cheating_scores = defaultdict(float)
last_suspicious_time = defaultdict(lambda: 0)

# New variables for glance counting
glance_threshold = 4           # Number of glances before flagging cheating likely
glance_window_seconds = 10     # Time window to count glances
glance_timestamps = defaultdict(lambda: deque())  # To track timestamps of suspicious glances per face

def update_scores(faces, phone_boxes, hands_near_face, now):
    for face in faces:
        face_id = face['id']
        min_x, min_y, max_x, max_y = face['bbox']
        pitch = face['pitch']
        yaw = face['yaw']

        suspicion_level = 0.0

        # Pose-based suspicion and glance detection
        is_glance = False
        if abs(yaw) > 40:
            suspicion_level += 0.3
            is_glance = True
        if pitch < -30:
            suspicion_level += 0.3
            is_glance = True

        suspicious = suspicion_level > 0

        print(f"Face {face_id} - pitch: {pitch}, yaw: {yaw}, suspicion_level so far: {suspicion_level:.2f}")

        # Track glance timestamps for pose-based suspicious events
        if is_glance:
            glance_timestamps[face_id].append(now)

            # Remove old timestamps outside the glance window
            while glance_timestamps[face_id] and now - glance_timestamps[face_id][0] > glance_window_seconds:
                glance_timestamps[face_id].popleft()

            # Check if glance count exceeds threshold
            if len(glance_timestamps[face_id]) >= glance_threshold:
                suspicion_level = 1.0  # Max suspicion level on glance threshold reached
                suspicious = True
                print(f"Face {face_id} - exceeds glance threshold with {len(glance_timestamps[face_id])} glances!")

        # Hands near face adds suspicion
        if hands_near_face:
            suspicion_level += 0.5
            suspicious = True
            print(f"Face {face_id} - hands near face detected, adding suspicion")

        # Phone overlap adds high suspicion
        phone_near_face = False
        for px1, py1, px2, py2 in phone_boxes:
            if px1 < max_x and px2 > min_x and py1 < max_y and py2 > min_y:
                suspicion_level += 0.7
                suspicious = True
                phone_near_face = True
                print(f"Face {face_id} - phone detected in face bbox, adding suspicion")
                break

        # Clamp suspicion level to max 1.0
        suspicion_level = min(1.0, suspicion_level)

        alpha = 0.2  # EMA smoothing factor
        prev_score = cheating_scores[face_id]
        new_score = prev_score * (1 - alpha) + suspicion_level * 100 * alpha

        # Boost cheating score by 10% if glance threshold exceeded
        if len(glance_timestamps[face_id]) >= glance_threshold:
            new_score = min(100, new_score + 10)  
            print(f"Face {face_id} - cheating score boosted by 10% due to glance count")

        # Apply slower decay based on actual elapsed time since last suspicious event
        if not suspicious:
            time_since_last_suspicious = now - last_suspicious_time[face_id]
            decay_rate = 0.2  # much slower decay: 0.2 points per second
            decay_amount = decay_rate * time_since_last_suspicious
            new_score = max(0, new_score - decay_amount)
        else:
            last_suspicious_time[face_id] = now

        cheating_scores[face_id] = max(0, min(100, new_score))


def visualize(frame, faces):
    for face in faces:
        face_id = face['id']
        min_x, min_y, max_x, max_y = face['bbox']
        score = cheating_scores[face_id]

        # Color code: green (safe), yellow (suspicious), red (highly suspicious)
        if score > 85:
            color = (0, 0, 255)  # red
            label = f"Face {face_id} - CHEATING LIKELY! {int(score)}%"
        elif score > 50:
            color = (0, 255, 255)  # yellow
            label = f"Face {face_id} - Suspicious {int(score)}%"
        else:
            color = (0, 255, 0)  # green
            label = f"Face {face_id} - {int(score)}%"

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
        text_y = max(min_y - 10, 0)
        cv2.putText(frame, label, (min_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
