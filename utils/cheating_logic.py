import time
from collections import defaultdict, deque
import cv2
import os
from datetime import datetime

# ===== NEW IMPORT FOR DATABASE LOGGING =====
from utils.async_logger import enqueue_log  # Use async logger now

rolling_window_seconds = 10
frame_interval = 0.5

# Dictionary to store cheating scores for each face_id
cheating_scores = defaultdict(float)
# Dictionary to store the last suspicious activity time per face_id
last_suspicious_time = defaultdict(lambda: 0)

# Parameters for glance detection
glance_threshold = 4
glance_window_seconds = 10
glance_timestamps = defaultdict(lambda: deque())
# Timer for sustained hands near face per face_id
hands_on_face_start = defaultdict(lambda: None)

snapshot_dir = "snapshots"
# Ensure snapshot folder exists
os.makedirs("snapshots", exist_ok=True)

def clamp_bbox(bbox, frame_shape):
    h, w = frame_shape[:2]
    x1 = max(0, min(bbox[0], w - 1))
    y1 = max(0, min(bbox[1], h - 1))
    x2 = max(0, min(bbox[2], w - 1))
    y2 = max(0, min(bbox[3], h - 1))
    return x1, y1, x2, y2

_last_log_time = defaultdict(lambda: defaultdict(lambda: 0))
LOG_COOLDOWN_SECONDS = 10  # don't log same event for same face more than once every 10 seconds

def log_event(timestamp_str, face_id, activity, severity, cropped_face=None, class_id="default_class"):
    valid_activities = {
        "Looking around frequently",
        "Phone detected",
        "Phone detected NEAR HAND",
        "Phone detected near face",
        "Suspicious behavior",
        "CHEATING LIKELY"
    }
    if activity not in valid_activities:
        return

    if severity not in ["warning", "critical"]:
        return

    now = time.time()
    if now - _last_log_time[face_id][activity] < LOG_COOLDOWN_SECONDS:
        return
    _last_log_time[face_id][activity] = now

    enqueue_log(timestamp_str, face_id, activity, severity, cropped_face, class_id)

def is_near(box1, box2, max_dist=50):
    # Calculate distance between center points of two boxes
    x1c = (box1[0] + box1[2]) / 2
    y1c = (box1[1] + box1[3]) / 2
    x2c = (box2[0] + box2[2]) / 2
    y2c = (box2[1] + box2[3]) / 2
    dist = ((x1c - x2c)**2 + (y1c - y2c)**2)**0.5
    return dist < max_dist

def draw_boxes(frame, boxes, color, label):
    # Draw bounding boxes with labels on the frame
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {i}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def update_scores(faces, phone_boxes, hands_near_face_dict, now, frame, hand_boxes=None):
    """
    Update cheating scores based on detections and log events for suspicious activities only.
    faces: list of dicts with keys 'id', 'bbox', 'pitch', 'yaw'
    phone_boxes: list of phone bounding boxes
    hands_near_face_dict: dict mapping face_id -> bool (True if hand near that face)
    now: current timestamp float (time.time())
    frame: current frame image (numpy array)
    hand_boxes: list of bounding boxes for hands, optional
    """
    timestamp_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

    for face in faces:
        face_id = face['id']
        min_x, min_y, max_x, max_y = face['bbox']
        pitch = face['pitch']
        yaw = face['yaw']
        suspicion_level = 0.0
        is_glance = False
        suspicious = False

        # Check glance (yaw/pitch angles)
        if abs(yaw) > 40:
            suspicion_level += 0.3
            is_glance = True
        if pitch < -30:
            suspicion_level += 0.3
            is_glance = True

        if is_glance:
            glance_timestamps[face_id].append(now)
            while glance_timestamps[face_id] and now - glance_timestamps[face_id][0] > glance_window_seconds:
                glance_timestamps[face_id].popleft()

            if len(glance_timestamps[face_id]) >= glance_threshold:
                suspicion_level = 1.0
                suspicious = True
                x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
                cropped_face = frame[y1:y2, x1:x2]
                log_event(timestamp_str, face_id, "Looking around frequently", "warning", cropped_face)

        # Check hands near face for this face only
        if hands_near_face_dict.get(face_id, False):
            suspicion_level += 0.5
            suspicious = True
            x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
            cropped_face = frame[y1:y2, x1:x2]

        for px1, py1, px2, py2 in phone_boxes:
            if px1 < max_x and px2 > min_x and py1 < max_y and py2 > min_y:
                suspicion_level += 0.7
                suspicious = True
                x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
                cropped_face = frame[y1:y2, x1:x2]
                log_event(timestamp_str, face_id, "Phone detected", "critical", cropped_face)
                break

        phone_near = False
        phone_near_hand = False

        for phone_box in phone_boxes:
            if is_near(phone_box, face['bbox'], max_dist=50):
                phone_near = True
            if hand_boxes:
                for hand_box in hand_boxes:
                    if is_near(phone_box, hand_box, max_dist=50):
                        phone_near_hand = True
                        break
                if phone_near_hand:
                    break

        if phone_near_hand:
            suspicion_level += 0.9
            suspicious = True
            x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
            cropped_face = frame[y1:y2, x1:x2]
            log_event(timestamp_str, face_id, "Phone detected NEAR HAND", "critical", cropped_face)

        elif phone_near:
            suspicion_level += 0.7
            suspicious = True
            x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
            cropped_face = frame[y1:y2, x1:x2]
            log_event(timestamp_str, face_id, "Phone detected near face", "critical", cropped_face)

        if hands_near_face_dict.get(face_id, False):
            if hands_on_face_start[face_id] is None:
                hands_on_face_start[face_id] = now
            elapsed = now - hands_on_face_start[face_id]
            if elapsed > 3.0:
                suspicion_level += 0.5
                suspicious = True
                x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
                cropped_face = frame[y1:y2, x1:x2]
        else:
            hands_on_face_start[face_id] = None

        suspicion_level = min(1.0, suspicion_level)

        alpha = 0.2
        prev_score = cheating_scores[face_id]
        new_score = prev_score * (1 - alpha) + suspicion_level * 100 * alpha

        if len(glance_timestamps[face_id]) >= glance_threshold:
            new_score = min(100, new_score + 10)

        if not suspicious:
            time_since_last = now - last_suspicious_time[face_id]
            decay_amount = 0.1 * time_since_last
            new_score = max(0, new_score - decay_amount)
        else:
            last_suspicious_time[face_id] = now

        cheating_scores[face_id] = max(0, min(100, new_score))

        # ====== [NEW LOGIC START] ======
        if cheating_scores[face_id] > 85:
            x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
            cropped_face = frame[y1:y2, x1:x2]
            log_event(timestamp_str, face_id, "CHEATING LIKELY", "critical", cropped_face)
        elif cheating_scores[face_id] > 50:
            x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
            cropped_face = frame[y1:y2, x1:x2]
            log_event(timestamp_str, face_id, "Suspicious behavior", "warning", cropped_face)
        # ====== [NEW LOGIC END] ======

        print(f"[DEBUG] Face {face_id}: suspicion={suspicion_level:.2f}, score={cheating_scores[face_id]:.2f}")

def visualize(frame, faces):
    for face in faces:
        face_id = face['id']
        min_x, min_y, max_x, max_y = face['bbox']
        score = cheating_scores[face_id]

        if score > 85:
            color = (0, 0, 255)
            label = f"Face {face_id} - CHEATING LIKELY! {int(score)}%"
        elif score > 50:
            color = (0, 255, 255)
            label = f"Face {face_id} - Suspicious {int(score)}%"
        else:
            color = (0, 255, 0)
            label = f"Face {face_id} - {int(score)}%"

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)
        text_y = max(min_y - 10, 0)
        cv2.putText(frame, label, (min_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
