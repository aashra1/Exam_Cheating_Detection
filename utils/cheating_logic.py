import time
from collections import defaultdict, deque
import cv2
from datetime import datetime

from utils.async_logger import enqueue_log
from Backend.cloud_uploader import upload_video_clip_from_frames

# === Constants ===
rolling_window_seconds = 10
frame_interval = 0.5
LOG_COOLDOWN_SECONDS = 10

# === Runtime State ===
cheating_scores = defaultdict(float)
last_suspicious_time = defaultdict(lambda: 0)
glance_timestamps = defaultdict(lambda: deque())
hands_on_face_start = defaultdict(lambda: None)
face_frame_buffer = defaultdict(lambda: deque(maxlen=60))
_last_log_time = defaultdict(lambda: defaultdict(lambda: 0))

# === Utility: Clamp bounding box inside frame ===
def clamp_bbox(bbox, frame_shape):
    h, w = frame_shape[:2]
    x1 = max(0, min(bbox[0], w - 1))
    y1 = max(0, min(bbox[1], h - 1))
    x2 = max(0, min(bbox[2], w - 1))
    y2 = max(0, min(bbox[3], h - 1))
    return x1, y1, x2, y2

# === Utility: Reject false positives (e.g. remote controls) ===
def is_valid_phone_box(box, min_area=1000, min_aspect=0.4, max_aspect=2.5):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return False
    aspect_ratio = height / width
    area = width * height
    return min_aspect < aspect_ratio < max_aspect and area >= min_area

# === Wrapper for async logging with cooldown ===
def log_event(timestamp_str, face_id, activity, severity, cropped_face=None, class_id="LR-10", video_clip=None):
    valid_activities = {
        "Looking around frequently",
        "Phone detected",
        "Phone detected NEAR HAND",
        "Phone detected near face",
        "Suspicious behavior",
        "CHEATING LIKELY"
    }

    if activity not in valid_activities or severity not in ["warning", "critical"]:
        return

    now = time.time()
    if now - _last_log_time[face_id][activity] < LOG_COOLDOWN_SECONDS:
        return
    _last_log_time[face_id][activity] = now

    enqueue_log(timestamp_str, face_id, activity, severity, cropped_face, class_id, video_clip)

# === Distance between bounding box centers ===
def is_near(box1, box2, max_dist=50):
    x1c = (box1[0] + box1[2]) / 2
    y1c = (box1[1] + box1[3]) / 2
    x2c = (box2[0] + box2[2]) / 2
    y2c = (box2[1] + box2[3]) / 2
    dist = ((x1c - x2c)**2 + (y1c - y2c)**2)**0.5
    return dist < max_dist

# === Main Scoring + Logging ===
def update_scores(faces, phone_boxes, hands_near_face_dict, now, frame, hand_boxes=None):
    timestamp_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

    for face in faces:
        face_id = face['id']
        min_x, min_y, max_x, max_y = face['bbox']
        pitch = face['pitch']
        yaw = face['yaw']
        suspicion_level = 0.0
        is_glance = False
        suspicious = False

        face_frame_buffer[face_id].append(frame.copy())

        # === Gaze suspicion ===
        if abs(yaw) > 40:
            suspicion_level += 0.3
            is_glance = True
        if pitch < -30:
            suspicion_level += 0.3
            is_glance = True

        if is_glance:
            glance_timestamps[face_id].append(now)
            while glance_timestamps[face_id] and now - glance_timestamps[face_id][0] > 10:
                glance_timestamps[face_id].popleft()

            if len(glance_timestamps[face_id]) >= 4:
                suspicion_level = 1.0
                suspicious = True
                x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
                cropped_face = frame[y1:y2, x1:x2]
                log_event(timestamp_str, face_id, "Looking around frequently", "warning", cropped_face)

        # === Hand near face ===
        if hands_near_face_dict.get(face_id, False):
            suspicion_level += 0.5
            suspicious = True

        # === Phone overlaps face ===
        for (px1, py1, px2, py2) in phone_boxes:
            if not is_valid_phone_box((px1, py1, px2, py2)):
                continue

            if px1 < max_x and px2 > min_x and py1 < max_y and py2 > min_y:
                suspicion_level += 0.7
                suspicious = True
                cropped_face = frame[min_y:max_y, min_x:max_x]
                log_event(timestamp_str, face_id, "Phone detected", "critical", cropped_face)
                break

        # === Phone near face/hand ===
        phone_near, phone_near_hand = False, False
        for phone_box in phone_boxes:
            if not is_valid_phone_box(phone_box):
                continue
            if is_near(phone_box, face['bbox']):
                phone_near = True
            if hand_boxes:
                for hand_box in hand_boxes:
                    if is_near(phone_box, hand_box):
                        phone_near_hand = True
                        break

        if phone_near_hand:
            suspicion_level += 0.9
            suspicious = True
            cropped_face = frame[min_y:max_y, min_x:max_x]
            log_event(timestamp_str, face_id, "Phone detected NEAR HAND", "critical", cropped_face)
        elif phone_near:
            suspicion_level += 0.7
            suspicious = True
            cropped_face = frame[min_y:max_y, min_x:max_x]
            log_event(timestamp_str, face_id, "Phone detected near face", "critical", cropped_face)

        # === Hands on face for too long ===
        if hands_near_face_dict.get(face_id, False):
            if hands_on_face_start[face_id] is None:
                hands_on_face_start[face_id] = now
            if now - hands_on_face_start[face_id] > 3.0:
                suspicion_level += 0.5
                suspicious = True
        else:
            hands_on_face_start[face_id] = None

        # === Update and clamp score ===
        suspicion_level = min(1.0, suspicion_level)
        alpha = 0.2
        prev_score = cheating_scores[face_id]
        new_score = prev_score * (1 - alpha) + suspicion_level * 100 * alpha

        if len(glance_timestamps[face_id]) >= 4:
            new_score = min(100, new_score + 10)

        if not suspicious:
            time_since_last = now - last_suspicious_time[face_id]
            new_score = max(0, new_score - 0.2 * time_since_last)
        else:
            last_suspicious_time[face_id] = now

        cheating_scores[face_id] = max(0, min(100, new_score))

        # === Log final classification ===
        if cheating_scores[face_id] > 85:
            cropped_face = frame[min_y:max_y, min_x:max_x]
            video_clip = list(face_frame_buffer[face_id])
            video_url = upload_video_clip_from_frames(video_clip, face_id)
            log_event(
                timestamp_str,
                face_id,
                "CHEATING LIKELY",
                "critical",
                cropped_face,
                class_id="LR-10",
                video_clip=video_url
            )
        elif cheating_scores[face_id] > 50:
            cropped_face = frame[min_y:max_y, min_x:max_x]
            log_event(timestamp_str, face_id, "Suspicious behavior", "warning", cropped_face)

        print(f"[DEBUG] Face {face_id}: suspicion={suspicion_level:.2f}, score={cheating_scores[face_id]:.2f}")

# === Draw labels and scores on frame ===
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