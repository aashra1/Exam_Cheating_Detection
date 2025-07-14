import time
from collections import defaultdict, deque
import cv2
from datetime import datetime
import numpy as np

from utils.async_logger import enqueue_log
from Backend.cloud_uploader import upload_video_clip_from_frames

DEBUG_MODE = False

rolling_window_seconds = 10
frame_interval = 0.5
LOG_COOLDOWN_SECONDS = 10

cheating_scores = defaultdict(float)
pose_only_scores = defaultdict(float)
last_suspicious_time = defaultdict(lambda: 0)
glance_timestamps = defaultdict(lambda: deque())
hands_on_face_start = defaultdict(lambda: None)
face_frame_buffer = defaultdict(lambda: deque(maxlen=60))
_last_log_time = defaultdict(lambda: defaultdict(lambda: 0))

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (5, 11), (6, 12),
    (11, 13), (12, 14)
]

def clamp_bbox(bbox, frame_shape):
    h, w = frame_shape[:2]
    x1 = max(0, min(bbox[0], w - 1))
    y1 = max(0, min(bbox[1], h - 1))
    x2 = max(0, min(bbox[2], w - 1))
    y2 = max(0, min(bbox[3], h - 1))
    return x1, y1, x2, y2

def is_valid_phone_box(box, min_area=1000, min_aspect=0.4, max_aspect=2.5):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return False
    aspect_ratio = height / width
    area = width * height
    return min_aspect < aspect_ratio < max_aspect and area >= min_area

def log_event(timestamp_str, face_id, activity, severity, cropped_face=None, class_id="LR-10", video_clip=None):
    valid_activities = {
        "Looking around frequently", "Phone detected", "Phone detected NEAR HAND",
        "Phone detected near face", "Suspicious behavior", "CHEATING LIKELY", "Turned back detected",
        "Phone detected (no face nearby)"
    }
    if activity not in valid_activities or severity not in ["warning", "critical"]:
        return
    now = time.time()
    if now - _last_log_time[face_id][activity] < LOG_COOLDOWN_SECONDS:
        return
    _last_log_time[face_id][activity] = now
    enqueue_log(timestamp_str, face_id, activity, severity, cropped_face, class_id, video_clip)

def boxes_intersect(b1, b2):
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

def is_near(box1, box2, max_dist=50):
    x1c = (box1[0] + box1[2]) / 2
    y1c = (box1[1] + box1[3]) / 2
    x2c = (box2[0] + box2[2]) / 2
    y2c = (box2[1] + box2[3]) / 2
    dist = ((x1c - x2c)**2 + (y1c - y2c)**2)**0.5
    return dist < max_dist

def is_turned_back(pose_keypoints):
    try:
        nose = pose_keypoints[0]
        left_shoulder = pose_keypoints[5]
        right_shoulder = pose_keypoints[6]
    except IndexError:
        return False
    if nose[2] < 0.3:
        return True
    if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
        nx = nose[0]
        lx = left_shoulder[0]
        rx = right_shoulder[0]
        if nx < min(lx, rx) or nx > max(lx, rx):
            return True
    return False

def update_scores(faces, phone_boxes, hands_near_face_dict, now, frame, hand_boxes=None, pose_keypoints_list=None):
    timestamp_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

    if pose_keypoints_list is None:
        pose_keypoints_list = []

    for pose_kpts in pose_keypoints_list:
        if is_turned_back(pose_kpts):
            valid_pts = pose_kpts[pose_kpts[:, 2] > 0.3][:, :2]
            if len(valid_pts) > 0:
                x1, y1 = np.min(valid_pts, axis=0).astype(int)
                x2, y2 = np.max(valid_pts, axis=0).astype(int)
                x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), frame.shape)
                cropped_pose = frame[y1:y2, x1:x2]

    face_centers = {face['id']: np.array([(face['bbox'][0] + face['bbox'][2]) / 2, (face['bbox'][1] + face['bbox'][3]) / 2]) for face in faces}

    pose_to_face_id = {}
    unmatched_poses = []

    for i, pose_kpts in enumerate(pose_keypoints_list):
        valid_kpts = pose_kpts[pose_kpts[:, 2] > 0.3][:, :2]
        if len(valid_kpts) == 0:
            unmatched_poses.append(i)
            continue
        pose_center = np.mean(valid_kpts, axis=0)

        closest_face_id = None
        min_dist = float('inf')
        for face_id, center in face_centers.items():
            dist = np.linalg.norm(pose_center - center)
            if dist < min_dist:
                min_dist = dist
                closest_face_id = face_id

        if min_dist < 100:
            pose_to_face_id[i] = closest_face_id
        else:
            unmatched_poses.append(i)

    for face in faces:
        face_id = face['id']
        min_x, min_y, max_x, max_y = face['bbox']
        pitch = face.get('pitch', 0)
        yaw = face.get('yaw', 0)
        suspicion_level = 0.0
        is_glance = False
        suspicious = False

        face_frame_buffer[face_id].append(frame.copy())

        if abs(yaw) > 60 or pitch < -40:
            suspicion_level += 0.15
            is_glance = True
            suspicious = True

        if is_glance:
            glance_timestamps[face_id].append(now)
            while glance_timestamps[face_id] and now - glance_timestamps[face_id][0] > rolling_window_seconds:
                glance_timestamps[face_id].popleft()

            if len(glance_timestamps[face_id]) >= 6:
                suspicion_level = 1.0
                suspicious = True
                x1, y1, x2, y2 = clamp_bbox(face['bbox'], frame.shape)
                cropped_face = frame[y1:y2, x1:x2]
                log_event(timestamp_str, face_id, "Looking around frequently", "warning", cropped_face)

        if hands_near_face_dict.get(face_id, False):
            suspicion_level += 0.4
            suspicious = True

        for (px1, py1, px2, py2) in phone_boxes:
            if not is_valid_phone_box((px1, py1, px2, py2)):
                continue
            if boxes_intersect((px1, py1, px2, py2), face['bbox']):
                suspicion_level += 0.7
                suspicious = True
                cropped_face = frame[min_y:max_y, min_x:max_x]
                log_event(timestamp_str, face_id, "Phone detected", "critical", cropped_face)
                break

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

        if hands_near_face_dict.get(face_id, False):
            if hands_on_face_start[face_id] is None:
                hands_on_face_start[face_id] = now
            elif now - hands_on_face_start[face_id] > 3.0:
                suspicion_level += 0.4
                suspicious = True
        else:
            hands_on_face_start[face_id] = None

        for pose_idx, f_id in pose_to_face_id.items():
            if f_id == face_id:
                pose_kpts = pose_keypoints_list[pose_idx]
                if is_turned_back(pose_kpts):
                    suspicion_level += 0.7
                    suspicious = True
                    cropped_face = frame[min_y:max_y, min_x:max_x]
                    log_event(timestamp_str, face_id, "Turned back detected", "warning", cropped_face)

        suspicion_level = min(1.0, suspicion_level)
        alpha = 0.5
        prev_score = cheating_scores[face_id]
        new_score = prev_score * (1 - alpha) + suspicion_level * 100 * alpha

        if len(glance_timestamps[face_id]) >= 6:
            new_score = min(100, new_score + 10)

        if not suspicious:
            time_since_last = now - last_suspicious_time[face_id]
            decay_amount = 5.0 * time_since_last
            new_score = max(0, new_score - decay_amount)
        else:
            last_suspicious_time[face_id] = now

        cheating_scores[face_id] = max(0, min(100, new_score))

        if cheating_scores[face_id] > 85:
            cropped_face = frame[min_y:max_y, min_x:max_x]
            video_clip = list(face_frame_buffer[face_id])
            video_url = upload_video_clip_from_frames(video_clip, face_id)
            log_event(timestamp_str, face_id, "CHEATING LIKELY", "critical", cropped_face, class_id="LR-10", video_clip=video_url)
        elif cheating_scores[face_id] > 50:
            cropped_face = frame[min_y:max_y, min_x:max_x]
            log_event(timestamp_str, face_id, "Suspicious behavior", "warning", cropped_face)

    for i in unmatched_poses:
        pose_id = f"pose_only_{i}"
        pose_kpts = pose_keypoints_list[i]
        suspicion_level = 0.0
        if is_turned_back(pose_kpts):
            suspicion_level += 0.7
            cropped_pose = None
            valid_pts = pose_kpts[pose_kpts[:, 2] > 0.3][:, :2]
            if len(valid_pts) > 0:
                x1, y1 = np.min(valid_pts, axis=0).astype(int)
                x2, y2 = np.max(valid_pts, axis=0).astype(int)
                x1, y1, x2, y2 = clamp_bbox((x1, y1, x2, y2), frame.shape)
                cropped_pose = frame[y1:y2, x1:x2]
            pose_only_scores[pose_id] = min(100, pose_only_scores.get(pose_id, 0) * 0.8 + suspicion_level * 100 * 0.2)
            if pose_only_scores[pose_id] > 85:
                log_event(timestamp_str, pose_id, "CHEATING LIKELY", "critical", cropped_pose)
            elif pose_only_scores[pose_id] > 50:
                log_event(timestamp_str, pose_id, "Suspicious behavior", "warning", cropped_pose)

    for phone_box in phone_boxes:
        if not is_valid_phone_box(phone_box):
            continue
        phone_logged = False
        for face in faces:
            if is_near(phone_box, face['bbox']):
                phone_logged = True
                break
        if not phone_logged:
            x1, y1, x2, y2 = clamp_bbox(phone_box, frame.shape)
            cropped_phone = frame[y1:y2, x1:x2]
            log_event(timestamp_str, face_id="phone_only", activity="Phone detected (no face nearby)", severity="warning", cropped_face=cropped_phone)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Phone?", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx, ty = min_x, max(min_y - 10, th + 5)
        cv2.rectangle(frame, (tx, ty - th - bl), (tx + tw, ty + bl), color, thickness=cv2.FILLED)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    y_offset = 50
    for pose_id, score in pose_only_scores.items():
        if score > 50:
            label = f"{pose_id} - Pose Suspicious {int(score)}%"
            color = (0, 165, 255)
            cv2.putText(frame, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
 