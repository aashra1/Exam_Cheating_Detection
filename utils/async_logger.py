import threading
import queue
import time
import os
import cv2
from collections import defaultdict

log_queue = queue.Queue()
_last_log_time = defaultdict(lambda: defaultdict(lambda: 0))
LOG_COOLDOWN_SECONDS = 10  # per face_id per activity

snapshot_dir = "snapshots"
os.makedirs(snapshot_dir, exist_ok=True)

def enqueue_log(timestamp_str, face_id, activity, severity, cropped_face=None, class_id="LR-10"):
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

    log_queue.put((timestamp_str, face_id, activity, severity, cropped_face, class_id))
    
    print(f"Enqueue log: face_id={face_id}, activity={activity}, severity={severity}")
    log_queue.put((timestamp_str, face_id, activity, severity, cropped_face, class_id))

def logging_worker():
    from utils import db  # Lazy import to avoid circular dependency
    while True:
        item = log_queue.get()
        if item is None:
            break

        timestamp_str, face_id, activity, severity, cropped_face, class_id = item
        image_path = None

        if cropped_face is not None:
            filename = f"{timestamp_str.replace(':', '-').replace(' ', '_')}_face{face_id}.jpg"
            image_path = os.path.join(snapshot_dir, filename)
            try:
                cv2.imwrite(image_path, cropped_face)
                print(f"Saved snapshot to {image_path}")
            except Exception as e:
                print(f"Image saving error: {e}")

        try:
            print(f"[LOGGER DEBUG] Logging event: {timestamp_str}, {face_id}, {activity}, {severity}")
            db.insert_log(
                class_id=class_id,
                face_id=f"S{int(face_id):03d}",
                activity=activity,
                severity=severity,
                image_url=image_path,
                video_url=None
            )

            print("[LOGGER DEBUG] Log inserted successfully")
        except Exception as e:
            print(f"[DB logging error: {e}")


        log_queue.task_done()

# Start logging thread
logging_thread = threading.Thread(target=logging_worker, daemon=True)
logging_thread.start()
