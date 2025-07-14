import threading
import queue
import time
import tempfile
import os
import cv2
import numpy as np
from collections import defaultdict
from Backend import db
from Backend.cloud_uploader import upload_image_to_cloudinary, upload_video_to_cloudinary

log_queue = queue.Queue()
_last_log_time = defaultdict(lambda: defaultdict(lambda: 0))
LOG_COOLDOWN_SECONDS = 10

def enqueue_log(timestamp_str, face_id, activity, severity, cropped_face=None, class_id="LR-10", video_clip=None):
    valid_activities = {
        "Looking around frequently",
        "Phone detected",
        "Phone detected NEAR HAND",
        "Phone detected near face",
        "Suspicious behavior",
        "CHEATING LIKELY",
        "Turned back detected",
        "Phone detected (no face nearby)"
    }

    if activity not in valid_activities or severity not in ["warning", "critical"]:
        return

    now = time.time()
    if now - _last_log_time[face_id][activity] < LOG_COOLDOWN_SECONDS:
        return
    _last_log_time[face_id][activity] = now

    log_queue.put((timestamp_str, face_id, activity, severity, cropped_face, class_id, video_clip))
    print(f"[ASYNC] Enqueued log for face {face_id}, activity={activity}")

def logging_worker():
    while True:
        item = log_queue.get()
        if item is None:
            break

        timestamp_str, face_id, activity, severity, cropped_face, class_id, video_clip = item
        image_url, video_url = None, None
        suffix = f"{timestamp_str.replace(':', '-').replace(' ', '_')}_face{face_id}"

        # ======= Upload Image to Cloudinary =======
        if cropped_face is not None:
            try:
                image_url = upload_image_to_cloudinary(
                    cropped_face,
                    public_id=suffix,
                    tags=[class_id, f"face_{face_id}", activity, severity],
                    class_id=class_id,
                    face_id=face_id
                )
                if not image_url:
                    print("[Cloudinary] Image upload failed.")
            except Exception as e:
                print(f"[Cloudinary Image Upload Error] {e}")
                image_url = None

        # ======= Upload Video to Cloudinary or Use Existing URL =======
        if video_clip is not None:
            if isinstance(video_clip, list) and len(video_clip) > 0:
                try:
                    height, width = video_clip[0].shape[:2]
                    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
                    os.close(fd)

                    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
                    for frame in video_clip:
                        out.write(frame)
                    out.release()

                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                        video_url = upload_video_to_cloudinary(
                            temp_path,
                            public_id=suffix,
                            tags=[class_id, f"face_{face_id}", activity, severity],
                            class_id=class_id,
                            face_id=face_id
                        )
                        print(f"[DEBUG] Uploaded video URL: {video_url}")
                    else:
                        print(f"[ERROR] Video file not saved properly or too small: {temp_path}")

                    os.remove(temp_path)

                except Exception as e:
                    print(f"[Cloudinary Video Upload Error] {e}")
                    video_url = None
            elif isinstance(video_clip, str) and video_clip.startswith("http"):
                video_url = video_clip

        # ======= Type Safety Before DB =======
        if isinstance(image_url, np.ndarray):
            print(f"[CRITICAL FIX] image_url was ndarray, setting to None for face {face_id}")
            image_url = None
        if isinstance(video_url, list) or (
            hasattr(video_url, '__len__') and len(video_url) > 0 and isinstance(video_url[0], np.ndarray)
        ):
            print(f"[CRITICAL FIX] video_url is raw frames! Clearing before DB insert for face {face_id}")
            video_url = None

        print(f"[DEBUG] enqueue_log: image_url={image_url}, video_url={video_url}")

        # ======= Log to Database =======
        try:
            db.insert_log(
                class_id=class_id,
                face_id=f"S{int(face_id):03d}" if str(face_id).isdigit() else face_id,
                activity=activity,
                severity=severity,
                image_url=image_url,
                video_url=video_url
            )
            print(f"[ASYNC] Log written to DB for face {face_id}")
        except Exception as e:
            print(f"[DB ERROR] {e}")

        log_queue.task_done()

# ===== Start Background Thread =====
logging_thread = threading.Thread(target=logging_worker, daemon=True)
logging_thread.start()
