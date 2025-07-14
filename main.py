import cv2
import time
import os
import numpy as np

from detection import face_detection, object_detection, pose_detection
from utils import cheating_logic
from utils import tracker
from detection.pose_detection import draw_pose

DEBUG_MODE = True

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea) if areaA + areaB - interArea > 0 else 0

def merge_pose_to_tracked(tracked_faces, detected_faces):
    merged = []
    for t in tracked_faces:
        best_match, best_iou = None, 0
        for d in detected_faces:
            iou = compute_iou(t['bbox'], d['bbox'])
            if iou > best_iou:
                best_iou, best_match = iou, d
        merged.append({
            'id': t['id'],
            'bbox': tuple(map(int, best_match['bbox'])) if best_match else tuple(map(int, t['bbox'])),
            'pitch': best_match.get('pitch', 0) if best_match else 0,
            'yaw': best_match.get('yaw', 0) if best_match else 0,
            'roll': best_match.get('roll', 0) if best_match else 0,
            'landmarks': None
        })
    return merged

def main():
    print("Loading models...")
    yolo_model = object_detection.load_model('models/yolov5su.pt').to("cuda")
    yolo_face_model, face_mesh = face_detection.init_face_mesh()
    pose_detector = pose_detection.init_pose()
    print("Models loaded.")

    video_path = 'videos/cheating_video4.mp4'
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open the video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / fps if fps > 0 else 1 / 30
    print(f"Video FPS: {fps}, frame duration: {frame_duration:.3f}s")

    tracked_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        display = frame.copy()

        try:
            phone_boxes = object_detection.detect_phones(yolo_model, frame)
            if DEBUG_MODE:
                print(f"[DEBUG] Phone boxes: {phone_boxes}")

            faces = face_detection.get_faces(yolo_face_model, face_mesh, rgb, w, h)
            if DEBUG_MODE:
                print(f"[DEBUG] Faces detected: {len(faces)}")

            face_detections = [
                ([face['bbox'][0], face['bbox'][1], face['bbox'][2], face['bbox'][3]], 1.0, 0)
                for face in faces
            ]
            tracked_faces = tracker.get_tracked_faces(frame, face_detections)
            tracked_faces = merge_pose_to_tracked(tracked_faces, faces)

            hands_near = pose_detection.hands_near_faces(pose_detector, rgb, tracked_faces)
            hands_near_face_dict = {face['id']: hands_near.get(face['id'], False) for face in tracked_faces}
            if DEBUG_MODE:
                print(f"[DEBUG] Hands near face dict: {hands_near_face_dict}")

            # ✅ Improved pose detection parsing
            pose_results = pose_detector(frame)
            pose_keypoints_list = []
            try:
                for result in pose_results:
                    if result.keypoints is not None and result.keypoints.xy is not None:
                        xy = result.keypoints.xy.cpu().numpy()
                        conf = result.keypoints.conf.cpu().numpy()
                        for i in range(xy.shape[0]):
                            keypoints = np.hstack((xy[i], conf[i][:, np.newaxis]))  # shape (17, 3)
                            pose_keypoints_list.append(keypoints)
            except Exception as e:
                if DEBUG_MODE:
                    print(f"[DEBUG] Error parsing pose keypoints: {e}")

            for i, keypoints in enumerate(pose_keypoints_list):
                confs = keypoints[:, 2]
                avg_conf = confs.mean()
                print(f"[DEBUG] Pose {i} confidence: avg={avg_conf:.2f}")

        except Exception as e:
            print(f"[❌] Detection error: {e}")
            continue

        cheating_logic.update_scores(
            tracked_faces,
            phone_boxes,
            hands_near_face_dict,
            now,
            display,
            hand_boxes=None,
            pose_keypoints_list=pose_keypoints_list
        )

        for i, keypoints in enumerate(pose_keypoints_list):
            if i < len(tracked_faces):
                draw_pose(display, keypoints, color=(0, 255, 255))
            else:
                draw_pose(display, keypoints, color=(0, 165, 255))

        for (x1, y1, x2, y2) in phone_boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cheating_logic.visualize(display, tracked_faces)

        if DEBUG_MODE:
            debug_text = f"Faces: {len(tracked_faces)} | Phones: {len(phone_boxes)}"
            cv2.putText(display, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Cheating Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
