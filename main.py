import cv2
import time
from detection import face_detection, object_detection, pose_detection
from utils import cheating_logic
from utils import tracker  # DeepSort tracker wrapper


# === Compute Intersection over Union (IoU) ===
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0


# === Match detected face poses to tracked faces (by IoU) ===
def merge_pose_to_tracked(tracked_faces, detected_faces):
    merged = []
    for t in tracked_faces:
        t_box = t['bbox']
        best_match = None
        best_iou = 0
        for d in detected_faces:
            d_box = d['bbox']
            iou = compute_iou(t_box, d_box)
            if iou > best_iou:
                best_iou = iou
                best_match = d
        if best_match is not None:
            merged.append({
                'id': t['id'],
                'bbox': tuple(map(int, best_match['bbox'])),
                'pitch': best_match.get('pitch', 0),
                'yaw': best_match.get('yaw', 0),
                'roll': best_match.get('roll', 0),
                'landmarks': best_match.get('landmarks', None)
            })
        else:
            merged.append({
                'id': t['id'],
                'bbox': tuple(map(int, t_box)),
                'pitch': 0,
                'yaw': 0,
                'roll': 0,
                'landmarks': None
            })
    return merged


def main():
    # === Load models ===
    yolo_model = object_detection.load_model('models/yolov5su.pt')
    face_mesh = face_detection.init_face_mesh()
    pose_detector = pose_detection.init_pose()

    cap = cv2.VideoCapture(0)
    last_update_time = 0
    frame_interval = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.time()

        # === Detect phones ===
        phone_boxes = object_detection.detect_phones(yolo_model, frame)

        # === Detect faces ===
        faces = face_detection.get_faces(face_mesh, rgb_frame, w, h)

        # === Prepare detections for tracking ===
        face_detections = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            face_detections.append(([x1, y1, x2, y2], 1.0, 0))  # dummy confidence and class_id

        tracked_faces = tracker.get_tracked_faces(frame, face_detections)
        tracked_faces = merge_pose_to_tracked(tracked_faces, faces)

        # === Detect hands near face ===
        hands_near = pose_detection.hands_near_face(pose_detector, rgb_frame)
        hands_near_face_dict = {face['id']: hands_near for face in tracked_faces}

        # === Run cheating detection logic ===
        if now - last_update_time > frame_interval:
            cheating_logic.update_scores(
                tracked_faces,
                phone_boxes,
                hands_near_face_dict,
                now,
                frame
            )
            last_update_time = now

        # === Draw phone boxes (debug only) ===
        for (x1, y1, x2, y2) in phone_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # === Draw bounding boxes, scores ===
        cheating_logic.visualize(frame, tracked_faces)

        # === Show output ===
        cv2.imshow("Cheating Detection - Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()