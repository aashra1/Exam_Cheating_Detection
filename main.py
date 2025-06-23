import cv2
import time
from detection import face_detection, object_detection, pose_detection
from utils import cheating_logic  # <-- Import your cheating logic here

def main():
    # Initialize models
    yolo_model = object_detection.load_model('models/yolov5s.pt')
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

        phone_boxes = object_detection.detect_phones(yolo_model, frame)
        hands_near_face = pose_detection.hands_near_face(pose_detector, rgb_frame)
        faces = face_detection.get_faces(face_mesh, rgb_frame, w, h)

        if now - last_update_time > frame_interval:
            cheating_logic.update_scores(faces, phone_boxes, hands_near_face, now)
            last_update_time = now

        cheating_logic.visualize(frame, faces)

        cv2.imshow("Cheating Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
