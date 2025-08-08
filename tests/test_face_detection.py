import cv2
import os
import pytest

import sys
# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.face_detection import init_face_mesh, get_faces


from detection.face_detection import init_face_mesh, get_faces

@pytest.fixture(scope="module")
def face_detectors():
    yolo_model, face_mesh = init_face_mesh()
    return yolo_model, face_mesh

def test_get_faces_from_video(face_detectors):
    yolo_model, face_mesh = face_detectors

    video_path = "videos/cheating_video5.mp4"
    assert os.path.exists(video_path), f"Video not found at {video_path}"

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    assert success, "Failed to read frame from video"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb_frame.shape[:2]

    faces = get_faces(yolo_model, face_mesh, rgb_frame, w, h)

    assert isinstance(faces, list), "get_faces() should return a list"
    assert len(faces) > 0, "At least one face should be detected in the frame"

    for face in faces:
        assert 'id' in face
        assert 'bbox' in face and len(face['bbox']) == 4
        assert 'pitch' in face and isinstance(face['pitch'], float)
        assert 'yaw' in face and isinstance(face['yaw'], float)
        assert 'roll' in face and isinstance(face['roll'], float)