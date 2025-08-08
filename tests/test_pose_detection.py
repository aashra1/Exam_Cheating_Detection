import os
import sys
import cv2
import pytest
import numpy as np

# Add project root to sys.path to import detection.pose_detection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.pose_detection import init_pose, hands_near_faces

def skip_if_missing_video(video_path):
    if not os.path.exists(video_path):
        pytest.skip(f"Video not found: {video_path}")

@pytest.fixture(scope="module")
def pose_model():
    model_path = os.path.join("models", "yolov8s-pose.pt")
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    return init_pose()

def test_hands_near_faces(pose_model):
    # Prepare dummy face bounding boxes for testing
    dummy_faces = [
        {'id': 0, 'bbox': (200, 100, 300, 200)},
        {'id': 1, 'bbox': (400, 120, 500, 220)},
    ]

    video_path = os.path.join(os.path.dirname(__file__), "test_vid", "object_test.mp4")
    skip_if_missing_video(video_path)

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    assert success, "Failed to read frame from video"

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_flags = hands_near_faces(pose_model, rgb_frame, dummy_faces)

    assert isinstance(hands_flags, dict), "Return value should be a dictionary"
    assert set(hands_flags.keys()) == {0, 1}, "Keys should match dummy face IDs"
    assert all(isinstance(v, bool) for v in hands_flags.values()), "All values should be boolean"
