import cv2
import os
import pytest
import sys

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.object_detection import load_model, detect_phones

@pytest.fixture(scope="module")
def phone_model():
    model_path = os.path.join("models", "yolov5su.pt")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    return load_model(model_path)

def test_phone_detection_from_video(phone_model):
    video_path = os.path.join(os.path.dirname(__file__), "test_vid", "object_test.mp4")
    assert os.path.exists(video_path), f"Video not found at {video_path}"

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    assert success, "Failed to read frame from video"

    phone_boxes = detect_phones(phone_model, frame)

    assert isinstance(phone_boxes, list), "Output must be a list"
    for box in phone_boxes:
        assert len(box) == 4, "Each box should be a tuple of 4 coordinates"
        x1, y1, x2, y2 = box
        assert x2 > x1 and y2 > y1, "Invalid box dimensions"
