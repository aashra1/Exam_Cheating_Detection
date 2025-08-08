import os
import sys
import cv2
import numpy as np
import pytest
from datetime import datetime
import time

# Add project root to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.cheating_logic import update_scores, visualize, cheating_probabilities, pose_only_scores

@pytest.mark.skipif(not os.path.exists("videos/cheating_video5.mp4"), reason="Test video not found")
def test_update_scores_and_visualize():
    video_path = "videos/cheating_video5.mp4"
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Failed to open test video"

    # Dummy inputs for update_scores
    # You should replace these with actual detections or mocks
    faces = [
        {"id": 1, "bbox": (100, 100, 200, 200), "pitch": 0.0, "yaw": 0.0},
        {"id": 2, "bbox": (300, 100, 400, 200), "pitch": 0.0, "yaw": 0.0},
    ]
    phone_boxes = [
        (250, 150, 280, 180),  # Near face 2 approx
    ]
    hands_near_face_dict = {
        1: False,
        2: True,
    }
    pose_keypoints_list = [
        np.array([[320, 110, 0.9], [330, 120, 0.8], [310, 115, 0.7]]),  # Some valid pose for face 2
        np.array([[150, 150, 0.1], [160, 160, 0.1], [140, 145, 0.1]])   # Low confidence pose (ignored)
    ]

    frame_count = 0
    max_frames_to_test = 20
    ret, frame = cap.read()
    while ret and frame_count < max_frames_to_test:
        now = time.time()
        timestamp_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

        # Call update_scores - core function to test
        update_scores(faces, phone_boxes, hands_near_face_dict, now, frame, pose_keypoints_list=pose_keypoints_list)

        # Call visualize - just ensure no exceptions
        visualize(frame, faces)

        frame_count += 1
        ret, frame = cap.read()

    cap.release()

    # Check that cheating_probabilities dict contains updated models for faces
    assert 1 in cheating_probabilities, "Face ID 1 missing from cheating probabilities"
    assert 2 in cheating_probabilities, "Face ID 2 missing from cheating probabilities"

    # Check probabilities are floats and in [0,1]
    for face_id in [1, 2]:
        prob = cheating_probabilities[face_id].get_probability()
        assert isinstance(prob, float), "Probability must be a float"
        assert 0.0 <= prob <= 1.0, "Probability must be between 0 and 1"

    # Check pose_only_scores values
    for key, score in pose_only_scores.items():
        assert isinstance(score, float), "Pose only scores must be float"
        assert 0.0 <= score <= 100.0, "Pose score must be between 0 and 100"

    print("Cheating logic update_scores and visualize test passed.")

