import os
import cv2
import numpy as np
import pytest
import tempfile

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import your upload functions
from Backend import cloudinary_config  # Ensure config is loaded for cloudinary
from Backend.cloud_uploader import (  # Replace 'your_module' with actual module name
    upload_image_to_cloudinary,
    upload_video_to_cloudinary,
    upload_video_clip_from_frames,
)

@pytest.fixture
def dummy_image():
    # Create a simple black square image
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def dummy_video_path():
    # Create a temporary small video file to simulate video input
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video.close()

    # Create a 1-second black video at 20fps with OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, 20.0, (100, 100))
    for _ in range(20):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    yield temp_video.name

    # Cleanup
    os.remove(temp_video.name)

def test_upload_image_to_cloudinary_with_path(dummy_image):
    # Save dummy image to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, dummy_image)
        tmp_path = tmp.name

    url = upload_image_to_cloudinary(tmp_path, public_id="test_image", class_id="TEST", face_id="face1")
    os.remove(tmp_path)
    assert url is not None and url.startswith("http"), "Image upload failed or invalid URL returned"

def test_upload_image_to_cloudinary_with_array(dummy_image):
    url = upload_image_to_cloudinary(dummy_image, public_id="test_image_array", class_id="TEST", face_id="face2")
    assert url is not None and url.startswith("http"), "Image array upload failed or invalid URL returned"

def test_upload_video_to_cloudinary_valid_path(dummy_video_path):
    url = upload_video_to_cloudinary(dummy_video_path, public_id="test_video", class_id="TEST", face_id="face3")
    assert url is not None and url.startswith("http"), "Video upload failed or invalid URL returned"

def test_upload_video_to_cloudinary_invalid_path():
    url = upload_video_to_cloudinary("non_existent_file.mp4", public_id="fail_video", class_id="TEST", face_id="face4")
    assert url is None, "Invalid video path should return None"

def test_upload_video_clip_from_frames(dummy_image):
    frames = [dummy_image] * 10  # 10 frames of black image
    url = upload_video_clip_from_frames(frames, class_id="TEST", face_id="face5", fps=10)
    assert url is not None and url.startswith("http"), "Video clip upload failed or invalid URL returned"

def test_upload_video_clip_from_frames_empty():
    url = upload_video_clip_from_frames([], class_id="TEST", face_id="face6")
    assert url is None, "Empty frames should return None"
