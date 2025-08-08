import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

import utils.tracker as tracker_module

def test_get_tracked_faces_filters_and_formats():
    # Create mock tracks
    mock_track_confirmed = MagicMock()
    mock_track_confirmed.is_confirmed.return_value = True
    mock_track_confirmed.to_tlbr.return_value = [10.5, 20.5, 110.5, 120.5]
    mock_track_confirmed.track_id = 1

    mock_track_unconfirmed = MagicMock()
    mock_track_unconfirmed.is_confirmed.return_value = False

    mock_tracks = [mock_track_confirmed, mock_track_unconfirmed]

    with patch.object(tracker_module.tracker, 'update_tracks', return_value=mock_tracks):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detections = [[10, 20, 110, 120, 0.9]]

        results = tracker_module.get_tracked_faces(frame, detections)

        # Only confirmed track should be included
        assert len(results) == 1
        assert results[0]['id'] == 1

        # bbox coordinates should be integers and truncated as expected
        assert results[0]['bbox'] == [10, 20, 110, 120]
