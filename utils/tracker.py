# utils/tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSort tracker globally (do this once to keep track states)
tracker = DeepSort(max_age=30)  # You can tweak max_age and other params if needed

def get_tracked_faces(frame, detections):
    """
    Args:
        frame: Current video frame (numpy array).
        detections: List of detections in format [x1, y1, x2, y2, confidence].

    Returns:
        List of dicts: [{'id': track_id, 'bbox': [x1, y1, x2, y2]}, ...]
    """
    tracks = tracker.update_tracks(detections, frame=frame)

    tracked_faces = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr()  # bbox in [x1, y1, x2, y2] format
        bbox = list(map(int, bbox))
        tracked_faces.append({
            'id': track.track_id,
            'bbox': bbox
        })
    return tracked_faces
