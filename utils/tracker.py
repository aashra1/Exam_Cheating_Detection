from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Initialize DeepSort tracker with tuned parameters for stability
tracker = DeepSort(max_age=60, max_cosine_distance=0.15, n_init=3)

def get_tracked_faces(frame, detections):
    """
    detections: list of tuples/lists:
      ([x1, y1, x2, y2], conf, embedding=None)
      embedding is optional, default None
    """
    deep_sort_detections = []

    for det in detections:
        if not (isinstance(det, (list, tuple)) and len(det) >= 2):
            print(f"Skipping invalid detection (wrong format): {det}")
            continue

        bbox = det[0]
        conf = det[1]
        embedding = det[2] if len(det) > 2 else None

        # Validate bbox
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            print(f"Skipping invalid bbox (expected length 4): {bbox}")
            continue

        if embedding is not None:
            embedding = np.array(embedding)

        deep_sort_detections.append((bbox, conf, embedding))

    tracks = tracker.update_tracks(deep_sort_detections, frame=frame)

    tracked_faces = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()
        bbox = list(map(int, bbox))
        tracked_faces.append({
            'id': track_id,
            'bbox': bbox
        })
    return tracked_faces
