# utils/tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize the DeepSort tracker once globally
tracker = DeepSort(max_age=30)  # You can adjust max_age and other params as needed

def get_tracked_faces(frame, detections):
    # detections: list of [x1, y1, x2, y2, conf]
    tracks = tracker.update_tracks(detections, frame=frame)

    tracked_faces = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        # track.to_tlbr() returns bbox as (x1, y1, x2, y2)
        bbox = track.to_tlbr()  
        # convert bbox floats to ints
        bbox = list(map(int, bbox))
        tracked_faces.append({
            'id': track_id,
            'bbox': bbox
        })
    return tracked_faces

