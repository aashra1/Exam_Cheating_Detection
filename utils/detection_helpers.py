# utils/detection_helpers.py

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def merge_pose_to_tracked(tracked_faces, detected_faces):
    merged = []
    for t in tracked_faces:
        t_box = t['bbox']
        best_match = None
        best_iou = 0
        for d in detected_faces:
            d_box = d['bbox']
            iou = compute_iou(t_box, d_box)
            if iou > best_iou:
                best_iou = iou
                best_match = d
        if best_match is not None:
            merged.append({
                'id': t['id'],
                'bbox': tuple(map(int, best_match['bbox'])),
                'pitch': best_match.get('pitch', 0),
                'yaw': best_match.get('yaw', 0),
                'roll': best_match.get('roll', 0),
                'landmarks': best_match.get('landmarks', None)
            })
        else:
            merged.append({
                'id': t['id'],
                'bbox': tuple(map(int, t_box)),
                'pitch': 0,
                'yaw': 0,
                'roll': 0,
                'landmarks': None
            })
    return merged
