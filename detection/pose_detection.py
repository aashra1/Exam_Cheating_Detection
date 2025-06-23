import mediapipe as mp

mp_pose = mp.solutions.pose

def init_pose():
    return mp_pose.Pose(static_image_mode=False)

def hands_near_face(pose_detector, rgb_frame):
    pose_results = pose_detector.process(rgb_frame)
    if not pose_results.pose_landmarks:
        return False
    lm = pose_results.pose_landmarks.landmark
    nose = lm[mp_pose.PoseLandmark.NOSE]
    left_hand = lm[mp_pose.PoseLandmark.LEFT_WRIST]
    right_hand = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    for hand in [left_hand, right_hand]:
        if abs(hand.x - nose.x) < 0.15 and abs(hand.y - nose.y) < 0.15:
            return True
    return False
