from ultralytics import YOLO
import cv2

def load_model(model_path):
    return YOLO(model_path)

def detect_phones(model, frame):
    phone_boxes = []
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        if 'phone' in label.lower():
            phone_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return phone_boxes
