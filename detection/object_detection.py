from ultralytics import YOLO
import cv2

# === Load YOLO Model ===
def load_model(model_path, device='cpu'):
    model = YOLO(model_path)
    model.fuse()  # Optional: optimizes inference speed
    model.to(device)
    return model

# === Helper: Check if bounding box is shaped like a real phone ===
def is_valid_phone_box(box, min_area=1000, min_aspect=0.4, max_aspect=2.5):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return False
    aspect_ratio = height / width
    area = width * height
    return min_aspect < aspect_ratio < max_aspect and area >= min_area

# === Detect phones ===
def detect_phones(model, frame, conf_threshold=0.5):
    phone_boxes = []

    # Run detection
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    class_names = model.names
    phone_class_ids = [cls_id for cls_id, name in class_names.items()
                       if name.lower() in ['cell phone', 'mobile phone', 'phone']]

    # Process detections
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        if cls_id in phone_class_ids and is_valid_phone_box((x1, y1, x2, y2)):
            phone_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            # Optional: draw for debugging
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # cv2.putText(frame, f"{class_names[cls_id]} {conf:.2f}", (int(x1), int(y1) - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return phone_boxes