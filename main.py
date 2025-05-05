import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict

model = YOLO("yolov8l.pt")
vehicle_class_ids = [2, 3, 5, 7]
max_history = 30
distance_threshold = 200
motion_threshold = 50.0
max_disappeared = 30

video_path = r'C:\Users\malid\Desktop\thesis 2025\video footage\video.MP4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

next_object_id = 0
tracked_objects = {}
track_trails = {}
disappeared = defaultdict(int)
object_boxes = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)[0]
    detections = []
    boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in vehicle_class_ids:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detections.append((cx, cy))
        boxes.append((x1, y1, x2, y2))

    matched_ids = set()
    new_tracked_objects = {}
    matched_detections = set()
    new_object_boxes = {}

    for obj_id, (prev_x, prev_y) in tracked_objects.items():
        min_dist = float('inf')
        best_idx = -1

        for i, det in enumerate(detections):
            if i in matched_detections:
                continue
            dist = np.linalg.norm(np.array([prev_x, prev_y]) - np.array(det))
            if dist < min_dist and dist < distance_threshold:
                min_dist = dist
                best_idx = i

        if best_idx != -1:
            matched_detections.add(best_idx)
            new_tracked_objects[obj_id] = detections[best_idx]
            new_object_boxes[obj_id] = boxes[best_idx]
            matched_ids.add(obj_id)
            disappeared[obj_id] = 0

    for obj_id in tracked_objects.keys():
        if obj_id not in matched_ids:
            disappeared[obj_id] += 1
            if disappeared[obj_id] < max_disappeared:
                new_tracked_objects[obj_id] = tracked_objects[obj_id]
                new_object_boxes[obj_id] = object_boxes.get(obj_id, (0, 0, 0, 0))
            else:
                track_trails.pop(obj_id, None)
                disappeared.pop(obj_id, None)
                object_boxes.pop(obj_id, None)

    for i, det in enumerate(detections):
        if i not in matched_detections:
            new_tracked_objects[next_object_id] = det
            new_object_boxes[next_object_id] = boxes[i]
            disappeared[next_object_id] = 0
            next_object_id += 1

    tracked_objects = new_tracked_objects
    object_boxes = new_object_boxes

    for obj_id, (x, y) in tracked_objects.items():
        if obj_id not in track_trails:
            track_trails[obj_id] = deque(maxlen=max_history)
        trail = track_trails[obj_id]
        trail.append((x, y))

        if len(trail) >= 2:
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-1][1] - trail[0][1]
            distance = (dx**2 + dy**2) ** 0.5

            if distance > motion_threshold:
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)

                x1, y1, x2, y2 = object_boxes.get(obj_id, (0, 0, 0, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

    resized = cv2.resize(frame, (1280, 720))
    cv2.imshow('vehicle detection', resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
