import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace
import math

model = YOLO("yolov8m.pt")
vehicle_class_ids = [2, 3, 5, 7]  # Class IDs for vehicles
max_history = 30  # Max length of trail history
motion_threshold = 60.0  # Minimum movement distance for drawing trails
min_trail_length = 120  # Minimum trail length to log

video_path = r'C:\Users\malid\OneDrive\Belgeler\GitHub\thesis-2025\video footage\video.MP4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Estimate how many meters one pixel represents
meters_per_pixel = 3.5 / 150

# Initialize tracker
byte_cfg = Namespace(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=fps,
    mot20=False,
)
tracker = BYTETracker(byte_cfg, frame_rate=fps)

# Tracking variables
track_trails = {}
object_speeds = defaultdict(float)
vehicle_data = defaultdict(list)
object_id_mapping = {}
true_next_object_id = 0

# Open a file to save the IDs and coordinates
output_file = open("vehicle_data_bytetrack.csv", "w")
output_file.write("ID, X, Y, km/h\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in vehicle_class_ids:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append([x1, y1, x2, y2, conf])

    dets_np = np.array(detections, dtype=np.float32)
    online_targets = tracker.update(dets_np, [frame.shape[0], frame.shape[1]], frame.shape[:2])

    for track in online_targets:
        if not track.is_activated:
            continue

        track_id = track.track_id
        x1, y1, w, h = map(int, track.tlwh)
        x2, y2 = x1 + w, y1 + h
        cx, cy = x1 + w // 2, y1 + h // 2

        if track_id not in track_trails:
            track_trails[track_id] = deque(maxlen=max_history)
        trail = track_trails[track_id]
        trail.append((cx, cy))

        if len(trail) >= 2:
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-2][1] - trail[-1][1]

            distance = (dx**2 + dy**2) ** 0.5
            speed_pxps = (distance / len(trail)) * fps
            speed_kmph = speed_pxps * meters_per_pixel * 3.6
            object_speeds[track_id] = speed_kmph

            if distance > motion_threshold and len(trail) >= min_trail_length:
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id} ({cx},{cy}) {speed_kmph:.1f} km/h",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                # Assign logging ID
                if track_id not in object_id_mapping:
                    object_id_mapping[track_id] = true_next_object_id
                    true_next_object_id += 1

                vehicle_data[track_id].append((cx, cy, speed_kmph))

    resized = cv2.resize(frame, (1280, 720))
    cv2.imshow('YOLOv8 + ByteTrack', resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Sort and write vehicle data by ID
for obj_id in sorted(vehicle_data):
    if len(vehicle_data[obj_id]) >= min_trail_length:
        logging_id = object_id_mapping[obj_id]
        for x, y, speed in vehicle_data[obj_id]:
            output_file.write(f"{logging_id}, {x}, {y}, {speed:.2f}\n")

output_file.close()
