import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import math

model = YOLO("yolov8m.pt")
vehicle_class_ids = [2, 3, 5, 7]  # Class IDs for vehicles
max_history = 30  # Maximum trail history
distance_threshold = 200  # Threshold for matching detections
motion_threshold = 60  # Minimum motion distance for drawing trails
min_trail_length = 10  # Minimum trail length to consider valid movement
max_disappeared = 30  # Max frames an object can disappear

video_path = r'C:\Users\malid\OneDrive\Belgeler\GitHub\thesis-2025\video footage\video.MP4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

meters_per_pixel = 3.5 / 150

raw_next_object_id = 0  # Internal counter for raw detection IDs
true_next_object_id = 0  # Actual ID for logged objects
tracked_objects = {}  # Stores active tracked objects
track_trails = {}  # Stores trails of tracked objects
disappeared = defaultdict(int)  # Tracks how long objects have been missing
object_boxes = {}  # Stores bounding boxes of tracked objects
object_speeds = defaultdict(float)  # Stores estimated speed in km/h for each object
object_id_mapping = {}  # Maps internal IDs to logged IDs
vehicle_data = defaultdict(list)  # Store data grouped by object ID

# Open a file to save the IDs and coordinates
output_file = open("vehicle_data.csv", "w")
output_file.write("ID, X, Y, km/h\n")

# Process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the current frame
    results = model(frame, conf=0.4, verbose=False)[0]
    detections = []
    boxes = []

    # Filter detections for vehicles and extract relevant data
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in vehicle_class_ids:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detections.append((cx, cy))
        boxes.append((x1, y1, x2, y2))

    # Match detections with existing tracked objects
    matched_ids = set()
    new_tracked_objects = {}
    matched_detections = set()
    new_object_boxes = {}

    for obj_id, (prev_x, prev_y) in tracked_objects.items():
        min_dist = float('inf')
        best_idx = -1

        # Find the closest detection to the tracked object
        for i, det in enumerate(detections):
            if i in matched_detections:
                continue
            dist = np.linalg.norm(np.array([prev_x, prev_y]) - np.array(det))
            if dist < min_dist and dist < distance_threshold:
                min_dist = dist
                best_idx = i

        # Update tracking information if a match is found
        if best_idx != -1:
            matched_detections.add(best_idx)
            new_tracked_objects[obj_id] = detections[best_idx]
            new_object_boxes[obj_id] = boxes[best_idx]
            matched_ids.add(obj_id)
            disappeared[obj_id] = 0

    # Handle objects that are no longer detected
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
                object_speeds.pop(obj_id, None)

    # Add new detections as tracked objects
    for i, det in enumerate(detections):
        if i not in matched_detections:
            new_tracked_objects[raw_next_object_id] = det
            new_object_boxes[raw_next_object_id] = boxes[i]
            disappeared[raw_next_object_id] = 0
            raw_next_object_id += 1

    # Update tracking data
    tracked_objects = new_tracked_objects
    object_boxes = new_object_boxes

    # Draw trails and bounding boxes for tracked objects
    for obj_id, (x, y) in tracked_objects.items():
        if obj_id not in track_trails:
            track_trails[obj_id] = deque(maxlen=max_history)
        trail = track_trails[obj_id]
        trail.append((x, y))

        if len(trail) >= 2:
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-2][1] - trail[-1][1]

            distance = (dx**2 + dy**2) ** 0.5
            speed_pxps = (distance / len(trail)) * fps
            speed_kmph = speed_pxps * meters_per_pixel * 3.6
            object_speeds[obj_id] = speed_kmph

            if distance > motion_threshold and len(trail) >= min_trail_length:
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)

                x1, y1, x2, y2 = object_boxes.get(obj_id, (0, 0, 0, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"ID {obj_id} ({x}, {y}) {speed_kmph:.1f} km/h",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

                # Assign logging ID if first time
                if obj_id not in object_id_mapping:
                    object_id_mapping[obj_id] = true_next_object_id
                    true_next_object_id += 1

                # Store the data to be written later
                vehicle_data[obj_id].append((x, y, speed_kmph))

    resized = cv2.resize(frame, (1280, 720))
    cv2.imshow('vehicle detection', resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Sort and write vehicle data by ID
for obj_id in sorted(vehicle_data):
    if len(vehicle_data[obj_id]) >= 120:
        logging_id = object_id_mapping[obj_id]
        for x, y, speed in vehicle_data[obj_id]:
            output_file.write(f"{logging_id}, {x}, {y}, {speed:.2f}\n")

output_file.close()
