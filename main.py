import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import math  # For circular mean

# Initialize the YOLO model and define parameters
model = YOLO("yolov8m.pt")
vehicle_class_ids = [2, 3, 5, 7]  # Class IDs for vehicles
max_history = 30  # Maximum trail history
distance_threshold = 200  # Threshold for matching detections
motion_threshold = 50.0  # Minimum motion distance for drawing trails
max_disappeared = 30  # Max frames an object can disappear

# Load the video file and retrieve its properties
video_path = r'C:\Users\malid\OneDrive\Belgeler\GitHub\thesis-2025\video footage\video.MP4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Estimate how many meters one pixel represents in the scene (e.g., 3.5 meters ≈ 150 pixels)
meters_per_pixel = 3.5 / 150

# Initialize tracking variables
next_object_id = 0
tracked_objects = {}  # Stores active tracked objects
track_trails = {}  # Stores trails of tracked objects
disappeared = defaultdict(int)  # Tracks how long objects have been missing
object_boxes = {}  # Stores bounding boxes of tracked objects
object_speeds = defaultdict(float)  # Stores estimated speed in km/h for each object
angle_history = defaultdict(lambda: deque(maxlen=5))  # Stores recent angles for smoothing

vehicle_data = []  # Store data to be sorted and written later

# Open a file to save the IDs and coordinates
output_file = open("vehicle_coordinates.txt", "w")
output_file.write("ID, X, Y, km/h, angle(deg)\n")

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
                angle_history.pop(obj_id, None)

    # Add new detections as tracked objects
    for i, det in enumerate(detections):
        if i not in matched_detections:
            new_tracked_objects[next_object_id] = det
            new_object_boxes[next_object_id] = boxes[i]
            disappeared[next_object_id] = 0
            next_object_id += 1

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

            angle_rad = np.arctan2(dx, dy)
            angle_deg = (np.degrees(angle_rad) + 360) % 360
            angle_history[obj_id].append(angle_deg)

            sin_sum = sum(np.sin(np.radians(a)) for a in angle_history[obj_id])
            cos_sum = sum(np.cos(np.radians(a)) for a in angle_history[obj_id])
            smoothed_angle = (np.degrees(np.arctan2(sin_sum, cos_sum)) + 360) % 360

            if distance > motion_threshold:
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)

                x1, y1, x2, y2 = object_boxes.get(obj_id, (0, 0, 0, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"ID {obj_id} ({x}, {y}) {speed_kmph:.1f} km/h {smoothed_angle:.1f} deg",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

                # Store the data to be written later
                vehicle_data.append((obj_id, x, y, speed_kmph, smoothed_angle))

    resized = cv2.resize(frame, (1280, 720))
    cv2.imshow('vehicle detection', resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Sort and write vehicle data by ID
vehicle_data.sort()
for obj_id, x, y, speed, angle in vehicle_data:
    output_file.write(f"{obj_id}, {x}, {y}, {speed:.2f}, {angle:.2f}\n")

output_file.close()
