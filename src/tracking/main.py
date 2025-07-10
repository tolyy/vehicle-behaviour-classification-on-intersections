import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
from tqdm import tqdm 
import math

# Set this flag to True if you want to see the debug window.
show_debug = True 

model = YOLO("yolov8m.pt")
vehicle_class_ids = [2, 3, 5, 7]  # Class IDs for vehicles
max_history = 30  # Maximum trail history
distance_threshold = 175  # Threshold for matching detections
motion_threshold = 60  # Minimum motion distance for drawing trails
min_trail_length = 25  # Minimum trail length to consider valid movement
max_disappeared = 30  # Max frames an object can disappear
repeat_coord_limit = 5  # Max consecutive identical entries allowed

# !!! IMPORTANT NOTE: It is advised that if a different video is used, the parameters below should be adjusted accordingly.
# It is also advised to use a different model, such as YOLOv8n or YOLOv8s, if the video is not very complex.
video_path = "video_footage/video.mp4"  # Path to the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

raw_next_object_id = 0  # Internal counter for raw detection IDs
true_next_object_id = 0  # Actual ID for logged objects
tracked_objects = {}  # Stores active tracked objects
track_trails = {}  # Stores trails of tracked objects
disappeared = defaultdict(int)  # Tracks how long objects have been missing
object_boxes = {}  # Stores bounding boxes of tracked objects
object_id_mapping = {}  # Maps internal IDs to logged IDs
vehicle_data = defaultdict(list)  # Store data grouped by object ID
smoothed_positions = {}  # Smoothed (x, y) positions per object


initial_positions  = {} # first (x,y)
initial_directions = {} # vector at 15th coord

# Function to normalize a vector
def norm(v):
    n = math.hypot(v[0], v[1])
    return (v[0]/n, v[1]/n) if n else (0, 0)

# Function to determine the turn label based on initial and final vectors
def turn_label(start_pt, init_vec, end_pt, angle_thresh=15):
    final_vec = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
    unit_init, unit_final = norm(init_vec), norm(final_vec)
    dot_val = np.clip(unit_init[0]*unit_final[0] + unit_init[1]*unit_final[1], -1.0, 1.0)
    heading_change = math.degrees(math.acos(dot_val))
    cross_val = -(unit_init[0]*unit_final[1] - unit_init[1]*unit_final[0])

    if heading_change < angle_thresh:
        return "straight"
    return "left" if cross_val > 0 else "right"

# Process each frame of the video
for _ in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the current frame
    results = model(frame, conf=0.4, verbose=False, device=0)[0]

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

    # Draw trails and process data
    for obj_id, (x, y) in tracked_objects.items():
        if obj_id not in track_trails:
            track_trails[obj_id] = deque(maxlen=max_history)
        trail = track_trails[obj_id]
        trail.append((x, y))

        # Apply smoothing to reduce jitter
        N = 5
        prev = smoothed_positions.get(obj_id, (x, y))
        alpha = 1.0 / N
        smoothed_x = int((1 - alpha) * prev[0] + alpha * x)
        smoothed_y = int((1 - alpha) * prev[1] + alpha * y)
        smoothed_positions[obj_id] = (smoothed_x, smoothed_y)
        x, y = smoothed_x, smoothed_y

        if len(trail) >= 2:
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-2][1] - trail[-1][1]
            distance = (dx**2 + dy**2) ** 0.5

            if distance > motion_threshold and len(trail) >= min_trail_length:
                if show_debug:
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)

                    x1, y1, x2, y2 = object_boxes.get(obj_id, (0, 0, 0, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {obj_id} ({x}, {y})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

                    # Assign logging ID if first time
                    if obj_id not in object_id_mapping:
                        object_id_mapping[obj_id] = true_next_object_id
                        true_next_object_id += 1

                    logging_id = object_id_mapping[obj_id]

                    if logging_id in initial_directions:
                        unit_init = norm(initial_directions[logging_id])
                        start_pt  = initial_positions[logging_id]
                        unit_now  = norm((x - start_pt[0], y - start_pt[1]))
                        L = 100
                        blue_tip = (int(x + unit_init[0]*L), int(y + unit_init[1]*L))
                        red_tip  = (int(x + unit_now[0]*L),  int(y + unit_now[1]*L))
                        cv2.arrowedLine(frame, (x, y), blue_tip, (255, 0, 0), 2, tipLength=0.3)
                        cv2.arrowedLine(frame, (x, y), red_tip,  (0, 0, 255), 2, tipLength=0.3)

                        # turn label text
                        lbl = turn_label(start_pt, initial_directions[logging_id], (x, y))
                        cv2.putText(frame, lbl, (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Store coordinates for the logging ID
                if len(vehicle_data[logging_id]) < repeat_coord_limit or not all((x, y) == (vx, vy) for vx, vy in vehicle_data[logging_id][-repeat_coord_limit:]):
                    vehicle_data[logging_id].append((x, y))

                # capture first pos & direction
                if logging_id not in initial_positions:
                    initial_positions[logging_id] = (x, y)
                if len(vehicle_data[logging_id]) == 15 and logging_id not in initial_directions:
                    p0 = initial_positions[logging_id]
                    p15 = (x, y)
                    initial_directions[logging_id] = (p15[0]-p0[0], p15[1]-p0[1])

    # Show window if debugging
    if show_debug:
        resized = cv2.resize(frame, (1280, 720))
        cv2.imshow('vehicle detection', resized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# It is mandatory to change the name of the csv file manually on each run to avoid overwriting the files, 
# and in a case where multiple files are generated, it is also mandatory to update the seq_id variable manually.
# The seq_id variable is used to assign a unique ID to each sequence in the CSV file.
# It is not expected by the model for it to be sequential, but it is advised for better reading of the dataset.
with open("dataset/data1.csv", "w") as csv_out:
    csv_out.write("ID,Direction,Coordinates\n")
    seq_id = 1
    for log_id in sorted(vehicle_data):
        if len(vehicle_data[log_id]) < 120:
            continue
        start_pt = initial_positions.get(log_id)
        init_vec = initial_directions.get(log_id)
        label = "unknown"
        if start_pt and init_vec:
            label = turn_label(start_pt, init_vec,vehicle_data[log_id][-1])
        coord_str = ",".join(f"{x}:{y}" for x, y in vehicle_data[log_id])
        csv_out.write(f"{seq_id},{label},{coord_str}\n")
        seq_id += 1
