import cv2
import torch
import torch.nn as nn
import numpy as np
import math
from collections import deque, defaultdict
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("yolov8m.pt")

# Define LSTM model
class TrajectoryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, 3)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h_cat)

model = TrajectoryClassifier()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Parameters
vehicle_class_ids = [2, 3, 5, 7]
max_history = 30
distance_threshold = 175
motion_threshold = 60
min_trail_length = 25
max_disappeared = 30
min_points_for_prediction = 60
lock_points = 180

raw_next_object_id = 0
true_next_object_id = 0
tracked_objects = {}
track_trails = {}
disappeared = defaultdict(int)
object_boxes = {}
object_id_mapping = {}
smoothed_positions = {}

vehicle_labels = {}
locked_ids = set()
trajectory_buffers = defaultdict(list)
initial_positions = {}
initial_directions = {}
heading_correction_flags = {}

def compute_angle_sequence(points):
    angle_sequence = [
        math.atan2(points[i+1][1] - points[i][1], points[i+1][0] - points[i][0])
        for i in range(len(points) - 1)
    ]
    angle_sequence.append(angle_sequence[-1])
    return angle_sequence

def normalize_sequence(seq):
    array = np.asarray(seq, np.float32)
    return ((array - array.mean(0)) / (array.std(0) + 1e-6)).tolist()

def normalize_angle(v):
    n = math.hypot(v[0], v[1])
    return (v[0]/n, v[1]/n) if n else (0, 0)

# !!! IMPORTANT NOTE: It is advised that if a different video is used, the parameters below should be adjusted accordingly.
# It is also advised to use a different model, such as YOLOv8n or YOLOv8s, if the video is not very complex.

# !!! IMPORTANT NOTE 2: This live classification script uses the same custom ID based tracking system as the main.py script with some modifications.
# All the parameters adjusted in the main.py script should be adjusted here as well.
cap = cv2.VideoCapture(video_path = "video_footage/sample_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, conf=0.4, verbose=False, device=0)[0]
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
                vehicle_labels.pop(obj_id, None)
                trajectory_buffers.pop(obj_id, None)
                locked_ids.discard(obj_id)
                smoothed_positions.pop(obj_id, None)
                heading_correction_flags.pop(obj_id, None)

    for i, det in enumerate(detections):
        if i not in matched_detections:
            new_tracked_objects[raw_next_object_id] = det
            new_object_boxes[raw_next_object_id] = boxes[i]
            disappeared[raw_next_object_id] = 0
            raw_next_object_id += 1

    tracked_objects = new_tracked_objects
    object_boxes = new_object_boxes

    for obj_id, (x, y) in tracked_objects.items():
        if obj_id not in track_trails:
            track_trails[obj_id] = deque(maxlen=max_history)
        trail = track_trails[obj_id]
        trail.append((x, y))

        N = 5
        prev = smoothed_positions.get(obj_id, (x, y))
        alpha = 1.0 / N
        smoothed_x = int((1 - alpha) * prev[0] + alpha * x)
        smoothed_y = int((1 - alpha) * prev[1] + alpha * y)
        smoothed_positions[obj_id] = (smoothed_x, smoothed_y)
        x, y = smoothed_x, smoothed_y

        if obj_id not in object_id_mapping:
            object_id_mapping[obj_id] = true_next_object_id
            true_next_object_id += 1

        logging_id = object_id_mapping[obj_id]
        trajectory_buffers[obj_id].append((x, y))

        if logging_id not in initial_positions:
            initial_positions[logging_id] = (x, y)
        if len(trajectory_buffers[obj_id]) == 15 and logging_id not in initial_directions:
            p0 = initial_positions[logging_id]
            p15 = (x, y)
            initial_directions[logging_id] = (p15[0] - p0[0], p15[1] - p0[1])

        if len(trajectory_buffers[obj_id]) >= min_points_for_prediction and obj_id not in locked_ids:
            if len(trajectory_buffers[obj_id]) % 5 == 0:
                points = trajectory_buffers[obj_id][-120:] if len(trajectory_buffers[obj_id]) >= 120 else trajectory_buffers[obj_id]
                angles = compute_angle_sequence(points)
                sequence = [(*xy, angle) for xy, angle in zip(points, angles)]
                normalized = normalize_sequence(sequence)
                tensor = torch.tensor([normalized], dtype=torch.float32)

                pred = model(tensor)
                label = torch.argmax(pred, dim=1).item()
                model_label = ["left", "right", "straight"][label]

                start_pt = initial_positions[logging_id]
                init_vec = initial_directions.get(logging_id, (1, 0))
                end_pt = points[-1]
                final_vec = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])

                unit_init = normalize_angle(init_vec)
                unit_final = normalize_angle(final_vec)

                dot_val = np.clip(unit_init[0]*unit_final[0] + unit_init[1]*unit_final[1], -1.0, 1.0)
                heading_change = math.degrees(math.acos(dot_val))

                if heading_change < 10:
                    final_label = "straight"
                    heading_correction_flags[obj_id] = True
                else:
                    final_label = model_label
                    heading_correction_flags[obj_id] = False

                vehicle_labels[obj_id] = final_label

            if len(trajectory_buffers[obj_id]) >= lock_points:
                locked_ids.add(obj_id)

        if len(trail) >= 2:
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-2][1] - trail[-1][1]
            distance = (dx**2 + dy**2) ** 0.5

            if distance > motion_threshold and len(trail) >= min_trail_length:
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)

                x1, y1, x2, y2 = object_boxes.get(obj_id, (0, 0, 0, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

                if obj_id in vehicle_labels:
                    color = (0, 0, 255) if obj_id in locked_ids else (0, 255, 0)
                    text = vehicle_labels[obj_id]

                    if heading_correction_flags.get(obj_id, False) and vehicle_labels[obj_id] == "straight":
                        text += " (corr.)"
                    cv2.putText(frame, text, (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    resized = cv2.resize(frame, (1280, 720))
    cv2.imshow("Live Classification", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
