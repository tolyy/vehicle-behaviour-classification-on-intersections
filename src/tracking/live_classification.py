import cv2
import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict, deque
from ultralytics import YOLO

# load YOLO model
yolo_model = YOLO("yolov8m.pt")

# define model
class TrajectoryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3,hidden_size=128,num_layers=2,dropout=0.3,bidirectional=True,batch_first=True)
        self.fc = nn.Linear(128 * 2, 3)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x) # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # take final forward and backward hidden states
        out = self.fc(h_cat)
        return out

model = TrajectoryClassifier()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# parameters
motion_threshold = 60
min_trail_length = 25
distance_threshold = 175
min_points_for_prediction = 30
raw_next_object_id = 0
tracked_objects = {}
trajectory_buffers = defaultdict(lambda: deque(maxlen=120))
disappeared = defaultdict(int)
max_disappeared = 30
vehicle_labels = {}
vehicle_class_ids = [2, 3, 5, 7]

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

cap = cv2.VideoCapture(r"C:\Users\malid\Desktop\thesis_footage\iyi_gibi\202207\20220704_0730\20220704_084254_Koh_Dor_4W_d_1_3_org.MP4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, verbose=False, device=0)[0]
    detections = []

    # filter detections for vehicles
    for box in results.boxes:
        if int(box.cls[0]) not in vehicle_class_ids:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detections.append((cx, cy, x1, y1, x2, y2))

    matched_ids = set()
    new_tracked_objects = {}

    # Match detections to existing objects
    for obj_id, (prev_x, prev_y) in tracked_objects.items():
        min_dist = float('inf')
        best_idx = -1

        for i, (cx, cy, _, _, _, _) in enumerate(detections):
            if i in matched_ids:
                continue
            dist = np.linalg.norm(np.array([prev_x, prev_y]) - np.array([cx, cy]))
            if dist < min_dist and dist < distance_threshold:
                min_dist = dist
                best_idx = i

        if best_idx != -1:
            matched_ids.add(best_idx)
            cx, cy, x1, y1, x2, y2 = detections[best_idx]
            new_tracked_objects[obj_id] = (cx, cy)

            buffer = trajectory_buffers[obj_id]
            buffer.append((cx, cy))

            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check for label assignment
            if obj_id not in vehicle_labels and len(buffer) >= min_points_for_prediction:
                points = list(buffer)[-120:] if len(buffer) >= 120 else list(buffer)
                angles = compute_angle_sequence(points)
                sequence = [(*xy, angle) for xy, angle in zip(points, angles)]
                normalized = normalize_sequence(sequence)
                tensor = torch.tensor([normalized], dtype=torch.float32)

                pred = model(tensor)
                label = torch.argmax(pred, dim=1).item()
                label_text = ["left", "right", "straight"][label]
                vehicle_labels[obj_id] = label_text # Assign label to object

            # Draw label
            if obj_id in vehicle_labels:
                label_text = vehicle_labels[obj_id]
                cv2.putText(frame, label_text, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Handle new detections
    for i, (cx, cy, x1, y1, x2, y2) in enumerate(detections):
        if i not in matched_ids:
            tracked_objects[raw_next_object_id] = (cx, cy)
            new_tracked_objects[raw_next_object_id] = (cx, cy)
            trajectory_buffers[raw_next_object_id].append((cx, cy))
            raw_next_object_id += 1

            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {raw_next_object_id - 1}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update disappeared counters
    disappeared_ids = set(tracked_objects.keys()) - set(new_tracked_objects.keys())
    for obj_id in disappeared_ids:
        disappeared[obj_id] += 1
        if disappeared[obj_id] > max_disappeared:
            tracked_objects.pop(obj_id, None)
            trajectory_buffers.pop(obj_id, None)
            vehicle_labels.pop(obj_id, None)
            disappeared.pop(obj_id, None)

    tracked_objects = new_tracked_objects

    resize = cv2.resize(frame, (1280, 720))
    cv2.imshow("live classification", resize)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
