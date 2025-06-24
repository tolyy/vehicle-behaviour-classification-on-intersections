import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# paths / constants
CSV_PATH = "dataset/full_dataset/dataset.csv" # raw input
NPZ_OUTPUT = "dataset/full_dataset/preprocessed.npz" # output
CROP_LENGTH = 120 # frames per sample
LABEL_TO_INT = {"left": 0, "right": 1, "straight": 2}

# helpers
def crop_last_n_points(points, n=CROP_LENGTH):
    return points[-n:] if len(points) >= n else points + [points[-1]] * (n - len(points))

def compute_angle_sequence(points):
    angle_sequence = [math.atan2(points[i+1][1] - points[i][1], points[i+1][0] - points[i][0]) for i in range(len(points) - 1)]
    angle_sequence.append(angle_sequence[-1]) # pad to match length
    return angle_sequence

def normalize_sequence(sequence):
    array = np.asarray(sequence, np.float32)
    return ((array - array.mean(0)) / (array.std(0) + 1e-6)).tolist()

print("loading csv …")
all_sequences = []
all_labels = []

with open(CSV_PATH) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        label_text = parts[1].strip().lower()
        if label_text not in LABEL_TO_INT:
            continue

        coordinates = []
        for coord_pair in parts[2:]:
            if ':' in coord_pair:
                try:
                    x, y = map(int, coord_pair.split(':'))
                    coordinates.append((x, y))
                except ValueError:
                    continue
        if not coordinates:
            continue

        cropped = crop_last_n_points(coordinates)
        angles = compute_angle_sequence(cropped)
        sequence_with_angle = [(*xy, angle) for xy, angle in zip(cropped, angles)]
        normalized = normalize_sequence(sequence_with_angle)

        all_sequences.append(normalized)
        all_labels.append(LABEL_TO_INT[label_text])

print(f"parsed {len(all_sequences)} samples")

X = np.array(all_sequences, dtype=np.float32)
y = np.array(all_labels, dtype=np.int64)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

np.savez(NPZ_OUTPUT,X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

print("saved", NPZ_OUTPUT)
print("shapes - train", X_train.shape, "val", X_val.shape, "test", X_test.shape)
