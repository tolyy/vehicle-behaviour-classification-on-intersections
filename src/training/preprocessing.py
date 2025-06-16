import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CROP_LENGTH = 120
LABEL_MAP = {'left': 0, 'right': 1, 'straight': 2}

# Functions
def crop_last_n(seq, n=120):
    return seq[-n:] if len(seq) >= n else seq + [seq[-1]] * (n - len(seq))

def normalize_sequence(seq):
    arr = np.array(seq)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    return ((arr - mean) / std).tolist()

# Read and process
sequences, labels = [], []

with open("dataset/full_dataset/dataset.csv", 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        id_ = parts[0].strip()
        label = parts[1].strip().lower()
        coords_raw = parts[2:]
        coords = []
        for c in coords_raw:
            if ':' in c:
                try:
                    x, y = map(int, c.split(':'))
                    coords.append((x, y))
                except:
                    continue
        if coords:
            cropped = crop_last_n(coords, CROP_LENGTH)
            normed = normalize_sequence(cropped)
            sequences.append(normed)
            labels.append(LABEL_MAP[label])

# Convert to arrays
X = np.array(sequences, dtype=np.float32)
y = np.array(labels, dtype=np.int64)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Save for training
np.savez("vehicle_data_preprocessed.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)