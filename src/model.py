import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Load preprocessed data
data = np.load("dataset/full_dataset/preprocessed.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define model
class TrajectoryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, 3)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # last forward and backward hidden states
        return self.fc(h_cat)

model = TrajectoryClassifier()
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

val_accuracies = []
train_losses = []

# Train
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} | Training Loss: {avg_loss:.4f}")

    # Validation accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            predicted = preds.argmax(dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    acc = correct / total
    val_accuracies.append(acc)
    print(f"Validation Accuracy: {acc:.2%}")

# Plot graph
plt.figure()
epochs = range(1, len(val_accuracies) + 1)
plt.plot(epochs, val_accuracies, marker='o', color='blue', label='Validation Accuracy')
plt.plot(epochs, train_losses, marker='x', color='red', label='Training Loss')
plt.title("Validation Accuracy & Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.savefig("accuracy_loss_over_epochs.png")
print("Accuracy and loss plot saved as accuracy_loss_over_epochs.png")

# Save model
torch.save(model.state_dict(), "models/model.pth")
print("Model saved as model.pth")

# Test set evaluation
model.eval()
y_true_test = []
y_pred_test = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        predicted = torch.argmax(output, dim=1)
        y_true_test.extend(y_batch.tolist())
        y_pred_test.extend(predicted.tolist())

print("Test Set Performance")
print(classification_report(y_true_test, y_pred_test, target_names=["left", "right", "straight"]))

# Confusion matrix
cm = confusion_matrix(y_true_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["left", "right", "straight"], yticklabels=["left", "right", "straight"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix on Test Set")
plt.savefig("confusion_matrix_test.png")
print("Confusion matrix plot saved as confusion_matrix_test.png")
