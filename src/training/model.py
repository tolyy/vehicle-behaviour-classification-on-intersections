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

# load data
data = np.load("dataset/full_dataset/preprocessed.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]

# convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

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
class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Weighted loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

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
    print(f"Validation Accuracy: {acc:.2%}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        output = model(X_batch)
        predicted = torch.argmax(output, dim=1)
        y_true.extend(y_batch.tolist())
        y_pred.extend(predicted.tolist())

# Print classification report
print(classification_report(y_true, y_pred, target_names=["left", "right", "straight"]))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=["left", "right", "straight"], yticklabels=["left", "right", "straight"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")
