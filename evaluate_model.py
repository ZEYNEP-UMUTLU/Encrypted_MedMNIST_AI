import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST, INFO
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_dataset = PathMNIST(
    split="test",
    transform=transform,
    download=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2
)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CNN().to(DEVICE)
model.load_state_dict(
    torch.load("pathmnist_cnn.pth", map_location=DEVICE)
)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.squeeze().long()

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.numpy())


info = INFO["pathmnist"]

class_names = [
    info["label"][str(i)]
    for i in range(9)
]

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

print("\n==============================")
print("BASELINE TEST RESULTS")
print("==============================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("PathMNIST CNN - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(9),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(9),
    class_names
)

for i in range(9):
    for j in range(9):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=200
)

print("\nConfusion matrix kaydedildi:")
print("confusion_matrix.png")
