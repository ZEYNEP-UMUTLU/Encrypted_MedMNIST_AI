import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

val_dataset = PathMNIST(
    split="val",
    transform=transform,
    download=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=2
)


class PolynomialActivation(nn.Module):
    def forward(self, x):
        return 0.5 * x + 0.1 * x * x


class HE_MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(768, 64)
        self.poly = PolynomialActivation()
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.poly(x)
        x = self.fc2(x)
        return x


model = HE_MLP().to(device)

model.load_state_dict(
    torch.load(
        "he_mlp_best.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


accuracy = accuracy_score(
    all_labels,
    all_preds
)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average="macro"
)

class_names = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium"
]

print("=" * 70)
print("HE-MLP VALIDATION EVALUATION")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("HE-MLP Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.xticks(
    range(9),
    range(9)
)

plt.yticks(
    range(9),
    range(9)
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
    "he_mlp_confusion_matrix.png",
    dpi=200
)

print("\nSaved:")
print("he_mlp_confusion_matrix.png")
