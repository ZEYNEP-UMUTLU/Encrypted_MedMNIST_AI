import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_PATH = "he_linear_best.pth"

class HE_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.fc2(self.fc1(x))

transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_ds = PathMNIST(
    split="test",
    transform=transform,
    download=False
)

model = HE_Linear()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for i in range(len(test_ds)):
        image, label = test_ds[i]

        x = image.reshape(1, -1)

        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        true = int(np.asarray(label).reshape(-1)[0])

        y_true.append(true)
        y_pred.append(pred)

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

print("=" * 70)
print("HE LINEAR MODEL - FULL TEST SET")
print("=" * 70)
print(f"Test samples:       {len(test_ds)}")
print(f"Test accuracy:       {acc:.4f}")
print(f"Test Macro F1:       {macro_f1:.4f}")
print()
print(classification_report(
    y_true,
    y_pred,
    target_names=[
        "adipose",
        "background",
        "debris",
        "lymphocytes",
        "mucus",
        "smooth muscle",
        "normal colon mucosa",
        "cancer-associated stroma",
        "colorectal adenocarcinoma epithelium"
    ],
    digits=4
))
