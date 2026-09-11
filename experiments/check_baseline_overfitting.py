import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataset = PathMNIST(
    split="train",
    transform=transform,
    download=False
)

val_dataset = PathMNIST(
    split="val",
    transform=transform,
    download=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
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
        x = self.features(x)
        return self.classifier(x)


model = CNN().to(device)
model.load_state_dict(
    torch.load(
        "pathmnist_cnn.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()


def evaluate(loader, name):

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    print(f"{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {f1:.4f}")


print("=" * 60)
print("BASELINE OVERFITTING CHECK")
print("=" * 60)

evaluate(train_loader, "TRAIN")
evaluate(val_loader, "VAL")

print("=" * 60)
print("Tamamlandı.")
