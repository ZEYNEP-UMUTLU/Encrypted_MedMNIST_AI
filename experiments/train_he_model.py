import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# -----------------------------
# HE-uyumlu aktivasyon
# x^2
# -----------------------------
class SquareActivation(nn.Module):
    def forward(self, x):
        return x * x


# -----------------------------
# Veri
# -----------------------------
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

test_dataset = PathMNIST(
    split="test",
    transform=transform,
    download=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)


# -----------------------------
# HE-uyumlu CNN
# -----------------------------
class HE_CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            SquareActivation(),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            SquareActivation(),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            SquareActivation(),
            nn.AvgPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 3 * 3, 64),
            SquareActivation(),

            nn.Linear(64, 9)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = HE_CNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.squeeze().long()

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.numpy()
            )

            targets.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    return accuracy, f1


# -----------------------------
# Training
# -----------------------------
print("\nHE-uyumlu CNN training başladı...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.squeeze().long()

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    val_acc, val_f1 = evaluate(
        val_loader
    )

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Val Accuracy: {val_acc:.4f} "
        f"Val F1: {val_f1:.4f}"
    )


# -----------------------------
# Test
# -----------------------------
test_acc, test_f1 = evaluate(
    test_loader
)

print("\n==============================")
print("HE-CNN TEST RESULTS")
print("==============================")

print(
    f"Test Accuracy : {test_acc:.4f}"
)

print(
    f"Test F1       : {test_f1:.4f}"
)


# -----------------------------
# Model kaydet
# -----------------------------
torch.save(
    model.state_dict(),
    "he_pathmnist_cnn.pth"
)

print(
    "\nModel kaydedildi: "
    "he_pathmnist_cnn.pth"
)
