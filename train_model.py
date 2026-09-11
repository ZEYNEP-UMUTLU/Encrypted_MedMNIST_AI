import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# Ayarlar
# -----------------------------
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# -----------------------------
# Veri dönüşümleri
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# -----------------------------
# Veri setleri
# -----------------------------
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

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# -----------------------------
# CNN Modeli
# -----------------------------
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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
        x = self.classifier(x)
        return x


model = CNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------
# Validation
# -----------------------------
def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
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
print("\nTraining başladı...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_acc, val_f1 = evaluate(val_loader)

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Val Accuracy: {val_acc:.4f} "
        f"Val F1: {val_f1:.4f}"
    )


# -----------------------------
# Final Test
# -----------------------------
test_acc, test_f1 = evaluate(test_loader)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test F1       : {test_f1:.4f}")

# -----------------------------
# Modeli kaydet
# -----------------------------
torch.save(
    model.state_dict(),
    "pathmnist_cnn.pth"
)

print("\nModel kaydedildi: pathmnist_cnn.pth")
