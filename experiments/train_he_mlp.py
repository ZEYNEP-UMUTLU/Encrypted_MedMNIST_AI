import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score
import copy

# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("HE-COMPATIBLE POLYNOMIAL MLP")
print("=" * 70)
print("Device:", device)

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ============================================================
# DATASET
# ============================================================

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

print("Train:", len(train_dataset))
print("Val  :", len(val_dataset))

# ============================================================
# POLYNOMIAL ACTIVATION
# p(x) = 0.5x + 0.1x^2
#
# This contains only additions and multiplications,
# making it suitable for homomorphic evaluation.
# ============================================================

class PolynomialActivation(nn.Module):

    def forward(self, x):
        return 0.5 * x + 0.1 * x * x


# ============================================================
# HE-COMPATIBLE MODEL
# ============================================================

class HE_MLP(nn.Module):

    def __init__(self, num_classes=9):
        super().__init__()

        self.fc1 = nn.Linear(
            16 * 16 * 3,
            64
        )

        self.poly = PolynomialActivation()

        self.fc2 = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc1(x)

        x = self.poly(x)

        x = self.fc2(x)

        return x


model = HE_MLP().to(device)

print("\nModel:")
print(model)

# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    all_preds = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.squeeze().long().to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            total_samples += images.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    loss = total_loss / total_samples

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    return loss, acc, f1


# ============================================================
# TRAINING
# ============================================================

best_f1 = -1.0
best_state = None

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Prevent occasional polynomial gradient explosions
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        total_samples += images.size(0)

    train_loss = running_loss / total_samples

    val_loss, val_acc, val_f1 = evaluate(
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.4f} | "
        f"Val F1 {val_f1:.4f}"
    )

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_state,
            "he_mlp_best.pth"
        )

        print(
            f"  -> Best model saved "
            f"(F1={best_f1:.4f})"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(best_state)

val_loss, val_acc, val_f1 = evaluate(
    val_loader
)

print("\n" + "=" * 70)
print("BEST HE-MLP RESULT")
print("=" * 70)

print(f"Validation Accuracy : {val_acc:.4f}")
print(f"Validation Macro F1 : {val_f1:.4f}")

print("\nSaved:")
print("he_mlp_best.pth")
