import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score

torch.manual_seed(42)

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((16,16)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

train_ds = PathMNIST(
    split="train",
    transform=transform,
    download=False
)

val_ds = PathMNIST(
    split="val",
    transform=transform,
    download=False
)

train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=512,
    shuffle=False,
    num_workers=0
)

class HE_Linear(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.fc2(x)

        return x


model = HE_Linear().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

best_f1 = -1

for epoch in range(1, 11):

    model.train()

    total_loss = 0
    total_n = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_n += images.size(0)

    train_loss = total_loss / total_n

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.squeeze().long()

            logits = model(images)

            pred = logits.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.numpy())

    val_acc = accuracy_score(targets, preds)

    val_f1 = f1_score(
        targets,
        preds,
        average="macro"
    )

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_acc={val_acc:.4f} | "
        f"val_macroF1={val_f1:.4f}"
    )

    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            model.state_dict(),
            "he_linear_best.pth"
        )

        print(
            f"  -> best model saved "
            f"(macro-F1={best_f1:.4f})"
        )

print()
print("Training complete.")
print("Best validation Macro-F1:", best_f1)
print("Saved: he_linear_best.pth")
