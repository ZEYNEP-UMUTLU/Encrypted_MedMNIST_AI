from medmnist import PathMNIST
from collections import Counter
import numpy as np

splits = ["train", "val", "test"]

print("=" * 70)
print("PATHMNIST DATASET ANALYSIS")
print("=" * 70)

for split in splits:

    dataset = PathMNIST(
        split=split,
        download=False
    )

    labels = []

    for _, label in dataset:
        labels.append(int(label[0]))

    counts = Counter(labels)
    total = len(labels)

    print(f"\n{split.upper()}")
    print("-" * 40)

    for cls in range(9):
        count = counts[cls]
        percentage = 100 * count / total

        print(
            f"Class {cls}: "
            f"{count:6d} samples "
            f"({percentage:6.2f}%)"
        )

    print(f"Total: {total}")

    values = np.array([
        counts[i] for i in range(9)
    ])

    print(
        f"Max/Min ratio: "
        f"{values.max() / values.min():.2f}"
    )

print("\nAnalysis tamamlandı.")
