from medmnist import PathMNIST
import hashlib

print("=" * 70)
print("PATHMNIST DATA LEAKAGE / DUPLICATE CHECK")
print("=" * 70)

datasets = {}

for split in ["train", "val", "test"]:
    print(f"\nLoading {split}...")
    ds = PathMNIST(
        split=split,
        download=False
    )

    hashes = set()

    for img, label in ds:
        # PIL image -> raw RGB bytes
        data = img.tobytes()
        h = hashlib.md5(data).hexdigest()
        hashes.add(h)

    datasets[split] = hashes

    print(f"{split.upper()}: {len(hashes)} unique images / {len(ds)} total")

print("\n" + "=" * 70)
print("CROSS-SPLIT OVERLAP")
print("=" * 70)

pairs = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]

for a, b in pairs:
    overlap = datasets[a] & datasets[b]

    print(
        f"{a.upper()} ∩ {b.upper()}: "
        f"{len(overlap)} exact duplicate(s)"
    )

print("\nAnalysis tamamlandı.")
