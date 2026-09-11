from medmnist import PathMNIST, INFO
import matplotlib.pyplot as plt

train = PathMNIST(split="train", download=False)

info = INFO["pathmnist"]

print("Sınıf sayısı:", len(info["label"]))
print("Sınıf isimleri:")

for k, v in info["label"].items():
    print(k, "->", v)

fig, axes = plt.subplots(3, 3, figsize=(7, 7))

for i, ax in enumerate(axes.flat):
    img, label = train[i]
    ax.imshow(img)
    ax.set_title(info["label"][str(label[0])], fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("pathmnist_samples.png", dpi=200)
print("\nÖrnek görüntüler kaydedildi: pathmnist_samples.png")

plt.show()
